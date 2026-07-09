#!/usr/bin/env python3
"""Merge per-entity NER JSON files by choosing a canonical text per document.

Pipeline:
1) Group records by normalized document id.
2) If all text variants are identical, merge tags directly (same as ner_caster.collect_jsons).
3) If text variants differ, evaluate each variant as candidate canonical anchor:
   - realign spans from other variants by
     a) exact positional match (same start/end and same span text),
     b) fallback contextual similarity using a 30-char window and
        SequenceMatcher score > 0.6.
   - select the anchor retaining the most annotations.
4) Deduplicate merged tags and write JSONL output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PRUNED_FROM_IDS = ["disease", "procedure", "symptom", "medication"]
DEFAULT_ENTITY_MAP = {
    "ENFERMEDAD": "DISEASE",
    "FARMACO": "MEDICATION",
    "SINTOMA": "SYMPTOM",
    "PROCEDIMIENTO": "PROCEDURE",
}


@dataclass
class Tag:
    start: int
    end: int
    tag: str


@dataclass
class Record:
    source_file: str
    raw_id: str
    normalized_id: str
    text: str
    tags: list[Tag]


@dataclass
class MergeResult:
    canonical_text: str
    tags: list[Tag]
    retained_raw: int
    dropped_raw: int
    exact_position_hits: int
    sequence_match_hits: int
    context_hits: int
    direct_text_matches: int
    mean_similarity: float
    strategy: str


class IndexMapper:
    """Approximate source->target index mapping using SequenceMatcher blocks."""

    def __init__(self, source_text: str, target_text: str) -> None:
        self.source_text = source_text
        self.target_text = target_text
        self.blocks = SequenceMatcher(
            None, source_text, target_text, autojunk=False
        ).get_matching_blocks()

    def map_index(self, source_index: int) -> int | None:
        if source_index < 0:
            return 0

        for block in self.blocks:
            if block.a <= source_index < block.a + block.size:
                return block.b + (source_index - block.a)

        previous = None
        for block in self.blocks:
            if block.a + block.size <= source_index:
                previous = block
            else:
                break

        if previous is not None:
            return (
                previous.b
                + previous.size
                + (source_index - (previous.a + previous.size))
            )

        for block in self.blocks:
            if block.a > source_index:
                return block.b - (block.a - source_index)

        return None


def normalize_id(raw_id: str, prune_terms: list[str]) -> str:
    out = str(raw_id)
    for term in prune_terms:
        out = out.replace(term, "")
        out = out.replace(term.lower(), "")
        out = out.replace(term.upper(), "")
        out = out.replace(term.capitalize(), "")

    while "--" in out:
        out = out.replace("--", "-")
    out = out.strip("-_ ")
    return out if out else str(raw_id)


def iter_json_files(json_dir: Path, input_glob: str) -> list[Path]:
    files = [p for p in sorted(json_dir.glob(input_glob)) if p.is_file()]
    files = [p for p in files if p.suffix.lower() in {".json", ".jsonl"}]
    return files


def load_json_objects(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []

    # Try full JSON first (object or list), then JSONL fallback.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [obj for obj in parsed if isinstance(obj, dict)]
    except json.JSONDecodeError:
        pass

    out: list[dict[str, Any]] = []
    for i, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at line {i}: {exc}") from exc
        if isinstance(obj, dict):
            out.append(obj)
    return out


def to_tag(tag_obj: Any) -> Tag | None:
    if not isinstance(tag_obj, dict):
        return None

    try:
        start = int(tag_obj["start"])
        end = int(tag_obj["end"])
        label = str(tag_obj["tag"])
    except Exception:
        return None

    if start < 0 or end < 0 or end < start:
        return None
    if not label:
        return None

    return Tag(start=start, end=end, tag=label)


def map_tag_label(label: str, entity_map: dict[str, str] | None) -> str:
    if entity_map is None:
        return label
    return entity_map.get(label, entity_map.get(label.upper(), label))


def dedupe_tags(tags: list[Tag]) -> list[Tag]:
    seen = set()
    out: list[Tag] = []
    for tag in sorted(tags, key=lambda t: (t.start, t.end, t.tag)):
        key = (tag.tag, tag.start, tag.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def find_all_occurrences(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    starts: list[int] = []
    cursor = 0
    while True:
        idx = haystack.find(needle, cursor)
        if idx == -1:
            break
        starts.append(idx)
        cursor = idx + 1
    return starts


def align_span(
    source_text: str,
    anchor_text: str,
    start: int,
    end: int,
    context_window: int,
    similarity_threshold: float,
    mapper: IndexMapper | None,
    max_candidates: int = 200,
) -> tuple[tuple[int, int] | None, str, float]:
    """Align one span from source_text to anchor_text.

    The primary strategy is to search the anchor text around the estimated original
    location for the best matching substring using SequenceMatcher.

    Returns: (aligned_span, method, score)
      method in {"exact_position", "sequence_match", "context_similarity", "invalid", "unaligned"}
    """
    if start < 0 or end > len(source_text) or start >= end:
        return None, "invalid", 0.0

    span_text = source_text[start:end]
    if not span_text:
        return None, "invalid", 0.0

    estimated_start = mapper.map_index(start) if mapper is not None else start
    if estimated_start is None:
        estimated_start = start

    max_start = max(0, len(anchor_text) - len(span_text))
    estimated_start = max(0, min(max_start, estimated_start))

    # 1) Exact substring match in anchor text.
    exact_occurrences = find_all_occurrences(anchor_text, span_text)
    if exact_occurrences:
        if len(exact_occurrences) > max_candidates:
            exact_occurrences = sorted(
                exact_occurrences,
                key=lambda s: abs(s - estimated_start),
            )[:max_candidates]
        best_occurrence = min(
            exact_occurrences,
            key=lambda s: abs(s - estimated_start),
        )
        return (
            (
                best_occurrence,
                best_occurrence + len(span_text),
            ),
            "exact_position",
            1.0,
        )

    # 2) SequenceMatcher-based search around the expected location.
    radius = max(context_window, len(span_text) * 2)
    min_span_length = max(1, len(span_text) - max(5, len(span_text) // 4))
    max_span_length = len(span_text) + max(5, len(span_text) // 4)

    best_span: tuple[int, int] | None = None
    best_score = 0.0

    start_positions = range(
        max(0, estimated_start - radius),
        min(len(anchor_text), estimated_start + radius + 1),
    )
    for s in start_positions:
        for span_length in range(min_span_length, max_span_length + 1):
            e = s + span_length
            if e > len(anchor_text):
                continue
            candidate_text = anchor_text[s:e]
            if not candidate_text:
                continue
            score = SequenceMatcher(
                None, span_text, candidate_text, autojunk=False
            ).ratio()
            if score > best_score:
                best_score = score
                best_span = (s, e)

    if best_span is not None and best_score > similarity_threshold:
        return best_span, "sequence_match", best_score

    # 3) Contextual fallback (kept as a last resort).
    source_ctx = source_text[
        max(0, start - context_window) : min(len(source_text), end + context_window)
    ]
    candidates: list[tuple[int, int]] = []
    for s in range(
        max(0, estimated_start - radius),
        min(len(anchor_text), estimated_start + radius + 1),
    ):
        for span_length in range(min_span_length, max_span_length + 1):
            e = s + span_length
            if e > len(anchor_text):
                continue
            candidates.append((s, e))

    if not candidates:
        return None, "unaligned", 0.0

    best_context_span: tuple[int, int] | None = None
    best_context_score = 0.0
    for s, e in candidates:
        anchor_ctx = anchor_text[
            max(0, s - context_window) : min(len(anchor_text), e + context_window)
        ]
        context_score = SequenceMatcher(
            None, source_ctx, anchor_ctx, autojunk=False
        ).ratio()
        if context_score > best_context_score:
            best_context_score = context_score
            best_context_span = (s, e)

    if best_context_span is not None and best_context_score > similarity_threshold:
        return best_context_span, "context_similarity", best_context_score

    return None, "unaligned", best_score


def merge_for_anchor(
    records: list[Record],
    anchor_text: str,
    context_window: int,
    similarity_threshold: float,
) -> MergeResult:
    merged_tags: list[Tag] = []
    retained_raw = 0
    dropped_raw = 0
    exact_position_hits = 0
    sequence_match_hits = 0
    context_hits = 0

    direct_text_matches = sum(1 for r in records if r.text == anchor_text)
    sims = [
        SequenceMatcher(None, anchor_text, r.text, autojunk=False).ratio()
        for r in records
    ]
    mean_similarity = sum(sims) / len(sims) if sims else 0.0

    mapper_cache: dict[str, IndexMapper] = {}

    for rec in records:
        mapper: IndexMapper | None = None
        if rec.text != anchor_text:
            mapper = mapper_cache.get(rec.text)
            if mapper is None:
                mapper = IndexMapper(rec.text, anchor_text)
                mapper_cache[rec.text] = mapper

        for tag in rec.tags:
            aligned, method, _score = align_span(
                source_text=rec.text,
                anchor_text=anchor_text,
                start=tag.start,
                end=tag.end,
                context_window=context_window,
                similarity_threshold=similarity_threshold,
                mapper=mapper,
            )
            if aligned is None:
                dropped_raw += 1
                continue

            retained_raw += 1
            if method == "exact_position":
                exact_position_hits += 1
            elif method == "sequence_match":
                sequence_match_hits += 1
            elif method == "context_similarity":
                context_hits += 1

            merged_tags.append(Tag(start=aligned[0], end=aligned[1], tag=tag.tag))

    deduped = dedupe_tags(merged_tags)
    return MergeResult(
        canonical_text=anchor_text,
        tags=deduped,
        retained_raw=retained_raw,
        dropped_raw=dropped_raw,
        exact_position_hits=exact_position_hits,
        sequence_match_hits=sequence_match_hits,
        context_hits=context_hits,
        direct_text_matches=direct_text_matches,
        mean_similarity=mean_similarity,
        strategy="canonical",
    )


def merge_records_for_id(
    records: list[Record],
    context_window: int,
    similarity_threshold: float,
) -> MergeResult:
    if not records:
        return MergeResult(
            canonical_text="",
            tags=[],
            retained_raw=0,
            dropped_raw=0,
            exact_position_hits=0,
            sequence_match_hits=0,
            context_hits=0,
            direct_text_matches=0,
            mean_similarity=0.0,
            strategy="empty",
        )

    unique_texts = list(dict.fromkeys(r.text for r in records))

    if len(unique_texts) == 1:
        result = merge_for_anchor(
            records=records,
            anchor_text=unique_texts[0],
            context_window=context_window,
            similarity_threshold=similarity_threshold,
        )
        result.strategy = "identical"
        return result

    candidate_results = [
        merge_for_anchor(
            records=records,
            anchor_text=candidate,
            context_window=context_window,
            similarity_threshold=similarity_threshold,
        )
        for candidate in unique_texts
    ]

    # Select anchor retaining the highest number of annotations.
    # Tie-breakers: more unique merged tags, more direct text matches, then higher mean similarity.
    best = max(
        candidate_results,
        key=lambda r: (
            r.retained_raw,
            len(r.tags),
            r.direct_text_matches,
            r.mean_similarity,
        ),
    )
    best.strategy = "canonical"
    return best


def load_records(
    json_files: list[Path],
    prune_terms: list[str],
) -> tuple[dict[str, list[Record]], dict[str, int]]:
    records_by_id: dict[str, list[Record]] = defaultdict(list)
    skipped = {
        "objects_without_id": 0,
        "objects_without_text": 0,
        "invalid_tags": 0,
    }

    for file_path in json_files:
        objects = load_json_objects(file_path)
        for obj in objects:
            raw_id = obj.get("id")
            if raw_id is None or str(raw_id).strip() == "":
                skipped["objects_without_id"] += 1
                continue

            text = obj.get("text")
            if not isinstance(text, str):
                if text is None:
                    skipped["objects_without_text"] += 1
                    continue
                text = str(text)

            normalized = normalize_id(str(raw_id), prune_terms)
            tags_raw = obj.get("tags", [])
            if not isinstance(tags_raw, list):
                tags_raw = []

            tags: list[Tag] = []
            for tag_obj in tags_raw:
                tag = to_tag(tag_obj)
                if tag is None:
                    skipped["invalid_tags"] += 1
                    continue
                tags.append(tag)

            records_by_id[normalized].append(
                Record(
                    source_file=file_path.name,
                    raw_id=str(raw_id),
                    normalized_id=normalized,
                    text=text,
                    tags=tags,
                )
            )

    return records_by_id, skipped


def parse_entity_map(raw: str | None, disable: bool) -> dict[str, str] | None:
    if disable:
        return None
    if raw is None:
        return DEFAULT_ENTITY_MAP

    # First: inline JSON string. Second: path to JSON file.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass

    candidate_path = Path(raw)
    if candidate_path.exists() and candidate_path.is_file():
        parsed = json.loads(candidate_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("entity_map JSON file must contain an object/dict")
        return {str(k): str(v) for k, v in parsed.items()}

    raise ValueError("--entity_map must be valid inline JSON or a path to a JSON file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-entity NER JSON/JSONL files by selecting one canonical text per id "
            "and realigning spans when text variants differ."
        )
    )
    parser.add_argument(
        "--json_dir",
        type=Path,
        required=True,
        help="Directory with per-entity JSON/JSONL files (e.g. dis/proc/symp outputs).",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        required=True,
        help="Output merged JSONL file path.",
    )
    parser.add_argument(
        "--input_glob",
        default="*.json*",
        help="Glob for input files inside --json_dir (default: *.json*).",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=30,
        help="Context window size (in characters) used for fallback alignment (default: 30).",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.6,
        help="Fallback context similarity threshold; must score > threshold (default: 0.6).",
    )
    parser.add_argument(
        "--max_text_variants",
        type=int,
        default=None,
        help=(
            "Optional safeguard: fail fast when a normalized id has more than this "
            "many unique text variants."
        ),
    )
    parser.add_argument(
        "--prune_terms",
        nargs="+",
        default=PRUNED_FROM_IDS,
        help=(
            "Terms removed from ids before merging (default: disease procedure symptom medication)."
        ),
    )
    parser.add_argument(
        "--entity_map",
        type=str,
        default=None,
        help=(
            "Tag mapping as inline JSON or path to JSON file. "
            "If omitted, uses ner_caster defaults."
        ),
    )
    parser.add_argument(
        "--disable_entity_map",
        action="store_true",
        help="Disable tag mapping and keep input labels unchanged.",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=None,
        help="Optional path to write a JSON report with merge/alignment statistics.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-document merge decisions for divergent texts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.context_window < 0:
        print("Error: --context_window must be >= 0", file=sys.stderr)
        sys.exit(1)

    if args.similarity_threshold < 0 or args.similarity_threshold > 1:
        print("Error: --similarity_threshold must be in [0, 1]", file=sys.stderr)
        sys.exit(1)

    if args.max_text_variants is not None and args.max_text_variants < 1:
        print("Error: --max_text_variants must be >= 1", file=sys.stderr)
        sys.exit(1)

    json_dir: Path = args.json_dir.resolve()
    out_path: Path = args.out_path.resolve()

    if not json_dir.is_dir():
        print(f"Error: --json_dir is not a directory: {json_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        entity_map = parse_entity_map(args.entity_map, args.disable_entity_map)
    except Exception as exc:
        print(f"Error parsing entity map: {exc}", file=sys.stderr)
        sys.exit(1)

    json_files = iter_json_files(json_dir, args.input_glob)
    if not json_files:
        print(
            f"Error: no .json/.jsonl files found in {json_dir} for glob '{args.input_glob}'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Merging files from: {json_dir}")
    print(f"Input files      : {len(json_files)}")
    for f in json_files:
        print(f"  - {f.name}")
    print(f"Context window   : {args.context_window}")
    print(f"Sim threshold    : > {args.similarity_threshold}")
    print(
        "Max text variants: "
        f"{args.max_text_variants if args.max_text_variants is not None else 'disabled'}"
    )
    print()

    records_by_id, skipped = load_records(
        json_files=json_files, prune_terms=args.prune_terms
    )

    variant_counts = {
        doc_id: len(set(r.text for r in records))
        for doc_id, records in records_by_id.items()
    }

    if args.max_text_variants is not None:
        offenders = [
            (doc_id, variant_counts[doc_id])
            for doc_id in sorted(variant_counts.keys())
            if variant_counts[doc_id] > args.max_text_variants
        ]
        if offenders:
            print(
                "Error: found ids exceeding --max_text_variants "
                f"({args.max_text_variants}).",
                file=sys.stderr,
            )
            for doc_id, count in offenders[:20]:
                print(f"  - id={doc_id} variants={count}", file=sys.stderr)
            if len(offenders) > 20:
                print(
                    f"  ... and {len(offenders) - 20} more ids",
                    file=sys.stderr,
                )
            sys.exit(1)

    total_docs = len(records_by_id)
    identical_docs = 0
    canonical_docs = 0

    total_input_tags = 0
    total_retained_raw = 0
    total_dropped_raw = 0
    total_exact_hits = 0
    total_sequence_hits = 0
    total_context_hits = 0

    merged_rows: list[dict[str, Any]] = []
    per_doc_report: list[dict[str, Any]] = []

    doc_ids = sorted(records_by_id.keys())
    show_progress = total_docs > 0

    for idx, doc_id in enumerate(doc_ids, start=1):
        if show_progress:
            print(f"\rProcessing texts: {idx}/{total_docs}", end="", flush=True)

        records = records_by_id[doc_id]
        total_input_tags += sum(len(r.tags) for r in records)

        result = merge_records_for_id(
            records=records,
            context_window=args.context_window,
            similarity_threshold=args.similarity_threshold,
        )

        if result.strategy == "identical":
            identical_docs += 1
        elif result.strategy == "canonical":
            canonical_docs += 1

        mapped_tags = [
            Tag(start=t.start, end=t.end, tag=map_tag_label(t.tag, entity_map))
            for t in result.tags
        ]
        mapped_tags = dedupe_tags(mapped_tags)

        merged_rows.append(
            {
                "id": doc_id,
                "tags": [
                    {"start": t.start, "end": t.end, "tag": t.tag} for t in mapped_tags
                ],
                "text": result.canonical_text,
            }
        )

        total_retained_raw += result.retained_raw
        total_dropped_raw += result.dropped_raw
        total_exact_hits += result.exact_position_hits
        total_sequence_hits += result.sequence_match_hits
        total_context_hits += result.context_hits

        unique_text_variants = variant_counts[doc_id]
        if args.verbose and unique_text_variants > 1:
            if show_progress:
                print()
            print(
                f"[canonical] id={doc_id} variants={unique_text_variants} "
                f"retained={result.retained_raw} dropped={result.dropped_raw} "
                f"dedup_tags={len(mapped_tags)}"
            )
            if show_progress:
                print(f"\rProcessing texts: {idx}/{total_docs}", end="", flush=True)

        per_doc_report.append(
            {
                "id": doc_id,
                "num_records": len(records),
                "num_text_variants": unique_text_variants,
                "strategy": result.strategy,
                "retained_raw": result.retained_raw,
                "dropped_raw": result.dropped_raw,
                "dedup_tags": len(mapped_tags),
                "exact_position_hits": result.exact_position_hits,
                "sequence_match_hits": result.sequence_match_hits,
                "context_hits": result.context_hits,
            }
        )

    if show_progress:
        print()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tag_counts: dict[str, int] = defaultdict(int)
    for row in merged_rows:
        for tag in row["tags"]:
            tag_counts[str(tag["tag"])] += 1

    print()
    print("Merge summary")
    print("-------------")
    print(f"Documents                : {total_docs}")
    print(f"Identical-text merges    : {identical_docs}")
    print(f"Canonical realign merges : {canonical_docs}")
    print(f"Input tags (raw)         : {total_input_tags}")
    print(f"Retained tags (raw)      : {total_retained_raw}")
    print(f"Dropped tags (raw)       : {total_dropped_raw}")
    print(f"Exact position hits      : {total_exact_hits}")
    print(f"SequenceMatcher hits     : {total_sequence_hits}")
    print(f"Context fallback hits    : {total_context_hits}")
    print(f"Output rows              : {len(merged_rows)}")
    print(f"Output file              : {out_path}")

    print("\nOutput tag counts:")
    for label, count in sorted(tag_counts.items()):
        print(f"  {label}: {count}")

    if any(skipped.values()):
        print("\nSkipped/cleaned input items:")
        for k, v in skipped.items():
            print(f"  {k}: {v}")

    if args.report_path is not None:
        report_path = args.report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_obj = {
            "summary": {
                "documents": total_docs,
                "identical_docs": identical_docs,
                "canonical_docs": canonical_docs,
                "input_tags_raw": total_input_tags,
                "retained_tags_raw": total_retained_raw,
                "dropped_tags_raw": total_dropped_raw,
                "exact_position_hits": total_exact_hits,
                "sequence_match_hits": total_sequence_hits,
                "context_hits": total_context_hits,
                "output_rows": len(merged_rows),
                "skipped": skipped,
            },
            "per_document": per_doc_report,
        }
        report_path.write_text(
            json.dumps(report_obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Report file              : {report_path}")


if __name__ == "__main__":
    main()
