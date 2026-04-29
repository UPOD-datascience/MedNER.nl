#!/usr/bin/env python3
"""
Inspect JSONL training data for NER-style span annotations.

Computes:
- Number of documents
- Number of tags per tag label
- Average and median span length per tag label
- If "lang" exists on any record, aggregates by language and tag.
  Records without a valid lang are grouped under "unknown".

Expected JSONL format per line (example):
{
  "id": "...",
  "text": "...",
  "lang": "nl",
  "tags": [{"start": 10, "end": 15, "tag": "DISEASE"}, ...]
}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple, Union, cast


UNKNOWN_LANG = "unknown"

TagStats = Dict[str, float]
FlatStats = Dict[str, TagStats]
NestedStats = Dict[str, Dict[str, TagStats]]
OutputTags = Union[FlatStats, NestedStats]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect JSONL training data and report tag/span statistics."
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print output as JSON instead of a text table.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}: {exc}"
                ) from exc


def inspect_training_jsonl(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    document_count = 0
    has_any_lang = False

    # Used when no lang is present in data
    tag_counts: Counter[str] = Counter()
    span_lengths_by_tag: Dict[str, List[int]] = defaultdict(list)

    # Used when lang is present in data (with unknown bucket for missing/invalid lang)
    nested_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    nested_span_lengths: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for line_number, record in iter_jsonl(path):
        document_count += 1

        raw_lang = record.get("lang")
        lang_value = raw_lang if isinstance(raw_lang, str) and raw_lang else None
        if lang_value is not None:
            has_any_lang = True

        tags = record.get("tags", [])
        if tags is None:
            tags = []

        if not isinstance(tags, list):
            raise ValueError(
                f"Line {line_number}: 'tags' must be a list, got {type(tags).__name__}"
            )

        for i, ann in enumerate(tags):
            if not isinstance(ann, dict):
                raise ValueError(
                    f"Line {line_number}, tag index {i}: annotation must be an object."
                )

            tag_label = ann.get("tag")
            start = ann.get("start")
            end = ann.get("end")

            if not isinstance(tag_label, str) or not tag_label:
                raise ValueError(
                    f"Line {line_number}, tag index {i}: missing/invalid 'tag'."
                )
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(
                    f"Line {line_number}, tag index {i}: 'start' and 'end' must be integers."
                )
            if end < start:
                raise ValueError(
                    f"Line {line_number}, tag index {i}: 'end' ({end}) < 'start' ({start})."
                )

            length = end - start

            # Keep flat counts for the no-lang case
            tag_counts[tag_label] += 1
            span_lengths_by_tag[tag_label].append(length)

            # Always collect nested using unknown fallback; used when any lang is present
            nested_lang = lang_value if lang_value is not None else UNKNOWN_LANG
            nested_counts[nested_lang][tag_label] += 1
            nested_span_lengths[nested_lang][tag_label].append(length)

    if has_any_lang:
        tags_output: OutputTags = {}
        for lang in sorted(nested_counts):
            tags_output[lang] = {}
            for tag in sorted(nested_counts[lang]):
                lengths = nested_span_lengths[lang][tag]
                avg_length = sum(lengths) / len(lengths) if lengths else 0.0
                med_length = float(median(lengths)) if lengths else 0.0
                tags_output[lang][tag] = {
                    "count": nested_counts[lang][tag],
                    "avg_span_length": avg_length,
                    "median_span_length": med_length,
                }
    else:
        tags_output = {}
        for tag in sorted(tag_counts):
            lengths = span_lengths_by_tag[tag]
            avg_length = sum(lengths) / len(lengths) if lengths else 0.0
            med_length = float(median(lengths)) if lengths else 0.0
            tags_output[tag] = {
                "count": tag_counts[tag],
                "avg_span_length": avg_length,
                "median_span_length": med_length,
            }

    return {
        "documents": document_count,
        "tags": tags_output,
    }


def _is_nested_tags(tags: OutputTags) -> bool:
    if not tags:
        return False
    first_value = next(iter(tags.values()))
    return isinstance(first_value, dict) and "count" not in first_value


def print_text_report(stats: Dict[str, object]) -> None:
    print(f"Documents: {stats['documents']}")

    tags = cast(OutputTags, stats["tags"])
    if not tags:
        print("No tags found.")
        return

    if _is_nested_tags(tags):
        nested_tags = cast(NestedStats, tags)
        print("\nPer-language, per-tag statistics:")
        print(
            f"{'Lang':<12} {'Tag':<20} {'Count':>10} {'Avg span len':>15} {'Median span len':>17}"
        )
        print("-" * 82)
        for lang, tag_map in nested_tags.items():
            for tag, values in tag_map.items():
                print(
                    f"{lang:<12} "
                    f"{tag:<20} "
                    f"{values['count']:>10.0f} "
                    f"{values['avg_span_length']:>15.2f} "
                    f"{values['median_span_length']:>17.2f}"
                )
    else:
        flat_tags = cast(FlatStats, tags)
        print("\nPer-tag statistics:")
        print(f"{'Tag':<20} {'Count':>10} {'Avg span len':>15} {'Median span len':>17}")
        print("-" * 65)
        for tag, values in flat_tags.items():
            print(
                f"{tag:<20} "
                f"{values['count']:>10.0f} "
                f"{values['avg_span_length']:>15.2f} "
                f"{values['median_span_length']:>17.2f}"
            )


def main() -> None:
    args = parse_args()
    stats = inspect_training_jsonl(args.jsonl_path)

    if args.as_json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print_text_report(stats)


if __name__ == "__main__":
    main()
