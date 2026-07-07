from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from cardioner.predictor import PredictionNER
except ImportError:
    # Fallback for direct script execution from src/cardioner
    from predictor import PredictionNER


def _coalesce_for_masking(tags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge overlapping spans to avoid leaking text while masking.

    If overlapping spans have different tags, merged tag becomes "MASK".
    """
    if not tags:
        return []

    sorted_tags = sorted(tags, key=lambda t: (int(t["start"]), -int(t["end"])))
    merged: list[dict[str, Any]] = []

    for tag in sorted_tags:
        start = int(tag["start"])
        end = int(tag["end"])
        label = str(tag.get("tag", "MASK"))

        if not merged or start >= merged[-1]["end"]:
            merged.append({"start": start, "end": end, "tag": label})
            continue

        prev = merged[-1]
        prev["end"] = max(int(prev["end"]), end)
        if str(prev.get("tag")) != label:
            prev["tag"] = "MASK"

    return merged


def _repeat_to_length(token: str, length: int) -> str:
    if length <= 0:
        return ""
    if not token:
        token = "*"
    n_repeats = (length // len(token)) + 1
    return (token * n_repeats)[:length]


def mask_text(
    text: str,
    tags: list[dict[str, Any]],
    mask_token: str = "[MASK]",
    use_tag_as_mask: bool = False,
    preserve_length: bool = False,
) -> str:
    """
    Mask entity spans in text.

    Args:
        text: Original text.
        tags: Span dicts with keys: start, end, tag.
        mask_token: Replacement token for masked spans.
        use_tag_as_mask: If True, replacement is e.g. "[DISEASE]".
        preserve_length: If True, preserve masked span length by repeating mask_token.

    Returns:
        Masked text.
    """
    if not text or not tags:
        return text

    merged_tags = _coalesce_for_masking(tags)

    out: list[str] = []
    cursor = 0

    for tag in merged_tags:
        start = max(0, int(tag["start"]))
        end = min(len(text), int(tag["end"]))
        if end <= start:
            continue

        out.append(text[cursor:start])

        if preserve_length:
            replacement = _repeat_to_length(mask_token, end - start)
        elif use_tag_as_mask:
            replacement = f"[{tag.get('tag', 'MASK')}]"
        else:
            replacement = mask_token

        out.append(replacement)
        cursor = end

    out.append(text[cursor:])
    return "".join(out)


def _normalize_input(
    texts: str | Sequence[str] | Sequence[dict[str, Any]],
    ids: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Normalize accepted input shapes into [{"id": ..., "text": ...}, ...]."""
    if isinstance(texts, str):
        doc_id = ids[0] if ids and len(ids) > 0 else "doc-0"
        return [{"id": str(doc_id), "text": texts}]

    text_list = list(texts)
    if not text_list:
        return []

    if all(isinstance(t, str) for t in text_list):
        records: list[dict[str, str]] = []
        for i, txt in enumerate(text_list):
            doc_id = ids[i] if ids and i < len(ids) else f"doc-{i}"
            records.append({"id": str(doc_id), "text": txt})
        return records

    if all(isinstance(t, dict) for t in text_list):
        records = []
        for i, row in enumerate(text_list):
            text = row.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            doc_id = row.get("id", f"doc-{i}")
            records.append({"id": str(doc_id), "text": text})
        return records

    raise ValueError("Input must be str, list[str], or list[dict with 'text']")


def _predict_tags_for_text(
    ner: PredictionNER,
    text: str,
    batch_size: int,
    confidence_threshold: float,
    post_hoc_cleaning: bool,
    o_confidence_threshold: float,
    trim_trailing_cutoff_words: bool | None,
) -> list[dict[str, Any]]:
    raw_preds = ner.do_prediction_batch(
        text=text,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold,
        post_hoc_cleaning=post_hoc_cleaning,
        o_confidence_threshold=o_confidence_threshold,
        trim_trailing_cutoff_words=trim_trailing_cutoff_words,
    )

    # Deduplicate exact spans while preserving highest score if present.
    best_by_key: dict[tuple[int, int, str], float] = {}
    for pred in raw_preds:
        start = int(pred["start"])
        end = int(pred["end"])
        tag = str(pred.get("tag", pred.get("entity", "")))
        if not tag or end <= start:
            continue

        score = float(pred.get("score", 0.0))
        key = (start, end, tag)
        if key not in best_by_key or score > best_by_key[key]:
            best_by_key[key] = score

    tags = [
        {"start": k[0], "end": k[1], "tag": k[2]}
        for k in sorted(best_by_key.keys(), key=lambda x: (x[0], x[1], x[2]))
    ]
    return tags


def extract_spans(
    ner: PredictionNER,
    texts: str | Sequence[str] | Sequence[dict[str, Any]],
    ids: Sequence[str] | None = None,
    return_masked_text: bool = False,
    mask_token: str = "[MASK]",
    use_tag_as_mask: bool = False,
    preserve_mask_length: bool = False,
    batch_size: int = 8,
    confidence_threshold: float = 0.6,
    post_hoc_cleaning: bool = True,
    o_confidence_threshold: float = 0.7,
    trim_trailing_cutoff_words: bool | None = None,
) -> list[dict[str, Any]]:
    """
    Extract entity spans for one or multiple texts.

    Output shape mirrors CardioNER-style JSONL rows:
    {
      "id": "...",
      "text": "...",
      "tags": [{"start": int, "end": int, "tag": str}, ...],
      "masked_text": "..."   # optional
    }
    """
    records = _normalize_input(texts, ids=ids)
    outputs: list[dict[str, Any]] = []

    for row in records:
        text = row["text"]
        tags = _predict_tags_for_text(
            ner=ner,
            text=text,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
            post_hoc_cleaning=post_hoc_cleaning,
            o_confidence_threshold=o_confidence_threshold,
            trim_trailing_cutoff_words=trim_trailing_cutoff_words,
        )

        out: dict[str, Any] = {"id": row["id"], "text": text, "tags": tags}

        if return_masked_text:
            out["masked_text"] = mask_text(
                text=text,
                tags=tags,
                mask_token=mask_token,
                use_tag_as_mask=use_tag_as_mask,
                preserve_length=preserve_mask_length,
            )

        outputs.append(out)

    return outputs


def write_jsonl(records: Sequence[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonlike_file(path: str | Path) -> list[dict[str, str]]:
    """
    Load JSON/JSONL file into list[{id, text}].

    Accepted JSON payloads:
      - [{"id": "...", "text": "..."}, ...]
      - {"id": "...", "text": "..."}
      - ["text1", "text2", ...]

    Accepted JSONL rows:
      - {"id": "...", "text": "..."}
      - "just text"
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    suffix = p.suffix.lower()

    if suffix == ".jsonl":
        records: list[dict[str, str]] = []
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    text = row.get("text", "")
                    doc_id = row.get("id", f"doc-{i}")
                    records.append({"id": str(doc_id), "text": str(text)})
                elif isinstance(row, str):
                    records.append({"id": f"doc-{i}", "text": row})
                else:
                    raise ValueError(
                        f"Unsupported JSONL row type at line {i + 1}: {type(row)}"
                    )
        return records

    if suffix == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "text" in payload:
                return [
                    {
                        "id": str(payload.get("id", "doc-0")),
                        "text": str(payload["text"]),
                    }
                ]
            raise ValueError("JSON dict input must include at least a 'text' field.")

        if isinstance(payload, list):
            records = []
            for i, row in enumerate(payload):
                if isinstance(row, dict):
                    text = row.get("text", "")
                    doc_id = row.get("id", f"doc-{i}")
                    records.append({"id": str(doc_id), "text": str(text)})
                elif isinstance(row, str):
                    records.append({"id": f"doc-{i}", "text": row})
                else:
                    raise ValueError(
                        f"Unsupported JSON list row type at index {i}: {type(row)}"
                    )
            return records

    raise ValueError("Input file must be .json or .jsonl")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract CardioNER spans per text and optionally return masked text. "
            "Supports one text or multiple texts."
        )
    )

    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument(
        "--lang",
        default="nl",
        choices=["es", "nl", "en", "it", "ro", "sv", "cz", "multi"],
    )
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--stride", default=250, type=int)
    parser.add_argument("--overlap", default=0, type=int)

    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Input text. Repeat --text for multiple texts.",
    )
    parser.add_argument(
        "--id",
        action="append",
        default=None,
        help="Optional doc id. Repeat to match --text occurrences.",
    )
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="Optional .json or .jsonl file with texts (supports id/text rows).",
    )

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--confidence_threshold", default=0.6, type=float)
    parser.add_argument("--o_confidence_threshold", default=0.7, type=float)
    parser.add_argument(
        "--no_post_hoc_cleaning",
        action="store_true",
        default=False,
        help="Disable post-hoc span cleaning (enabled by default).",
    )
    parser.add_argument(
        "--trim_trailing_cutoff_words", action="store_true", default=False
    )

    parser.add_argument("--mask", action="store_true", default=False)
    parser.add_argument("--mask_token", default="[MASK]", type=str)
    parser.add_argument("--mask_with_tag", action="store_true", default=False)
    parser.add_argument("--preserve_mask_length", action="store_true", default=False)

    parser.add_argument(
        "--output_jsonl",
        default=None,
        type=str,
        help="If set, write one JSON object per line to this path.",
    )
    parser.add_argument(
        "--pretty_print",
        action="store_true",
        default=False,
        help="Pretty-print JSON list to stdout.",
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.text and not args.input_file:
        raise ValueError(
            "Provide input via --text (repeatable) or --input_file (.json/.jsonl)"
        )

    input_records: list[dict[str, str]] = []
    if args.input_file:
        input_records.extend(_load_jsonlike_file(args.input_file))

    if args.text:
        input_records.extend(_normalize_input(args.text, ids=args.id))

    ner = PredictionNER(
        model_checkpoint=args.model_path,
        revision=args.revision,
        stride=args.stride,
        overlap=args.overlap,
        device=args.device,
        lang=args.lang,
    )

    results = extract_spans(
        ner=ner,
        texts=input_records,
        return_masked_text=args.mask,
        mask_token=args.mask_token,
        use_tag_as_mask=args.mask_with_tag,
        preserve_mask_length=args.preserve_mask_length,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        o_confidence_threshold=args.o_confidence_threshold,
        post_hoc_cleaning=not args.no_post_hoc_cleaning,
        trim_trailing_cutoff_words=args.trim_trailing_cutoff_words,
    )

    if args.output_jsonl:
        write_jsonl(results, args.output_jsonl)

    if args.pretty_print:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
