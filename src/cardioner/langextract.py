"""
LangExtract-based NER utilities for CardioNER.

This module provides:
- `parse_examples`: convert CardioNER-style JSON examples into LangExtract examples
- `extract`: run LLM-based extraction with a merge/extract/unwind strategy
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import langextract as lx

from cardioner.llm import lx_extract, lx_prompts


@dataclass
class _SpanMap:
    doc_idx: int
    start: int  # global start (inclusive)
    end: int  # global end (exclusive)


def _load_json_or_path(example_json: str) -> Any:
    """Load JSON payload from either a JSON string or a file path."""
    if not example_json:
        return None

    candidate = Path(example_json)
    if candidate.exists() and candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8"))

    return json.loads(example_json)


def _normalize_docs(payload: Any) -> list[dict]:
    """
    Normalize multiple possible payload shapes into a list of CardioNER-style docs.

    Accepted shapes:
    - [ {id, text, tags}, ... ]
    - { "examples": [ ... ] }
    - { "data": [ ... ] }
    - {id, text, tags}
    """
    if payload is None:
        return []

    if isinstance(payload, list):
        return [d for d in payload if isinstance(d, dict)]

    if isinstance(payload, dict):
        if isinstance(payload.get("examples"), list):
            return [d for d in payload["examples"] if isinstance(d, dict)]
        if isinstance(payload.get("data"), list):
            return [d for d in payload["data"] if isinstance(d, dict)]
        if "text" in payload:
            return [payload]

    raise ValueError("Unsupported example JSON structure.")


def _tag_to_extraction(text: str, tag: dict) -> lx.data.Extraction | None:
    """Convert one CardioNER tag dict to one LangExtract Extraction."""
    if not isinstance(tag, dict):
        return None

    label = tag.get("tag")
    start = tag.get("start")
    end = tag.get("end")

    if not isinstance(label, str):
        return None
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end <= start or end > len(text):
        return None

    extraction_text = text[start:end]
    return lx.data.Extraction(
        extraction_class=label,
        extraction_text=extraction_text,
        char_interval=lx.data.CharInterval(start_pos=start, end_pos=end),
    )


def parse_examples(
    example_json: str, examples: list[dict] | None = None
) -> list[lx.data.ExampleData]:
    """
    Parse CardioNER-style JSON examples into LangExtract ExampleData objects.

    CardioNER-style document:
    {
      "id": "doc-1",
      "text": "...",
      "tags": [{"start": 10, "end": 18, "tag": "DISEASE"}, ...]
    }

    Args:
        example_json: JSON string or path to a JSON file.
        examples: Optional additional docs in the same structure.

    Returns:
        List of `lx.data.ExampleData`.
    """
    docs: list[dict] = []
    if example_json:
        payload = _load_json_or_path(example_json)
        docs.extend(_normalize_docs(payload))
    if examples:
        docs.extend([d for d in examples if isinstance(d, dict)])

    parsed: list[lx.data.ExampleData] = []
    for doc in docs:
        text = doc.get("text", "")
        tags = doc.get("tags", [])

        if not isinstance(text, str):
            continue
        if not isinstance(tags, list):
            tags = []

        extractions: list[lx.data.Extraction] = []
        for tag in tags:
            ex = _tag_to_extraction(text, tag)
            if ex is not None:
                extractions.append(ex)

        parsed.append(lx.data.ExampleData(text=text, extractions=extractions))

    return parsed


def _resolve_prompt_description() -> str:
    """Best-effort prompt resolution from `cardioner.llm.lx_prompts`."""
    # direct constants
    for attr in (
        "PROMPT_DESCRIPTION",
        "prompt_description",
        "DEFAULT_PROMPT",
        "DEFAULT_PROMPT_DESCRIPTION",
    ):
        value = getattr(lx_prompts, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # callables
    for attr in ("get_prompt", "build_prompt", "prompt"):
        fn = getattr(lx_prompts, attr, None)
        if callable(fn):
            value = fn()
            if isinstance(value, str) and value.strip():
                return value.strip()

    # fallback
    return (
        "Extract named entities from the text. "
        "Return exact spans and the correct entity class."
    )


def _resolve_extract_kwargs() -> dict[str, Any]:
    """Best-effort extraction defaults from `cardioner.llm.lx_extract` + env."""
    kwargs: dict[str, Any] = {}

    model_id = getattr(lx_extract, "MODEL_ID", None)
    if isinstance(model_id, str) and model_id.strip():
        kwargs["model_id"] = model_id.strip()

    api_key = getattr(lx_extract, "API_KEY", None)
    if not api_key:
        api_key = os.getenv("LANGEXTRACT_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key

    temperature = getattr(lx_extract, "TEMPERATURE", None)
    if isinstance(temperature, (int, float)):
        kwargs["temperature"] = float(temperature)

    extraction_passes = getattr(lx_extract, "EXTRACTION_PASSES", None)
    if isinstance(extraction_passes, int) and extraction_passes > 0:
        kwargs["extraction_passes"] = extraction_passes

    model_url = getattr(lx_extract, "MODEL_URL", None)
    if isinstance(model_url, str) and model_url.strip():
        kwargs["model_url"] = model_url.strip()

    language_model_params = getattr(lx_extract, "LANGUAGE_MODEL_PARAMS", None)
    if isinstance(language_model_params, dict):
        kwargs["language_model_params"] = language_model_params

    return kwargs


def _build_merged_text(txts: Sequence[str]) -> tuple[str, list[_SpanMap]]:
    """Merge texts into one large text and return global offset mapping."""
    merged_parts: list[str] = []
    spans: list[_SpanMap] = []

    cursor = 0
    for i, txt in enumerate(txts):
        if i > 0:
            separator = f"\n\n<<<DOC_BREAK_{i}>>>\n\n"
            merged_parts.append(separator)
            cursor += len(separator)

        start = cursor
        merged_parts.append(txt)
        cursor += len(txt)
        spans.append(_SpanMap(doc_idx=i, start=start, end=cursor))

    return "".join(merged_parts), spans


def _locate_doc_span(
    global_start: int, global_end: int, span_map: Sequence[_SpanMap]
) -> tuple[int, int, int] | None:
    """Map global span to (doc_idx, local_start, local_end)."""
    for m in span_map:
        if global_start >= m.start and global_end <= m.end:
            return (m.doc_idx, global_start - m.start, global_end - m.start)
    return None


def extract(
    txts: list[str],
    Examples: list[lx.data.ExampleData] | lx.data.ExampleData | None,
    batch_size: int = 128,
    batch_len_max: int = 64_000,
) -> list[dict]:
    """
    Extract NER entities with LangExtract using merge -> extract -> unwind.

    Args:
        txts: List of raw documents.
        Examples: LangExtract example(s), usually output of `parse_examples`.
        batch_size: Parallel workers for extraction calls.
        batch_len_max: Max character buffer sent to model per extraction chunk.

    Returns:
        List of extracted entities in CardioNER inference-like shape:
        {
          "filename": "<doc-index>",
          "label": "<entity-label>",
          "start_span": <int>,
          "end_span": <int>,
          "text": "<entity-text>"
        }
    """
    clean_txts = [t if isinstance(t, str) else "" for t in txts]
    if not clean_txts:
        return []

    if Examples is None:
        examples: list[lx.data.ExampleData] = []
    elif isinstance(Examples, list):
        examples = Examples
    else:
        examples = [Examples]

    merged_text, span_map = _build_merged_text(clean_txts)
    prompt_description = _resolve_prompt_description()
    provider_kwargs = _resolve_extract_kwargs()

    annotated = lx.extract(
        merged_text,
        prompt_description=prompt_description,
        examples=examples,
        max_char_buffer=batch_len_max,
        max_workers=max(1, int(batch_size)),
        **provider_kwargs,
    )

    # API may return a single AnnotatedDocument or a list
    annotated_docs = annotated if isinstance(annotated, list) else [annotated]

    results: list[dict] = []
    seen: set[tuple[int, str, int, int]] = set()

    for adoc in annotated_docs:
        extractions = getattr(adoc, "extractions", None) or []
        for ex in extractions:
            label = getattr(ex, "extraction_class", None)
            if not isinstance(label, str) or not label:
                continue

            char_interval = getattr(ex, "char_interval", None)
            if (
                char_interval is None
                or getattr(char_interval, "start_pos", None) is None
                or getattr(char_interval, "end_pos", None) is None
            ):
                continue

            g_start = int(char_interval.start_pos)
            g_end = int(char_interval.end_pos)
            if g_end <= g_start:
                continue

            mapped = _locate_doc_span(g_start, g_end, span_map)
            if mapped is None:
                # span may cross merged separators or be invalid
                continue

            doc_idx, local_start, local_end = mapped
            text = clean_txts[doc_idx]
            if local_start < 0 or local_end > len(text) or local_end <= local_start:
                continue

            entity_text = text[local_start:local_end]
            key = (doc_idx, label, local_start, local_end)
            if key in seen:
                continue
            seen.add(key)

            results.append(
                {
                    "filename": str(doc_idx),
                    "label": label,
                    "start_span": local_start,
                    "end_span": local_end,
                    "text": entity_text,
                }
            )

    return results


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="LangExtract NER extraction with CardioNER-style examples."
    )
    parser.add_argument(
        "--texts_json",
        type=str,
        required=True,
        help="JSON string or file path containing a list of strings.",
    )
    parser.add_argument(
        "--examples_json",
        type=str,
        default="",
        help="JSON string or file path with CardioNER-style example docs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Parallel workers for extraction.",
    )
    parser.add_argument(
        "--batch_len_max",
        type=int,
        default=64_000,
        help="Maximum char buffer per extraction chunk.",
    )
    args = parser.parse_args()

    txt_payload = _load_json_or_path(args.texts_json)
    if not isinstance(txt_payload, list) or not all(
        isinstance(x, str) for x in txt_payload
    ):
        raise ValueError("--texts_json must resolve to a JSON list of strings.")

    ex = parse_examples(args.examples_json) if args.examples_json else []
    out = extract(
        txts=txt_payload,
        Examples=ex,
        batch_size=args.batch_size,
        batch_len_max=args.batch_len_max,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
