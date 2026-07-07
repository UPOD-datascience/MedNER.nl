from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import gradio as gr

FORCE_LIGHT_JS = """
() => {
  const forceLight = () => {
    const root = document.documentElement;
    root.classList.remove('dark');
    root.classList.add('light');
  };

  forceLight();

  const observer = new MutationObserver(() => forceLight());
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class'],
  });
}
"""

LIGHT_MODE_CSS = """
button[title*='theme'],
button[aria-label*='theme'],
button[title*='Theme'],
button[aria-label*='Theme'] {
  display: none !important;
}
"""

COLOR_PALETTE = [
    "#ffe6e6",
    "#e6f4ff",
    "#e9fbe5",
    "#fff4db",
    "#efe8ff",
    "#e6fff8",
    "#fde6ff",
    "#f1f1f1",
]

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "MultiClinNER_combined"
    / "MultiClinNER-nl"
)


def list_annotation_files(data_dir: str) -> list[str]:
    base = Path(data_dir)
    if not base.exists() or not base.is_dir():
        return []

    files = [
        p.name
        for p in base.iterdir()
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
    ]
    return sorted(files)


def _load_json_objects(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try full JSON first: either a list[dict] or one dict.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [obj for obj in parsed if isinstance(obj, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Fallback to JSON Lines.
    records: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def _normalize_tags(tags: list[dict[str, Any]], text_len: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tag in tags:
        try:
            start = int(tag.get("start", -1))
            end = int(tag.get("end", -1))
        except (TypeError, ValueError):
            continue

        label = str(tag.get("tag", "UNKNOWN"))
        if start < 0:
            start = 0
        if end > text_len:
            end = text_len
        if start >= end:
            continue

        normalized.append({"start": start, "end": end, "tag": label})

    # Sort by start, then longest first (helps with overlaps).
    normalized.sort(key=lambda x: (x["start"], -(x["end"] - x["start"]), x["end"]))
    return normalized


def _labels_per_character(
    text_len: int, tags: list[dict[str, Any]]
) -> tuple[list[str | None], int]:
    labels: list[str | None] = [None] * text_len
    overlap_conflicts = 0

    for span in tags:
        label = span["tag"]
        for i in range(span["start"], span["end"]):
            if labels[i] is None:
                labels[i] = label
            elif labels[i] != label:
                overlap_conflicts += 1

    return labels, overlap_conflicts


def _segment_text(text: str, labels: list[str | None]) -> list[tuple[str, str | None]]:
    if not text:
        return []

    segments: list[tuple[str, str | None]] = []
    start = 0
    current = labels[0]

    for idx in range(1, len(text)):
        if labels[idx] != current:
            segments.append((text[start:idx], current))
            start = idx
            current = labels[idx]
    segments.append((text[start:], current))

    return segments


def _build_color_map(tags: list[str]) -> dict[str, str]:
    return {tag: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, tag in enumerate(tags)}


def render_record_html(record: dict[str, Any]) -> tuple[str, list[list[Any]], str]:
    text = str(record.get("text", ""))
    tags_raw = record.get("tags", [])
    if not isinstance(tags_raw, list):
        tags_raw = []

    tags = _normalize_tags(tags_raw, len(text))
    labels, overlap_conflicts = _labels_per_character(len(text), tags)
    segments = _segment_text(text, labels)

    unique_tags = sorted({t["tag"] for t in tags})
    color_map = _build_color_map(unique_tags)

    html_parts: list[str] = [
        '<div style="font-family: sans-serif;">',
        '<div style="margin-bottom: 10px; font-size: 0.95rem;">',
        "<b>Legend:</b> ",
    ]

    if unique_tags:
        for tag in unique_tags:
            color = color_map[tag]
            html_parts.append(
                (
                    '<span style="display:inline-block; margin-right:8px; '
                    "padding:2px 8px; border-radius:6px; "
                    f'background:{color};">{html.escape(tag)}</span>'
                )
            )
    else:
        html_parts.append("<i>No tags available for this record.</i>")

    html_parts.append("</div>")
    html_parts.append(
        '<div style="white-space: pre-wrap; line-height: 1.6; font-size: 1rem;">'
    )

    for chunk, label in segments:
        escaped = html.escape(chunk)
        if label is None:
            html_parts.append(escaped)
        else:
            color = color_map.get(label, "#f1f1f1")
            html_parts.append(
                (
                    '<span style="padding: 0.08em 0.2em; border-radius: 0.25em; '
                    f'background:{color};" title="{html.escape(label)}">{escaped}</span>'
                )
            )

    html_parts.append("</div></div>")

    counts = Counter(t["tag"] for t in tags)
    rows = [[tag, count] for tag, count in sorted(counts.items())]

    warning = ""
    if overlap_conflicts:
        warning = (
            "⚠️ Detected overlapping spans with different tags. "
            "The earliest-applied label is shown in overlapping characters."
        )

    return "".join(html_parts), rows, warning


def refresh_files(data_dir: str) -> tuple[gr.Dropdown, str]:
    files = list_annotation_files(data_dir)
    value = files[0] if files else None
    status = f"Found {len(files)} annotation file(s)."
    if not files:
        status += " Check that the directory exists and contains .json/.jsonl files."

    return gr.Dropdown(choices=files, value=value), status


def load_file(
    data_dir: str, filename: str | None
) -> tuple[list[dict[str, Any]], gr.Dropdown, str, str, list[list[Any]], str, str]:
    if not filename:
        return (
            [],
            gr.Dropdown(choices=[], value=None),
            "No file selected.",
            "",
            [],
            "",
            "",
        )

    path = Path(data_dir) / filename
    if not path.exists():
        msg = f"File not found: {path}"
        return [], gr.Dropdown(choices=[], value=None), msg, "", [], "", ""

    records = sorted(
        _load_json_objects(path),
        key=lambda rec: str(rec.get("id", "")).casefold(),
    )
    choices = [f"{i}: {str(rec.get('id', '<no-id>'))}" for i, rec in enumerate(records)]
    first_choice = choices[0] if choices else None

    status = f"Loaded {len(records)} record(s) from `{filename}`."
    if not records:
        return records, gr.Dropdown(choices=[], value=None), status, "", [], "", ""

    first_record = records[0]
    rendered_html, table_rows, warning = render_record_html(first_record)
    record_id = str(first_record.get("id", ""))

    return (
        records,
        gr.Dropdown(choices=choices, value=first_choice),
        status,
        record_id,
        table_rows,
        rendered_html,
        warning,
    )


def render_selected(
    selected_record: str | None, records: list[dict[str, Any]]
) -> tuple[str, list[list[Any]], str, str]:
    if not records or not selected_record:
        return "", [], "", ""

    try:
        index = int(selected_record.split(":", 1)[0])
    except (ValueError, IndexError):
        return "", [], "", ""

    if index < 0 or index >= len(records):
        return "", [], "", ""

    record = records[index]
    rendered_html, table_rows, warning = render_record_html(record)
    record_id = str(record.get("id", ""))

    return record_id, table_rows, rendered_html, warning


def build_app(data_dir: str | Path = DEFAULT_DATA_DIR) -> gr.Blocks:
    data_dir = str(data_dir)
    initial_files = list_annotation_files(data_dir)

    with gr.Blocks(
        title="NER Annotation Inspector",
        theme=gr.themes.Default(),
        css=LIGHT_MODE_CSS,
        js=FORCE_LIGHT_JS,
    ) as app:
        gr.Markdown(
            """
            # NER Annotation Inspector
            Select an annotation file and inspect record-level spans with color-coded tags.
            Supported formats: JSONL (one JSON object per line) and JSON (object or list).
            """
        )

        state_records = gr.State([])

        with gr.Row():
            data_dir_box = gr.Textbox(
                label="Annotation directory",
                value=data_dir,
                scale=5,
            )
            refresh_button = gr.Button("Refresh files", scale=1)

        with gr.Row():
            file_dropdown = gr.Dropdown(
                label="Annotation file (.json/.jsonl)",
                choices=initial_files,
                value=initial_files[0] if initial_files else None,
                scale=4,
            )
            load_button = gr.Button("Load file", scale=1)

        status_md = gr.Markdown()

        record_dropdown = gr.Dropdown(label="Record", choices=[], value=None)
        record_id = gr.Textbox(label="Record ID", interactive=False)
        warning_md = gr.Markdown()

        with gr.Row():
            tag_counts = gr.Dataframe(
                headers=["tag", "count"],
                datatype=["str", "number"],
                label="Tag counts in selected record",
                interactive=False,
                wrap=True,
                scale=1,
            )

        rendered = gr.HTML(label="Highlighted text")

        refresh_button.click(
            fn=refresh_files,
            inputs=[data_dir_box],
            outputs=[file_dropdown, status_md],
        )

        for trigger in (load_button.click, file_dropdown.change):
            trigger(
                fn=load_file,
                inputs=[data_dir_box, file_dropdown],
                outputs=[
                    state_records,
                    record_dropdown,
                    status_md,
                    record_id,
                    tag_counts,
                    rendered,
                    warning_md,
                ],
            )

        record_dropdown.change(
            fn=render_selected,
            inputs=[record_dropdown, state_records],
            outputs=[record_id, tag_counts, rendered, warning_md],
        )

        app.load(
            fn=load_file,
            inputs=[data_dir_box, file_dropdown],
            outputs=[
                state_records,
                record_dropdown,
                status_md,
                record_id,
                tag_counts,
                rendered,
                warning_md,
            ],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect NER annotations with Gradio")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory with annotation .json/.jsonl files",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.data_dir)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
