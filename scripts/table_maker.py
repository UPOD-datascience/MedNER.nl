#!/usr/bin/env python3
"""Build LaTeX tables from MultiClinAI JSON result files.

The script scans a directory recursively for JSON files, infers:
- language (Czech, Dutch, English, Italian, Romanian, Spanish, Swedish)
- model (CardioDeBERTa, EuroBERT-610m)
- category (DISEASE, PROCEDURE, SYMPTOM)

and extracts STRICT metrics as micro (macro) values per metric:
- F1
- Recall
- Precision

Output is LaTeX tables matching the requested format.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

LANGUAGE_ORDER = [
    "Czech",
    "Dutch",
    "English",
    "Italian",
    "Romanian",
    "Spanish",
    "Swedish",
]

MODEL_ORDER = ["CardioDeBERTa", "EuroBERT-610m"]
CATEGORY_ORDER = ["DISEASE", "PROCEDURE", "SYMPTOM"]
METRICS = ["F1", "Recall", "Precision"]

LANGUAGE_CODE_MAP = {
    "cz": "Czech",
    "nl": "Dutch",
    "en": "English",
    "it": "Italian",
    "ro": "Romanian",
    "es": "Spanish",
    "sv": "Swedish",
}


def _infer_from_filename(
    path: Path,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Infer (language, model, category) from result filename tokens.

    Expected examples:
    - DISEASE_DeBERTa_multiclass_cz_testsequence_result.json
    - PROCEDURE_EuroBERT610_multiclass_en_test_sequence_result.json
    """
    stem = path.stem  # without .json
    parts = [p.lower() for p in re.split(r"[_\-]+", stem) if p]

    # Category is usually the first token.
    category = None
    if parts:
        first = parts[0].upper()
        if first in CATEGORY_ORDER:
            category = first

    model = None
    if "deberta" in parts:
        model = "CardioDeBERTa"
    elif "eurobert610" in parts or "eurobert" in parts:
        model = "EuroBERT-610m"

    language = None
    for token in parts:
        if token in LANGUAGE_CODE_MAP:
            language = LANGUAGE_CODE_MAP[token]
            break

    return language, model, category


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        # aggregated outputs often look like {"mean": ..., "std": ...}
        if "mean" in value:
            return _to_float(value["mean"])
    return None


def _get_metric_triplet_micro_macro(
    data: dict,
    match_type: str,
) -> Optional[dict[str, tuple[float, float]]]:
    block = data.get(match_type)
    if not isinstance(block, dict):
        return None

    micro = block.get("micro", {})
    macro = block.get("macro", {})
    if not isinstance(micro, dict) or not isinstance(macro, dict):
        return None

    out: dict[str, tuple[float, float]] = {}
    for metric in METRICS:
        mic = _to_float(micro.get(metric))
        mac = _to_float(macro.get(metric))
        if mic is None or mac is None:
            return None
        out[metric] = (mic, mac)

    return out


def _format_micro_macro(pair: Optional[tuple[float, float]], ndigits: int = 2) -> str:
    if pair is None:
        return "-"
    micro, macro = pair
    return f"{micro:.{ndigits}f} ({macro:.{ndigits}f})"


def _latex_escape(text: str) -> str:
    # Minimal escaping for labels used in table text.
    return text.replace("_", r"\_")


def build_tables(input_dir: Path, match_type: str = "strict", ndigits: int = 2) -> str:
    """Create LaTeX tables for DISEASE/PROCEDURE/SYMPTOM from JSON results."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # table_data[category][language][model][metric] = (micro, macro)
    table_data: dict[str, dict[str, dict[str, dict[str, tuple[float, float]]]]] = {
        c: {l: {m: {} for m in MODEL_ORDER} for l in LANGUAGE_ORDER}
        for c in CATEGORY_ORDER
    }

    json_files = sorted(input_dir.rglob("*sequence*result*.json"))
    for json_path in json_files:
        language, model, category = _infer_from_filename(json_path)
        if not (language and model and category):
            continue

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        metrics = _get_metric_triplet_micro_macro(payload, match_type=match_type)
        if metrics is None:
            continue

        table_data[category][language][model] = metrics

    blocks = []
    caption_map = {
        "DISEASE": f"End-to-end span \\textit{{{match_type}}} performance for the DISEASE category, micro (macro) on hold-out set",
        "PROCEDURE": f"End-to-end span \\textit{{{match_type}}} performance for the PROCEDURE category, micro (macro) on hold-out set",
        "SYMPTOM": f"End-to-end span \\textit{{{match_type}}} performance for the SYMPTOM category, micro (macro) on hold-out set",
    }
    label_map = {
        "DISEASE": "tab:dataset3",
        "PROCEDURE": "tab:dataset4",
        "SYMPTOM": "tab:dataset5",
    }

    for category in CATEGORY_ORDER:
        lines = []
        lines.append(r"\begin{table}")
        lines.append(r"    \centering")
        lines.append(r"    \begin{tabular}{cc|ccc}")
        lines.append(
            r"  category                  & Base model              & F1 & Recall & Precision\\"
        )
        lines.append(r"    \hline")

        for language in LANGUAGE_ORDER:
            for idx, model in enumerate(MODEL_ORDER):
                f1 = _format_micro_macro(
                    table_data[category][language][model].get("F1"), ndigits=ndigits
                )
                rec = _format_micro_macro(
                    table_data[category][language][model].get("Recall"), ndigits=ndigits
                )
                pre = _format_micro_macro(
                    table_data[category][language][model].get("Precision"),
                    ndigits=ndigits,
                )

                if idx == 0:
                    lines.append(
                        f"   \\multirow{{2}}{{*}}{{{_latex_escape(language)}}} & {model:<23} & {f1} & {rec} & {pre}  \\\\"
                    )
                else:
                    lines.append(
                        f"                            & {model:<23} & {f1} & {rec} & {pre}  \\\\"
                    )

            lines.append(r"    \hline")

        lines.append(r"    \end{tabular}")
        lines.append(f"    \\caption{{{caption_map[category]}}}")
        lines.append(f"    \\label{{{label_map[category]}}}")
        lines.append(r"\end{table}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from strict micro/macro JSON results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/media/bramiozo/Storage1/bramiozo/DEV/GIT/UMCU/CardioNER.nl/output/MultiClinAI/multiclineai_results_deberta_eurobert"
        ),
        help="Directory containing JSON result files (searched recursively).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output .tex file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--match-type",
        choices=["strict", "relaxed"],
        default="strict",
        help="Which matching block to use from JSON results.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimals for metric formatting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latex = build_tables(
        args.input_dir, match_type=args.match_type, ndigits=args.digits
    )

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(latex + "\n", encoding="utf-8")
        print(f"Wrote LaTeX tables to: {args.output_file}")
    else:
        print(latex)


if __name__ == "__main__":
    main()
