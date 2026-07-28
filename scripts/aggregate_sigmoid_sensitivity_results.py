#!/usr/bin/env python3
"""Aggregate sigmoid sensitivity performance summaries per model.

This script searches for `aggregated_fold_on_fold_results.json` files under an
input root and writes a model-level CSV summary.

It supports two layouts:
1) Preferred (one aggregated file per model):
   output/<LANG>_SIGMOID_SENSITIVITY/<MODEL>/aggregated_fold_on_fold_results.json
2) Legacy/per-fold layout (one aggregated file per fold with n_folds=1):
   output/<LANG>_SIGMOID_SENSITIVITY/<MODEL>/fold_X/aggregated_fold_on_fold_results.json

If multiple files are found for one model, the script will:
- prefer the file with the highest `_metadata.n_folds` (>1), otherwise
- recompute a model-level mean/std across discovered files for that model.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from pathlib import Path
from typing import Any

MATCH_TYPES = ("strict", "relaxed")
SCOPES = ("micro", "macro")
METRICS = ("Precision", "Recall", "F1")


def _to_mean_std(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, dict):
        mean = value.get("mean")
        std = value.get("std", 0.0)
        if isinstance(mean, (int, float)):
            return float(mean), float(std) if isinstance(std, (int, float)) else 0.0
        return None, None
    if isinstance(value, (int, float)):
        return float(value), 0.0
    return None, None


def _extract_flat_metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    flat: dict[str, float | None] = {}
    for match in MATCH_TYPES:
        for scope in SCOPES:
            block = payload.get(match, {}).get(scope, {})
            for metric in METRICS:
                mean, std = _to_mean_std(block.get(metric))
                flat[f"{match}_{scope}_{metric}_mean"] = mean
                flat[f"{match}_{scope}_{metric}_std"] = std
    return flat


def _combine_entries(entries: list[dict[str, Any]]) -> dict[str, float | None]:
    """Aggregate already-aggregated entries by averaging means across entries."""
    out: dict[str, float | None] = {}
    keys = [
        f"{match}_{scope}_{metric}_mean"
        for match in MATCH_TYPES
        for scope in SCOPES
        for metric in METRICS
    ]

    for key in keys:
        values = [e["metrics"].get(key) for e in entries]
        values = [v for v in values if isinstance(v, (int, float))]
        if values:
            out[key] = float(stats.mean(values))
            out[key.replace("_mean", "_std")] = (
                float(stats.stdev(values)) if len(values) > 1 else 0.0
            )
        else:
            out[key] = None
            out[key.replace("_mean", "_std")] = None

    return out


def _model_from_result_path(input_root: Path, result_path: Path) -> str:
    rel = result_path.relative_to(input_root)
    return rel.parts[0] if rel.parts else result_path.parent.name


def aggregate(input_root: Path, result_filename: str) -> list[dict[str, Any]]:
    result_paths = sorted(input_root.rglob(result_filename))
    if not result_paths:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}

    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = _model_from_result_path(input_root, path)
        n_folds = payload.get("_metadata", {}).get("n_folds")
        n_folds = int(n_folds) if isinstance(n_folds, int) else None

        grouped.setdefault(model, []).append(
            {
                "path": path,
                "n_folds": n_folds,
                "metrics": _extract_flat_metrics(payload),
            }
        )

    rows: list[dict[str, Any]] = []
    for model, entries in sorted(grouped.items()):
        # Prefer a directly aggregated file with the highest n_folds (>1).
        direct_candidates = [
            e for e in entries if isinstance(e["n_folds"], int) and e["n_folds"] > 1
        ]

        if direct_candidates:
            best = sorted(
                direct_candidates,
                key=lambda e: (e["n_folds"], -len(str(e["path"]))),
                reverse=True,
            )[0]
            metrics = best["metrics"]
            source = str(best["path"])
            n_folds = best["n_folds"]
            aggregation_mode = "direct_aggregated_file"
        else:
            metrics = _combine_entries(entries)
            source = f"{len(entries)} files"
            n_folds = len(entries)
            aggregation_mode = "recomputed_across_files"

        row: dict[str, Any] = {
            "model": model,
            "n_folds": n_folds,
            "aggregation_mode": aggregation_mode,
            "source": source,
        }
        row.update(metrics)
        rows.append(row)

    return rows


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    if not rows:
        return

    metric_cols = [
        f"{match}_{scope}_{metric}_{suffix}"
        for match in MATCH_TYPES
        for scope in SCOPES
        for metric in METRICS
        for suffix in ("mean", "std")
    ]
    fieldnames = ["model", "n_folds", "aggregation_mode", "source", *metric_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(x: Any) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return "-"


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No result files found.")
        return

    print("\nModel summary (strict/relaxed micro F1):")
    print("-" * 90)
    print(f"{'Model':30} {'n_folds':>7} {'strict μF1':>15} {'relaxed μF1':>15}")
    print("-" * 90)
    for r in rows:
        s_mean = _fmt(r.get("strict_micro_F1_mean"))
        s_std = _fmt(r.get("strict_micro_F1_std"))
        r_mean = _fmt(r.get("relaxed_micro_F1_mean"))
        r_std = _fmt(r.get("relaxed_micro_F1_std"))
        print(
            f"{r['model'][:30]:30} {str(r['n_folds']):>7} "
            f"{s_mean} ± {s_std:>7} {r_mean} ± {r_std:>7}"
        )
    print("-" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate sigmoid sensitivity fold results into a model-level CSV."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root folder containing per-model sigmoid sensitivity outputs.",
    )
    parser.add_argument(
        "--result-filename",
        default="aggregated_fold_on_fold_results.json",
        help="Result filename to search recursively for.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Default: <input-root>/model_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    output_csv = args.output_csv or (input_root / "model_summary.csv")

    rows = aggregate(input_root=input_root, result_filename=args.result_filename)
    if not rows:
        print(f"No '{args.result_filename}' files found under: {input_root}")
        return

    write_csv(rows, output_csv)
    print_summary(rows)
    print(f"\nSaved CSV: {output_csv}")


if __name__ == "__main__":
    main()
