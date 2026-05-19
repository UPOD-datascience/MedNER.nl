import argparse
import csv
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Constants
START_SPAN_TAG = "start_span"
END_SPAN_TAG = "end_span"
ENTITY_NAME_TAG = "text"
LABEL_TAG = "label"
FILE_NAME = "filename"


def parse_tsv_file(datapath: str, entities_to_evaluate: list) -> pd.DataFrame:
    """
    Parse a TSV file into a DataFrame and perform basic formatting and deduplication.

    Parameters:
    -----------
    datapath : str
        Path to the TSV file.
    entities_to_evaluate: list
        List of entities to evaluate. If none, take all entities

    Returns:
    --------
    pd.DataFrame
        Formatted and deduplicated DataFrame.
    """
    try:
        # Load the TSV file
        df = pd.read_csv(
            datapath,
            sep="\t",
            header=0,
            quoting=csv.QUOTE_NONE,
            keep_default_na=False,
            dtype=str,
        )

        if entities_to_evaluate:
            df = df.loc[df[LABEL_TAG].isin(entities_to_evaluate), :].copy()

        # Format DataFrame
        df["offset"] = (
            df[START_SPAN_TAG].astype(str) + " " + df[END_SPAN_TAG].astype(str)
        )
        df = df[~df[LABEL_TAG].isna() & (df[LABEL_TAG].str.strip() != "")]
        df[LABEL_TAG] = df[LABEL_TAG].str.upper()

        # Check for duplicated entries
        if df.duplicated(subset=[FILE_NAME, LABEL_TAG, "offset"]).any():
            df = df.drop_duplicates(subset=[FILE_NAME, LABEL_TAG, "offset"]).copy()
            logger.warning("Duplicated entries found and removed.")

        return df

    except Exception as e:
        logger.error(f"Error parsing TSV file: {e}")
        raise


def parse_json_file(
    datapath: str, entities_to_evaluate: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Parse a JSON file into a DataFrame and perform basic formatting and deduplication.

    Expected JSON format:
    [
        {
            "entity_group": "MEDICATION",
            "word": "omeprazol",
            "start": 1462,
            "end": 1471
        },
        ...
    ]

    Parameters:
    -----------
    datapath : str
        Path to the JSON file.
    entities_to_evaluate : Optional[List[str]]
        List of entities to evaluate. If None, take all entities.

    Returns:
    --------
    pd.DataFrame
        Formatted and deduplicated DataFrame with columns matching TSV format:
        [filename, start_span, end_span, text, label, offset]
    """
    try:
        # Load the JSON file
        with open(datapath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of entity objects")

        # Convert to DataFrame
        df = pd.DataFrame(data)

        if df.empty:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=[
                    FILE_NAME,
                    START_SPAN_TAG,
                    END_SPAN_TAG,
                    ENTITY_NAME_TAG,
                    LABEL_TAG,
                    "offset",
                ]
            )

        # Rename columns to match TSV format
        column_mapping = {
            "entity_group": LABEL_TAG,
            "word": ENTITY_NAME_TAG,
            "start": START_SPAN_TAG,
            "end": END_SPAN_TAG,
        }
        df = df.rename(columns=column_mapping)

        # Add filename column (use common placeholder since JSON format doesn't include filename)
        df[FILE_NAME] = "document"

        # Filter by entities if specified
        if entities_to_evaluate:
            entities_upper = [e.upper() for e in entities_to_evaluate]
            df = df.loc[df[LABEL_TAG].str.upper().isin(entities_upper), :].copy()

        # Format DataFrame
        df[START_SPAN_TAG] = df[START_SPAN_TAG].astype(str)
        df[END_SPAN_TAG] = df[END_SPAN_TAG].astype(str)
        df["offset"] = df[START_SPAN_TAG] + " " + df[END_SPAN_TAG]
        df = df[~df[LABEL_TAG].isna() & (df[LABEL_TAG].str.strip() != "")]
        df[LABEL_TAG] = df[LABEL_TAG].str.upper()

        # Check for duplicated entries
        if df.duplicated(subset=[FILE_NAME, LABEL_TAG, "offset"]).any():
            df = df.drop_duplicates(subset=[FILE_NAME, LABEL_TAG, "offset"]).copy()
            logger.warning("Duplicated entries found and removed.")

        return df

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise


def calculate_metrics_strict(
    gs: pd.DataFrame, pred: pd.DataFrame
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    """
    Calculates strict matching metrics (exact span and label match) including precision, recall, and F1-score
    for each label, along with micro and macro-averaged scores.

    This function assumes the input dataframes contain an `offset` field, a `label` field, and a `filename` column.
    Strict evaluation considers a prediction correct only if the filename, offset, and label all match.

    Args:
        gs (pd.DataFrame): Ground truth mentions with columns ['filename', 'offset', 'label'].
        pred (pd.DataFrame): Predicted mentions with the same required columns.

    Returns:
        Tuple containing:
            - Dict[str, Dict[str, float]]: Per-label metrics with precision, recall, and F1-score.
            - Dict[str, float]: Micro-averaged precision, recall, and F1-score.
            - Dict[str, float]: Macro-averaged precision, recall, and F1-score.
    """

    # Keep duplicates for accounting, but match mentions in a 1-to-1 way per
    # (filename, start_span, end_span, label) key to avoid TP overcounting from
    # many-to-many dataframe merges.

    labels = sorted(set(gs[LABEL_TAG].unique()) | set(pred[LABEL_TAG].unique()))
    result_by_cat = {}

    total_tp = total_fp = total_fn = 0
    precision_list = []
    recall_list = []
    f1_list = []

    for label in labels:
        gs_label = gs[gs[LABEL_TAG] == label]
        pred_label = pred[pred[LABEL_TAG] == label]

        key_cols = [FILE_NAME, START_SPAN_TAG, END_SPAN_TAG, LABEL_TAG]

        # Count mentions per exact key in prediction and gold.
        pred_counts = pred_label.groupby(key_cols, dropna=False).size()
        gs_counts = gs_label.groupby(key_cols, dropna=False).size()

        Pred_Pos = int(pred_counts.sum())
        GS_Pos = int(gs_counts.sum())

        # Exact-match TP with 1-to-1 matching: for each key, matches are limited
        # by the smaller multiplicity between pred and gold.
        counts = pd.concat([pred_counts, gs_counts], axis=1).fillna(0)
        counts.columns = ["pred_n", "gs_n"]
        TP = int(counts[["pred_n", "gs_n"]].min(axis=1).sum())

        FP = max(Pred_Pos - TP, 0)
        FN = max(GS_Pos - TP, 0)

        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        result_by_cat[label] = {
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1": round(f1, 2),
        }

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        total_tp += TP
        total_fp += FP
        total_fn += FN

    # Micro-averaged scores
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    # Macro-averaged scores
    macro_precision = (
        sum(precision_list) / len(precision_list) if precision_list else 0.0
    )
    macro_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    summary_micro = {
        "Precision": round(micro_precision, 2),
        "Recall": round(micro_recall, 2),
        "F1": round(micro_f1, 2),
    }

    summary_macro = {
        "Precision": round(macro_precision, 2),
        "Recall": round(macro_recall, 2),
        "F1": round(macro_f1, 2),
    }

    return result_by_cat, summary_micro, summary_macro


def calculate_metrics_relaxed(
    gs: pd.DataFrame, pred: pd.DataFrame
) -> (Tuple)[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    """
    Compute relaxed precision, recall, and F1-score for entity recognition.

    This function compares predicted and ground truth entity spans using a relaxed (interval overlap) strategy,
    where a match is valid if the spans overlap and the labels match. Each prediction can match at most one gold entity.

    Args:
        gs (pd.DataFrame): Ground truth mentions. Must contain columns ['filename', 'start_span', 'end_span', 'label'].
        pred (pd.DataFrame): Predicted mentions with the same column structure.

    Returns:
        Tuple containing:
            - result_by_cat (Dict[str, Dict[str, float]]): Per-label scores with precision, recall, and F1.
            - summary_micro (Dict[str, float]): Micro-averaged precision, recall, and F1.
            - summary_macro (Dict[str, float]): Macro-averaged precision, recall, and F1.

    Notes:
        - Intervals with missing 'start_span' or 'end_span' are ignored.
        - Micro scores are calculated by summing true positives, false positives, and false negatives.
        - Macro scores are calculated by averaging the per-label scores.
        - All scores are rounded to 4 decimal places.
    """

    # Clean and prepare intervals
    for df in [gs, pred]:
        df["start_span"] = pd.to_numeric(df["start_span"], errors="coerce")
        df["end_span"] = pd.to_numeric(df["end_span"], errors="coerce")

    gs_mentions = gs.dropna(subset=["start_span", "end_span"]).copy()
    preds_mentions = pred.dropna(subset=["start_span", "end_span"]).copy()

    gs_mentions["interval"] = pd.arrays.IntervalArray.from_arrays(
        gs_mentions["start_span"], gs_mentions["end_span"], closed="both"
    )
    preds_mentions["interval"] = pd.arrays.IntervalArray.from_arrays(
        preds_mentions["start_span"], preds_mentions["end_span"], closed="both"
    )

    labels = sorted(gs_mentions["label"].unique())
    result_by_cat = {}

    relaxed_TP_total = relaxed_FP_total = relaxed_FN_total = 0
    precision_list = []
    recall_list = []
    f1_list = []

    gs_grouped = gs_mentions.groupby(["label", "filename"])
    preds_grouped = preds_mentions.groupby(["label", "filename"])

    for label in labels:
        tp = fp = fn = 0
        filenames = set(gs_mentions[gs_mentions["label"] == label]["filename"]) | set(
            preds_mentions[preds_mentions["label"] == label]["filename"]
        )

        for filename in filenames:
            gs_filtered = (
                gs_grouped.get_group((label, filename))
                if (label, filename) in gs_grouped.groups
                else pd.DataFrame()
            )
            preds_filtered = (
                preds_grouped.get_group((label, filename))
                if (label, filename) in preds_grouped.groups
                else pd.DataFrame()
            )

            gs_rows = list(gs_filtered.itertuples())
            pred_rows = list(preds_filtered.itertuples())

            gs_used = set()
            preds_used = set()

            for gs_idx, gs_row in enumerate(gs_rows):
                for pred_idx, pred_row in enumerate(pred_rows):
                    if pred_idx in preds_used:
                        continue
                    if gs_row.interval.overlaps(pred_row.interval):
                        tp += 1
                        gs_used.add(gs_idx)
                        preds_used.add(pred_idx)
                        break

            fp += len(pred_rows) - len(preds_used)
            fn += len(gs_rows) - len(gs_used)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        result_by_cat[label] = {
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1": round(f1, 2),
        }

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        relaxed_TP_total += tp
        relaxed_FP_total += fp
        relaxed_FN_total += fn

    # Micro scores
    micro_precision = (
        relaxed_TP_total / (relaxed_TP_total + relaxed_FP_total)
        if (relaxed_TP_total + relaxed_FP_total)
        else 0.0
    )
    micro_recall = (
        relaxed_TP_total / (relaxed_TP_total + relaxed_FN_total)
        if (relaxed_TP_total + relaxed_FN_total)
        else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    # Macro scores
    macro_precision = (
        sum(precision_list) / len(precision_list) if precision_list else 0.0
    )
    macro_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    summary_micro = {
        "Precision": round(micro_precision, 2),
        "Recall": round(micro_recall, 2),
        "F1": round(micro_f1, 2),
    }

    summary_macro = {
        "Precision": round(macro_precision, 2),
        "Recall": round(macro_recall, 2),
        "F1": round(macro_f1, 2),
    }

    return result_by_cat, summary_micro, summary_macro


def print_scores(
    scores: Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]],
) -> None:
    """
    Pretty print evaluation scores.

    Args:
        scores: Tuple containing:
            - result_by_cat: Per-label metrics with precision, recall, and F1-score.
            - summary_micro: Micro-averaged precision, recall, and F1-score.
            - summary_macro: Macro-averaged precision, recall, and F1-score.
    """
    result_by_cat, summary_micro, summary_macro = scores

    print("\n{:<25} {:>12} {:>12} {:>12}".format("Label", "Precision", "Recall", "F1"))
    print("-" * 65)

    for label, metrics in sorted(result_by_cat.items()):
        print(
            "{:<25} {:>12.2f} {:>12.2f} {:>12.2f}".format(
                label, metrics["Precision"], metrics["Recall"], metrics["F1"]
            )
        )

    print("-" * 65)
    print(
        "{:<25} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            "MICRO-AVG",
            summary_micro["Precision"],
            summary_micro["Recall"],
            summary_micro["F1"],
        )
    )
    print(
        "{:<25} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            "MACRO-AVG",
            summary_macro["Precision"],
            summary_macro["Recall"],
            summary_macro["F1"],
        )
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ref_tsv", type=str, help="Reference file path TSV")
    argparser.add_argument("--pred_tsv", type=str, help="Prediction file path TSV")
    argparser.add_argument("--ref_json", type=str, help="Reference file path JSON")
    argparser.add_argument("--pred_json", type=str, help="Prediction file path JSON")
    argparser.add_argument(
        "--entities", type=str, nargs="+", help="Entities to evaluate"
    )
    args = argparser.parse_args()

    # Validate that either TSV files or JSON files are provided, but not both
    if (args.ref_tsv and args.pred_tsv) and (args.ref_json and args.pred_json):
        raise ValueError(
            "Please provide either TSV files (--ref_tsv and --pred_tsv) OR JSON files (--ref_json and --pred_json), not both."
        )

    if not ((args.ref_tsv and args.pred_tsv) or (args.ref_json and args.pred_json)):
        raise ValueError(
            "Please provide either both TSV files (--ref_tsv and --pred_tsv) OR both JSON files (--ref_json and --pred_json)."
        )

    if args.entities and not all(isinstance(entity, str) for entity in args.entities):
        raise ValueError("Entities must be a list of strings")

    # Check if files exist and are valid\    import os
    if args.ref_tsv and args.pred_tsv:
        if not os.path.exists(args.ref_tsv):
            raise FileNotFoundError(f"Reference TSV file not found: {args.ref_tsv}")
        if not os.path.exists(args.pred_tsv):
            raise FileNotFoundError(f"Prediction TSV file not found: {args.pred_tsv}")
        # check if args.entities is a list of strings

        df_ref = parse_tsv_file(args.ref_tsv, args.entities)
        df_pred = parse_tsv_file(args.pred_tsv, args.entities)

        # check if df_ref and df_pred have the same columns
        if not set(df_ref.columns) == set(df_pred.columns):
            raise ValueError(
                "Reference and prediction TSV files must have the same columns"
            )

        # get scores strict and relaxeds
        scores_strict = calculate_metrics_strict(df_ref, df_pred)
        scores_relaxed = calculate_metrics_relaxed(df_ref, df_pred)

        # pretty print scores
        print("Strict Scores:")
        print("=" * 50)
        print_scores(scores_strict)
        print("\nRelaxed Scores:")
        print("=" * 50)
        print_scores(scores_relaxed)

    if args.ref_json and args.pred_json:
        if not os.path.exists(args.ref_json):
            raise FileNotFoundError(f"Reference JSON file not found: {args.ref_json}")
        if not os.path.exists(args.pred_json):
            raise FileNotFoundError(f"Prediction JSON file not found: {args.pred_json}")

        # parse json files
        df_ref = parse_json_file(args.ref_json, args.entities)
        df_pred = parse_json_file(args.pred_json, args.entities)

        # check if df_ref and df_pred have the same columns
        if not set(df_ref.columns) == set(df_pred.columns):
            raise ValueError(
                "Reference and prediction JSON files must have the same columns"
            )

        # get scores strict and relaxed
        scores_strict = calculate_metrics_strict(df_ref, df_pred)
        scores_relaxed = calculate_metrics_relaxed(df_ref, df_pred)

        # pretty print scores
        print("Strict Scores:")
        print("=" * 50)
        print_scores(scores_strict)
        print("\nRelaxed Scores:")
        print("=" * 50)
        print_scores(scores_relaxed)
