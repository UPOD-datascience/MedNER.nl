# process model folds with inference
import argparse
import json
import os
from typing import List, Literal, Tuple

import numpy as np

from cardioner import main

"""
 splits_file:
     {
     "folds": [
        {
            "train_files": [],
            "val_files": [],
        },
        {
            "train_files": [],
            "val_files": [],
        },
        ...
     ],
     "test_files": []
     }
"""


def make_corpora(bulk_file, splits_file):
    print(f"Using the relative split file {splits_file}.")
    print(50 * "=")

    corpus_list = []
    with open(bulk_file, "r", encoding="utf-8") as fr:
        for line in fr:
            corpus_list.append(json.loads(line))

    with open(splits_file, "r", encoding="utf-8") as fr:
        split_data = json.load(fr)
        corpus_folds = split_data["folds"]
        corpus_validation_ids = [
            entry.strip(".txt") for entry in split_data["test_files"]
        ]

    corpus_train_id_lists = [
        [entry.strip(".txt") for entry in fold["train_files"]] for fold in corpus_folds
    ]
    corpus_test_id_lists = [
        [entry.strip(".txt") for entry in fold["val_files"]] for fold in corpus_folds
    ]

    # splits: [([train ids],[test ids]), ([train ids],[test ids])...]
    #
    splits = list(zip(corpus_train_id_lists, corpus_test_id_lists))

    corpus_validation_list = [
        entry for entry in corpus_list if entry["id"] in corpus_validation_ids
    ]

    # print overview of counts per fold
    print(100 * "=")
    print("Overview of counts per fold:")
    print(100 * "=")
    for k, fold in enumerate(corpus_folds):
        print(f"Fold {k}:")
        print(f"  Train: {len(fold['train_files'])}")
        print(f"  Test: {len(fold['val_files'])}")
    print(f"  Validation: {len(corpus_validation_list)}")
    print(100 * "=")
    print(100 * "=")

    return corpus_list, corpus_validation_list, splits


def make_corpora_from_rel_split(bulk_file, rel_split_file, model_list):
    print(f"Using the relative split file {rel_split_file}.")
    print(50 * "=")

    corpus_list = []
    with open(bulk_file, "r", encoding="utf-8") as fr:
        for line in fr:
            corpus_list.append(json.loads(line))

    splits = []
    for k, model_path in enumerate(model_list):
        split_path = os.path.join(model_path, rel_split_file)
        with open(split_path, "r", encoding="utf-8") as fr:
            split_data = json.load(fr)
            train_ids = list(set(split_data["train_gids"]))
            test_ids = list(set(split_data["test_gids"]))
            val_ids = split_data.get("val_gids", [])

            splits.append((train_ids, test_ids))

            if k == 0:
                _val_prev = val_ids
            elif _val_prev != val_ids:
                raise ValueError(
                    f"Inconsistent val_gids across folds at fold {k}: expected same list as fold 0."
                )

    # print overview of counts per fold
    print(100 * "=")
    print("Overview of counts per fold:")
    print(100 * "=")
    for k, fold in enumerate(splits):
        print(f"Fold {k}:")
        print(f"  Train: {len(fold[0])}")
        print(f"  Test: {len(fold[1])}")
    print(f"  Validation: {len(val_ids)}")
    print(100 * "=")
    print(100 * "=")
    pass

    return corpus_list, val_ids, splits


def get_model_folders(model_folder: str, folder_prefix="fold_"):
    return [f for f in os.listdir(model_folder) if f.startswith(folder_prefix)]


def aggregate_results(
    resultfile_list: List[str], output_file: str | None = None
) -> dict:
    """
    Aggregate results from multiple fold result files.

    Computes mean and standard deviation for all metrics across folds:
    - strict/relaxed
    - per_category (DISEASE, MEDICATION, PROCEDURE, SYMPTOM)
    - micro/macro
    - Precision, Recall, F1

    Args:
        resultfile_list: List of paths to result JSON files
        output_file: Optional path to save aggregated results

    Returns:
        Dictionary with aggregated results (mean and std for each metric)
    """
    # Load all results
    results = []
    for f in resultfile_list:
        with open(f, "r", encoding="utf-8") as fr:
            results.append(json.load(fr))

    if not results:
        return {}

    # Initialize structure to collect values
    metrics = ["Precision", "Recall", "F1"]
    match_types = ["strict", "relaxed"]
    agg_types = ["micro", "macro"]

    # Get categories from first result file
    categories = list(results[0]["strict"]["per_category"].keys())

    # Collect all values
    collected = {
        match_type: {
            "per_category": {cat: {m: [] for m in metrics} for cat in categories},
            "micro": {m: [] for m in metrics},
            "macro": {m: [] for m in metrics},
        }
        for match_type in match_types
    }

    for result in results:
        for match_type in match_types:
            # Per category metrics
            for cat in categories:
                for metric in metrics:
                    value = result[match_type]["per_category"].get(cat, {}).get(metric)
                    if value is not None:
                        collected[match_type]["per_category"][cat][metric].append(value)

            # Micro and macro metrics
            for agg_type in agg_types:
                for metric in metrics:
                    value = result[match_type].get(agg_type, {}).get(metric)
                    if value is not None:
                        collected[match_type][agg_type][metric].append(value)

    # Compute mean and std
    aggregated = {
        match_type: {
            "per_category": {},
            "micro": {},
            "macro": {},
        }
        for match_type in match_types
    }

    for match_type in match_types:
        # Per category
        for cat in categories:
            aggregated[match_type]["per_category"][cat] = {}
            for metric in metrics:
                values = collected[match_type]["per_category"][cat][metric]
                if values:
                    aggregated[match_type]["per_category"][cat][metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        # "values": values,
                    }

        # Micro and macro
        for agg_type in agg_types:
            aggregated[match_type][agg_type] = {}
            for metric in metrics:
                values = collected[match_type][agg_type][metric]
                if values:
                    aggregated[match_type][agg_type][metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        # "values": values,
                    }

    # Add metadata
    aggregated["_metadata"] = {
        "n_folds": len(results),
        "source_files": resultfile_list,
    }

    # Save if output file specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as fw:
            json.dump(aggregated, fw, indent=2)
        print(f"Aggregated results saved to: {output_file}")

    # Print summary
    print_aggregated_summary(aggregated)

    return aggregated


def print_aggregated_summary(aggregated: dict):
    """Print a formatted summary of aggregated results."""
    n_folds = aggregated.get("_metadata", {}).get("n_folds", "?")
    print("\n" + "=" * 80)
    print(f"AGGREGATED RESULTS ACROSS {n_folds} FOLDS")
    print("=" * 80)

    for match_type in ["strict", "relaxed"]:
        print(f"\n{match_type.upper()} MATCHING:")
        print("-" * 60)

        # Micro/Macro
        for agg_type in ["micro", "macro"]:
            print(f"\n  {agg_type.capitalize()}:")
            for metric in ["Precision", "Recall", "F1"]:
                data = aggregated[match_type][agg_type].get(metric, {})
                mean = data.get("mean", 0)
                std = data.get("std", 0)
                print(f"    {metric}: {mean:.3f} ± {std:.3f}")

        # Per category
        print(f"\n  Per Category:")
        categories = list(aggregated[match_type]["per_category"].keys())
        for cat in categories:
            print(f"    {cat}:")
            for metric in ["Precision", "Recall", "F1"]:
                data = aggregated[match_type]["per_category"][cat].get(metric, {})
                mean = data.get("mean", 0)
                std = data.get("std", 0)
                print(f"      {metric}: {mean:.3f} ± {std:.3f}")

    print("\n" + "=" * 80)


def get_type(model_path):
    config_path = os.path.join(model_path, "config.json")
    is_multihead_crf = False
    is_multihead = False
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            # Check for multihead CRF indicators in config
            is_multihead_crf = "TokenClassificationModelMultiHeadCRF" in config.get(
                "architectures", []
            )
            # Check for multihead (no CRF) indicators in config
            is_multihead = "TokenClassificationModelMultiHead" in config.get(
                "architectures", []
            )

    if (is_multihead_crf == False) and (is_multihead == False):
        return "standard"
    elif is_multihead_crf:
        return "multihead_crf"
    else:
        return "multihead"


def process_splits(
    corpus: List[dict],
    corpus_validation: List[dict] | None,
    model_list: List[str],
    splits: List[Tuple],
    output_dir: str,
    lang: str,
    max_word_per_chunk: int = 256,
    trust_remote_code: bool = True,
    strategy: Literal["average", "min", "first", "simple"] = "simple",
    pipe: Literal["dt4h", "hf"] = "hf",
):
    # This runs and scores inference for each folds test samples
    resultfile_list = []
    for k, (_, test_indcs) in enumerate(splits):
        print(f"Running inference for fold {k}")

        model_type = get_type(model_list[k])
        _corpus = [d for d in corpus if d["id"] in test_indcs]
        output_file_prefix = f"fold_on_fold{k}_"

        if model_type == "standard":
            main.inference(
                _corpus,
                model_list[k],
                output_dir=output_dir,
                output_file_prefix=output_file_prefix,
                lang=lang,
                max_word_per_chunk=max_word_per_chunk,
                trust_remote_code=trust_remote_code,
                strategy=strategy,
                pipe=pipe,
            )
        elif model_type == "multihead_crf":
            main.inference_multihead_crf(
                corpus_data=_corpus,
                model_path=model_list[k],
                output_dir=output_dir,
                output_file_prefix=output_file_prefix,
                lang=lang,
                max_word_per_chunk=max_word_per_chunk,  # Auto-detect from tokenizer
                trust_remote_code=True,  # Always true for multihead CRF
            )
        else:
            main.inference_multihead(
                corpus_data=_corpus,
                model_path=model_list[k],
                output_dir=output_dir,
                output_file_prefix=output_file_prefix,
                lang=lang,
                max_word_per_chunk=max_word_per_chunk,  # Auto-detect from tokenizer
                trust_remote_code=True,  # Always true for multihead
            )

        resultfile_list.append(f"{output_dir}/{output_file_prefix}sequence_result.json")

    resultfile_val_list = []
    if corpus_validation is not None:
        # This runs and scores inference for each fold model on the same validation set
        for k, (_, test_indcs) in enumerate(splits):
            print(f"Running inference for fold {k} on validation")
            output_file_prefix = f"fold{k}_on_val_"

            model_type = get_type(model_list[k])

            if model_type == "standard":
                main.inference(
                    corpus_validation,
                    model_list[k],
                    output_dir=output_dir,
                    output_file_prefix=output_file_prefix,
                    lang=lang,
                    max_word_per_chunk=max_word_per_chunk,
                    trust_remote_code=trust_remote_code,
                    strategy=strategy,
                    pipe=pipe,
                )
            elif model_type == "multihead_crf":
                main.inference_multihead_crf(
                    corpus_data=corpus_validation,
                    model_path=model_list[k],
                    output_dir=output_dir,
                    output_file_prefix=output_file_prefix,
                    lang=lang,
                    max_word_per_chunk=max_word_per_chunk,  # Auto-detect from tokenizer
                    trust_remote_code=True,  # Always true for multihead CRF
                )
            else:
                main.inference_multihead(
                    corpus_data=corpus_validation,
                    model_path=model_list[k],
                    output_dir=output_dir,
                    output_file_prefix=output_file_prefix,
                    lang=lang,
                    max_word_per_chunk=max_word_per_chunk,  # Auto-detect from tokenizer
                    trust_remote_code=True,  # Always true for multihead
                )

            resultfile_val_list.append(
                f"{output_dir}/{output_file_prefix}sequence_result.json"
            )

    # Aggregate fold-on-fold results
    print("\n" + "=" * 80)
    print("FOLD-ON-FOLD RESULTS (each model tested on its own fold's test set)")
    print("=" * 80)
    fold_on_fold_aggregated = aggregate_results(
        resultfile_list,
        output_file=f"{output_dir}/aggregated_fold_on_fold_results.json",
    )

    # Aggregate validation results if available
    val_aggregated = None
    if corpus_validation is not None and resultfile_val_list:
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS (all models tested on held-out validation set)")
        print("=" * 80)
        val_aggregated = aggregate_results(
            resultfile_val_list,
            output_file=f"{output_dir}/aggregated_validation_results.json",
        )

    return fold_on_fold_aggregated, val_aggregated


def run_main():
    parser = argparse.ArgumentParser(
        description="Run inference on k-fold cross-validation splits and aggregate results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--bulk_file",
        type=str,
        required=True,
        help="Path to the bulk JSONL file containing all corpus entries.",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        required=False,
        help="Path to the JSON file containing fold splits and test files.",
    )
    parser.add_argument(
        "--rel_split_file",
        type=str,
        required=False,
        help="Path to the JSON file containing the train/test split, relative to the fold-folder: e.g. /fold_0/split.json = --rel_split_file=split.json.",
    )

    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Path to the folder containing trained fold models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save inference results.",
    )

    # Optional arguments
    parser.add_argument(
        "--lang",
        type=str,
        default="nl",
        help="Language code for the corpus (e.g., 'nl', 'en', or 'multi' for multilingual).",
    )
    parser.add_argument(
        "--max_word_per_chunk",
        type=int,
        default=None,
        help="Maximum number of words per chunk for inference.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Whether to trust remote code when loading models.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="simple",
        choices=["average", "min", "first", "simple"],
        help="Strategy for combining predictions.",
    )
    parser.add_argument(
        "--pipe",
        type=str,
        default="hf",
        choices=["dt4h", "hf"],
        help="Pipeline type to use for inference.",
    )
    parser.add_argument(
        "--folder_prefix",
        type=str,
        default="fold_",
        help="Prefix for fold model folders.",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        default=False,
        help="Skip inference on the held-out validation/test set.",
    )
    args = parser.parse_args()

    if (args.split_file is None) == (args.rel_split_file is None):
        raise ValueError("Provide exactly one of --split_file or --rel_split_file.")

    if not os.path.isdir(args.model_folder):
        raise FileNotFoundError(f"Model folder not found: {args.model_folder}")

    if not os.path.isfile(args.bulk_file):
        raise FileNotFoundError(f"Bulk file not found: {args.bulk_file}")

    # Get model folders and sort them
    model_folders = get_model_folders(args.model_folder, args.folder_prefix)
    model_folders = sorted(model_folders)  # Ensure consistent ordering
    model_list = [os.path.join(args.model_folder, f) for f in model_folders]

    # Prepare corpora and splits
    if args.rel_split_file is None:
        corpus, corpus_validation, splits = make_corpora(
            args.bulk_file, args.split_file
        )
    else:
        corpus, corpus_validation, splits = make_corpora_from_rel_split(
            args.bulk_file, args.rel_split_file, model_list
        )

    if args.skip_validation:
        corpus_validation = None

    if len(model_list) != len(splits):
        raise ValueError(
            f"Number of model folders ({len(model_list)}) does not match "
            f"number of splits ({len(splits)}). "
            f"Found models: {model_folders}"
        )

    print(f"Found {len(model_list)} fold models:")
    for i, model_path in enumerate(model_list):
        print(f"  Fold {i}: {model_path}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference on all splits
    fold_on_fold_aggregated, val_aggregated = process_splits(
        corpus=corpus,
        corpus_validation=corpus_validation,
        model_list=model_list,
        splits=splits,
        output_dir=args.output_dir,
        lang=args.lang,
        max_word_per_chunk=args.max_word_per_chunk,
        trust_remote_code=args.trust_remote_code,
        strategy=args.strategy,
        pipe=args.pipe,
    )

    return fold_on_fold_aggregated, val_aggregated


if __name__ == "__main__":
    run_main()
