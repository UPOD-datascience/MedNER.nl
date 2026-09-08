#!/usr/bin/env python
"""
Token-level evaluation of MULTILABEL fold models on the held-out test set,
excluding the O class.

Why this exists
---------------
The one problem is that their micro/macro aggregates INCLUDE the O class, which
inflates them relative to standard NER practice -- micro over tokens including O
is essentially token accuracy. 

Note that macro-without-O can simply be averaged from the per-class values
already in `fold_N_results.json` -- no rerun needed. Only micro requires this
script, because it must be recomputed from the underlying token matrix.


What is reused
--------------
Everything that touches data or models comes from the original package:

  * `cardioner.main.prepare`                    -- tokenisation / IOB labelling
  * `MultiLabelTokenClassificationModelHF`      -- the model class
  * `MultiLabelDataCollatorForTokenClassification`
  * `transformers.Trainer.predict`              -- the forward pass

Decoding matches training exactly: `sigmoid(logits) > 0.5`, independently per
label, so a token may carry several labels or none.

The only thing this script adds is the O-exclusion and the cross-fold
aggregation.

Two details that matter for correctness:

  * `prepare()` derives its label order from a set, and it does NOT match the
    order stored in the trained model's config (prepare puts DISEASE first, the
    configs put MEDICATION first). The multi-hot columns are realigned before
    scoring; without that every metric would be silently wrong. The script
    prints which branch it took.
  * Defaults match run_main.sh (chunk_size=256, chunk_type=paragraph,
    max_length=256). If you trained with different values, pass them explicitly
    -- mismatched chunking silently changes the token population being scored.

Usage
-----
    python run_token_level_holdout.py \
        --bulk_file    .../cz/annotations_all_entities.jsonl \
        --split_file   .../dataset_splits/splits_cv10_holdout_test.json \
        --model_folder .../multilabel_cardioberta/cz_all_entities_lr_7e-5 \
        --output_dir   .../cz_all_entities_lr_7e-5/token_eval_holdout
"""

import argparse
import copy
import json
import os
import statistics as stat
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments

KEEP_KEYS = {"input_ids", "attention_mask", "labels"}


def load_holdout_chunks(
    bulk_file: str,
    split_file: str,
    model_path: str,
    chunk_size: int,
    chunk_type: str,
    max_length: int,
    entity_types: List[str] | None,
):
    """
    Tokenise the corpus with the original `prepare()` and keep only the chunks
    belonging to held-out test documents.

    The full corpus is passed to `prepare()` (not just the holdout) so that the
    label inventory is discovered from all documents; per-language holdout sets
    can otherwise be missing an entity type entirely. Filtering happens after.
    """
    from cardioner import main as cardioner_main

    with open(split_file, "r", encoding="utf-8") as f:
        holdout_ids = {
            entry[:-4] if entry.endswith(".txt") else entry
            for entry in json.load(f)["test_files"]
        }

    with open(bulk_file, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    id2label = {int(k): v for k, v in config.id2label.items()}
    model_label2id = {v: int(k) for k, v in id2label.items()}

    if entity_types is None:
        entity_types = sorted(
            {
                label.replace("B-", "").replace("I-", "")
                for label in id2label.values()
                if label != "O"
            }
        )

    tokenized, _tags = cardioner_main.prepare(
        Model=model_path,
        corpus_train=corpus,
        corpus_validation=None,  # None (not []) so prepare() drops the batch entirely
        corpus_test=None,
        chunk_size=chunk_size,
        chunk_type=chunk_type,
        max_length=max_length,
        multi_class=False,  # multilabel
        use_iob=True,
        entity_types=entity_types,
    )

    chunks = [e for e in tokenized if e.get("gid") in holdout_ids]
    if not chunks:
        raise SystemExit(
            "No chunks matched the held-out test documents. Check that "
            "--bulk_file and --split_file refer to the same corpus."
        )

    # `prepare()` derives label2id from a set, so its column order is not
    # guaranteed to match the order stored in the trained model's config.
    # Realign the multi-hot columns before scoring, or every metric is wrong.
    prepare_label2id = chunks[0]["label2id"]
    if prepare_label2id != model_label2id:
        print("Label order differs from model config -- remapping columns.")
        perm = [prepare_label2id[id2label[i]] for i in range(len(id2label))]
        for chunk in chunks:
            chunk["labels"] = [
                [row[p] for p in perm] if isinstance(row, list) else row
                for row in chunk["labels"]
            ]
    else:
        print("Label order matches model config.")

    chunks = [{k: v for k, v in c.items() if k in KEEP_KEYS} for c in chunks]
    print(f"{len(chunks)} held-out chunks from {len(holdout_ids)} documents.")
    return chunks, id2label


def evaluate_fold(model_path: str, chunks: List[Dict], id2label: Dict[int, str],
                  batch_size: int, fp16: bool = False) -> Dict:
    """Run one fold model over the holdout chunks and score it, excluding O."""
    from cardioner.multilabel.trainer import (
        MultiLabelDataCollatorForTokenClassification,
        MultiLabelTokenClassificationModelHF,
    )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if config.num_labels is None:
        config.num_labels = len(id2label)
    model = MultiLabelTokenClassificationModelHF(config)

    state_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(state_path):
        import safetensors.torch

        state_dict = safetensors.torch.load_file(state_path)
    else:
        state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"), map_location="cpu"
        )
    missing, _unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if "pooler" not in k]
    if real_missing:
        print(f"  WARNING: missing weights: {real_missing[:5]}")
    model = model.float().eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/token_eval_holdout",
            per_device_eval_batch_size=batch_size,
            report_to="none",
            use_cpu=not torch.cuda.is_available(),
            # Training ran with fp16=True, so matching it reproduces the
            # reported per-label values exactly. fp32 (the default here) is
            # numerically more accurate but differs in the ~4th decimal on
            # tokens whose sigmoid sits near the 0.5 boundary.
            fp16=fp16 and torch.cuda.is_available(),
            dataloader_drop_last=False,
        ),
        data_collator=MultiLabelDataCollatorForTokenClassification(
            tokenizer=tokenizer
        ),
        processing_class=tokenizer,
    )

    output = trainer.predict(test_dataset=copy.deepcopy(chunks))
    logits, labels = output.predictions, output.label_ids

    # Native multilabel decoding, identical to MultiLabelTrainer.compute_metrics
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()

    labels = labels.reshape(-1, labels.shape[-1])
    preds = preds.reshape(-1, preds.shape[-1])

    keep = ~np.all(labels == -100, axis=1)  # drop padding / special tokens
    labels = np.where(labels[keep] == -100, 0, labels[keep])
    preds = preds[keep]

    # --- the point of this script: score without the O column ---------------
    o_index = next((i for i, name in id2label.items() if name == "O"), None)
    cols = [i for i in sorted(id2label) if i != o_index]

    y_true, y_pred = labels[:, cols], preds[:, cols]

    result = {
        "n_tokens": int(labels.shape[0]),
        "excluded_class": id2label.get(o_index),
        "micro": {
            "Precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        },
        "macro": {
            "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        },
    }

    p_all = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_all = recall_score(y_true, y_pred, average=None, zero_division=0)
    f_all = f1_score(y_true, y_pred, average=None, zero_division=0)
    result["per_label"] = {
        id2label[col]: {
            "Precision": float(p_all[j]),
            "Recall": float(r_all[j]),
            "F1": float(f_all[j]),
        }
        for j, col in enumerate(cols)
    }

    del model
    torch.cuda.empty_cache()
    return result


def aggregate(fold_results: List[Dict]) -> Dict:
    """Mean / std across folds, mirroring aggregated_validation_results.json."""

    def agg(values: List[float]) -> Dict:
        return {
            "mean": stat.mean(values),
            "std": stat.stdev(values) if len(values) > 1 else 0.0,
        }

    out: Dict = {"micro": {}, "macro": {}, "per_label": {}}
    for scope in ("micro", "macro"):
        for metric in ("Precision", "Recall", "F1"):
            out[scope][metric] = agg([r[scope][metric] for r in fold_results])

    for label in fold_results[0]["per_label"]:
        out["per_label"][label] = {
            metric: agg([r["per_label"][label][metric] for r in fold_results])
            for metric in ("Precision", "Recall", "F1")
        }

    out["_metadata"] = {
        "n_folds": len(fold_results),
        "excluded_class": fold_results[0]["excluded_class"],
        "decoding": "sigmoid > 0.5 (native multilabel)",
        "evaluation_set": "held-out test documents (split_file['test_files'])",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bulk_file", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument(
        "--model_folder", required=True, help="Experiment dir containing fold_0..fold_N"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_type", default="paragraph")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--entity_types", nargs="+", default=None)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Evaluate in mixed precision, matching training (fp16=True). "
             "Reproduces the reported per-label values exactly; without it, "
             "fp32 is used and values differ by <0.001.",
    )
    args = parser.parse_args()

    # Directories only: fold_N_results.json also starts with "fold_".
    fold_dirs = sorted(
        (
            d
            for d in os.listdir(args.model_folder)
            if d.startswith("fold_")
            and os.path.isdir(os.path.join(args.model_folder, d))
        ),
        key=lambda d: int(d.split("_")[1]),
    )
    if not fold_dirs:
        raise SystemExit(f"No fold_* directories in {args.model_folder}")
    print(f"Found {len(fold_dirs)} folds: {fold_dirs}")

    # Tokenise once; every fold is scored on the identical chunk set.
    chunks, id2label = load_holdout_chunks(
        bulk_file=args.bulk_file,
        split_file=args.split_file,
        model_path=os.path.join(args.model_folder, fold_dirs[0]),
        chunk_size=args.chunk_size,
        chunk_type=args.chunk_type,
        max_length=args.max_length,
        entity_types=args.entity_types,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    fold_results = []
    for fold in fold_dirs:
        print(f"\n=== {fold} ===")
        result = evaluate_fold(
            model_path=os.path.join(args.model_folder, fold),
            chunks=chunks,
            id2label=id2label,
            batch_size=args.batch_size,
            fp16=args.fp16,
        )
        print(
            f"  micro F1 {result['micro']['F1']:.4f} | "
            f"macro F1 {result['macro']['F1']:.4f}  (O excluded)"
        )
        fold_results.append(result)
        with open(
            os.path.join(args.output_dir, f"{fold}_results.json"), "w", encoding="utf-8"
        ) as fw:
            json.dump(result, fw, indent=2)

    aggregated = aggregate(fold_results)
    agg_path = os.path.join(args.output_dir, "aggregated_holdout_token_results.json")
    with open(agg_path, "w", encoding="utf-8") as fw:
        json.dump(aggregated, fw, indent=2)

    print("\n" + "=" * 60)
    print("AGGREGATED (O excluded, held-out test set)")
    print("=" * 60)
    for scope in ("micro", "macro"):
        m = aggregated[scope]
        print(
            f"  {scope:6s} P {m['Precision']['mean']:.4f}  "
            f"R {m['Recall']['mean']:.4f}  F1 {m['F1']['mean']:.4f} "
            f"(±{m['F1']['std']:.4f})"
        )
    print(f"\nSaved to {agg_path}")


if __name__ == "__main__":
    main()
