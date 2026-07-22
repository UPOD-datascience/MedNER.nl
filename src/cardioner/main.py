import os
from os import environ

import spacy

# Load a spaCy model for tokenization
nlp = spacy.blank("nl")
environ["WANDB_MODE"] = "disabled"
environ["WANDB_DISABLED"] = "true"
environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import evaluate
import pandas as pd
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.utils import shuffle

# https://huggingface.co/learn/nlp-course/en/chapter7/2
from torch import bfloat16, cuda
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

print(sys.executable)
print(transformers.__file__)
print(transformers.__version__)


from cardioner import evaluation, model_merger, parse_performance_json, predictor
from cardioner.multiclass.trainer import ModelTrainer as MultiClassModelTrainer
from cardioner.multiclass.trainer import MultiHeadCRFTrainer, MultiHeadTrainer
from cardioner.multilabel.trainer import ModelTrainer as MultiLabelModelTrainer
from cardioner.utils import calculate_class_weights, merge_annotations, process_pipe

# Clear any existing CUDA context
cuda.empty_cache()

# Check if CUDA is available
print(f"CUDA available: {cuda.is_available()}")


def _extract_reference_results(corpus_data: List[Dict]) -> List[Dict]:
    """Extract reference entities from corpus data when `tags` are available."""
    ref_results = []
    for doc in corpus_data:
        doc_id = doc.get("id", "unknown")
        text = doc.get("text", "")
        ref_tags = doc.get("tags", None)

        if not isinstance(ref_tags, list):
            continue

        for _tag in ref_tags:
            start = _tag.get("start")
            end = _tag.get("end")
            label = _tag.get("tag")
            if start is None or end is None or label is None:
                continue

            ref_results.append(
                {
                    "filename": doc_id,
                    "label": label,
                    "start_span": start,
                    "end_span": end,
                    "text": text[start:end],
                }
            )

    return ref_results


def _split_text_with_indices(
    text: str, text_splitter: RecursiveCharacterTextSplitter
) -> List[Tuple[str, int, int]]:
    """Split text into chunks and map each chunk back to absolute character offsets."""
    raw_chunks = text_splitter.split_text(text)
    used_indices = set()
    chunks: List[Tuple[str, int, int]] = []

    for chunk_text in raw_chunks:
        start_index = text.find(chunk_text)
        while start_index in used_indices:
            start_index = text.find(chunk_text, start_index + 1)

        if start_index < 0:
            continue

        used_indices.add(start_index)
        end_index = start_index + len(chunk_text)
        chunks.append((chunk_text, start_index, end_index))

    if not chunks:
        chunks = [(text, 0, len(text))]

    return chunks


def _get_word_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Create word-level tokens and their exact char spans, aligned to original text."""
    matches = predictor.split_sentence_with_indices(text)
    words: List[str] = []
    spans: List[Tuple[int, int]] = []

    for m in matches:
        token_text = m.group().strip()
        if not token_text:
            continue
        words.append(token_text)
        spans.append((m.start(), m.end()))

    return words, spans


def _aggregate_word_level_bio_predictions(
    tagged_tokens: List[Dict],
    chunk_text: str,
    chunk_offset: int,
    doc_id: str,
    default_entity_type: Optional[str] = None,
) -> List[Dict]:
    """Aggregate BIO-tagged word-level predictions into span-level entities."""
    entities: List[Dict] = []
    current_entity: Optional[str] = None
    current_start: Optional[int] = None
    current_end: Optional[int] = None

    for token in tagged_tokens:
        tag = token["tag"]
        start = int(token["start"])
        end = int(token["end"])

        is_b = tag.startswith("B-") or tag == "B"
        is_i = tag.startswith("I-") or tag == "I"

        if is_b:
            if (
                current_entity is not None
                and current_start is not None
                and current_end is not None
            ):
                entities.append(
                    {
                        "filename": doc_id,
                        "label": current_entity,
                        "start_span": chunk_offset + current_start,
                        "end_span": chunk_offset + current_end,
                        "text": chunk_text[current_start:current_end],
                    }
                )

            current_entity = (
                tag[2:] if tag.startswith("B-") else (default_entity_type or "")
            )
            current_start = start
            current_end = end

        elif is_i:
            tag_type = tag[2:] if tag.startswith("I-") else (default_entity_type or "")
            if current_entity == tag_type and current_start is not None:
                current_end = end
            else:
                if (
                    current_entity is not None
                    and current_start is not None
                    and current_end is not None
                ):
                    entities.append(
                        {
                            "filename": doc_id,
                            "label": current_entity,
                            "start_span": chunk_offset + current_start,
                            "end_span": chunk_offset + current_end,
                            "text": chunk_text[current_start:current_end],
                        }
                    )

                current_entity = tag_type
                current_start = start
                current_end = end

        else:  # O
            if (
                current_entity is not None
                and current_start is not None
                and current_end is not None
            ):
                entities.append(
                    {
                        "filename": doc_id,
                        "label": current_entity,
                        "start_span": chunk_offset + current_start,
                        "end_span": chunk_offset + current_end,
                        "text": chunk_text[current_start:current_end],
                    }
                )
            current_entity = None
            current_start = None
            current_end = None

    if (
        current_entity is not None
        and current_start is not None
        and current_end is not None
    ):
        entities.append(
            {
                "filename": doc_id,
                "label": current_entity,
                "start_span": chunk_offset + current_start,
                "end_span": chunk_offset + current_end,
                "text": chunk_text[current_start:current_end],
            }
        )

    return entities


def _write_reference_and_sequence_scores(
    ref_results: List[Dict],
    pred_df: Optional[pd.DataFrame],
    output_dir: str,
    output_file_prefix: str = "",
) -> None:
    """Write reference TSV and sequence-level strict/relaxed metrics when possible."""
    ref_tsv_path = os.path.join(output_dir, f"{output_file_prefix}reference.tsv")
    sequence_result_path = os.path.join(
        output_dir, f"{output_file_prefix}sequence_result.json"
    )

    if not isinstance(ref_results, list) or len(ref_results) == 0:
        print("No reference entities found in the corpus")
        return

    df_ref = pd.DataFrame(ref_results)
    df_ref.to_csv(ref_tsv_path, sep="\t", index=False)

    if pred_df is not None and not pred_df.empty and "label" in pred_df.columns:
        # Keep original behavior: evaluate only labels present in predictions.
        df_ref_for_scoring = df_ref.loc[df_ref.label.isin(pred_df.label.unique())]
    else:
        df_ref_for_scoring = df_ref.iloc[0:0].copy()

    print(f"Reference results saved to {ref_tsv_path}")
    print(f"Total entities in reference data: {len(df_ref_for_scoring)}")

    if pred_df is None or pred_df.empty:
        print("No predictions available; skipping sequence scoring")
        return

    print(f"Performing sequence scoring and writing to {sequence_result_path}")
    res_by_cat_strict, micro_summ_strict, macro_summ_strict = (
        evaluation.calculate_metrics_strict(df_ref_for_scoring, pred_df)
    )
    res_by_cat_relaxed, micro_summ_relaxed, macro_summ_relaxed = (
        evaluation.calculate_metrics_relaxed(df_ref_for_scoring, pred_df)
    )

    final_dict = {
        "strict": {
            "per_category": res_by_cat_strict,
            "micro": micro_summ_strict,
            "macro": macro_summ_strict,
        },
        "relaxed": {
            "per_category": res_by_cat_relaxed,
            "micro": micro_summ_relaxed,
            "macro": macro_summ_relaxed,
        },
    }

    with open(sequence_result_path, "w", encoding="utf-8") as fw:
        json.dump(final_dict, fw)


def _normalize_tag_entry(tag_entry: Any, source_id: str) -> Dict:
    if not isinstance(tag_entry, dict):
        raise ValueError(
            f"Tag entry for document '{source_id}' must be a dict, got {type(tag_entry)}"
        )

    start = tag_entry.get("start", tag_entry.get("start_span"))
    end = tag_entry.get("end", tag_entry.get("end_span"))
    label = tag_entry.get(
        "tag",
        tag_entry.get("label", tag_entry.get("entity", tag_entry.get("entity_group"))),
    )

    if start is None or end is None or label is None:
        raise ValueError(
            f"Tag entry for document '{source_id}' is missing one of required keys "
            "(start/end/tag). Supported aliases: start_span/end_span and label/entity/entity_group. "
            f"Got: {tag_entry}"
        )

    return {
        "start": int(start),
        "end": int(end),
        "tag": str(label),
    }


def _parse_tags_field(raw_tags: Any, source_id: str, tags_column: str) -> List[Dict]:
    if raw_tags is None:
        return []

    parsed_tags = raw_tags

    if isinstance(raw_tags, str):
        text = raw_tags.strip()
        if text == "":
            return []

        try:
            parsed_tags = json.loads(text)
        except json.JSONDecodeError:
            # Fallback for jsonl-style strings where each line is one JSON object.
            try:
                parsed_tags = [
                    json.loads(line) for line in text.splitlines() if line.strip() != ""
                ]
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse tags from column '{tags_column}' for document '{source_id}': {e}"
                ) from e

    if isinstance(parsed_tags, dict):
        if "tags" in parsed_tags and isinstance(parsed_tags["tags"], list):
            parsed_tags = parsed_tags["tags"]
        else:
            raise ValueError(
                f"Expected a list of tags in column '{tags_column}' for document '{source_id}', "
                f"but got a dict without a 'tags' list: {parsed_tags}"
            )

    if not isinstance(parsed_tags, list):
        raise ValueError(
            f"Expected a list of tags in column '{tags_column}' for document '{source_id}', "
            f"but got {type(parsed_tags)}"
        )

    return [_normalize_tag_entry(tag_entry=t, source_id=source_id) for t in parsed_tags]


def _load_hf_ner_split(
    split_dataset: Dataset,
    split_name: str,
    text_column: str,
    tags_column: str | None,
    selector_column: str | None = None,
    selection: List[str] | None = None,
) -> List[Dict]:
    available_columns = set(split_dataset.column_names)
    required_columns = {text_column}
    if tags_column is not None:
        required_columns.add(tags_column)
    if selector_column is not None:
        required_columns.add(selector_column)

    missing_columns = sorted(
        [c for c in required_columns if c not in available_columns]
    )
    if missing_columns:
        raise ValueError(
            f"Missing required columns in split '{split_name}': {missing_columns}. "
            f"Available columns: {sorted(available_columns)}"
        )

    selected_values = set(selection or [])
    records: List[Dict] = []

    for idx, row_raw in enumerate(split_dataset):
        row = dict(row_raw)

        if selector_column is not None:
            selector_val = row.get(selector_column)
            if not (
                (selector_val in selected_values)
                or (str(selector_val) in selected_values)
            ):
                continue

        source_id = row.get("id", row.get("doc_id", f"{split_name}_{idx}"))
        source_id = str(source_id)

        text_val = row.get(text_column)
        if text_val is None:
            continue
        text = text_val if isinstance(text_val, str) else str(text_val)

        if tags_column is not None:
            tags = _parse_tags_field(
                raw_tags=row.get(tags_column),
                source_id=source_id,
                tags_column=tags_column,
            )
        else:
            tags = []

        records.append(
            {
                "id": source_id,
                "text": text,
                "tags": tags,
            }
        )

    print(
        f"Loaded {len(records)} documents from HF split '{split_name}' "
        f"(text_column='{text_column}', tags_column='{tags_column}')"
    )
    return records


def load_hf_corpora(
    dataset_name: str,
    dataset_config: str | None,
    text_column: str,
    tags_column: str,
    hf_token: str | None,
    selector_column: str | None = None,
    selection: List[str] | None = None,
    train_split: str = "train",
    validation_split: str = "validation",
    test_split: str = "test",
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    if selector_column is not None and (selection is None or len(selection) == 0):
        raise ValueError(
            "When selector_column is set, selection must be provided with one or more values."
        )
    if selector_column is None and selection is not None and len(selection) > 0:
        raise ValueError("selection was provided, but selector_column is not set.")

    print(
        f"Loading Hugging Face dataset '{dataset_name}'"
        + (f" (config='{dataset_config}')" if dataset_config else "")
    )
    ds_obj = load_dataset(dataset_name, name=dataset_config, token=hf_token)

    if not isinstance(ds_obj, DatasetDict):
        raise ValueError(
            f"Expected a DatasetDict when loading '{dataset_name}', got {type(ds_obj)}"
        )

    available_splits = sorted(ds_obj.keys())
    if train_split not in ds_obj:
        raise ValueError(
            f"Train split '{train_split}' not found in dataset '{dataset_name}'. "
            f"Available splits: {available_splits}"
        )

    train_records = _load_hf_ner_split(
        split_dataset=ds_obj[train_split],
        split_name=train_split,
        text_column=text_column,
        tags_column=tags_column,
        selector_column=selector_column,
        selection=selection,
    )

    validation_records: List[Dict] = []
    if validation_split in ds_obj:
        validation_records = _load_hf_ner_split(
            split_dataset=ds_obj[validation_split],
            split_name=validation_split,
            text_column=text_column,
            tags_column=tags_column,
            selector_column=selector_column,
            selection=selection,
        )
    else:
        print(
            f"Validation split '{validation_split}' not present in dataset. "
            "Proceeding without explicit validation split."
        )

    test_records: List[Dict] = []
    if test_split in ds_obj:
        test_records = _load_hf_ner_split(
            split_dataset=ds_obj[test_split],
            split_name=test_split,
            text_column=text_column,
            tags_column=tags_column,
            selector_column=selector_column,
            selection=selection,
        )
    else:
        print(
            f"Test split '{test_split}' not present in dataset. Proceeding without test split."
        )

    if len(train_records) == 0:
        raise ValueError(
            "HF dataset train split produced 0 records after parsing/filtering. "
            "Check text/tags columns and selector settings."
        )

    return train_records, validation_records, test_records


def load_hf_inference_corpus(
    dataset_name: str,
    dataset_config: str | None,
    text_column: str,
    tags_column: str | None,
    hf_token: str | None,
    selector_column: str | None = None,
    selection: List[str] | None = None,
    inference_split: str = "validation",
) -> List[Dict]:
    if selector_column is not None and (selection is None or len(selection) == 0):
        raise ValueError(
            "When selector_column is set, selection must be provided with one or more values."
        )
    if selector_column is None and selection is not None and len(selection) > 0:
        raise ValueError("selection was provided, but selector_column is not set.")

    print(
        f"Loading Hugging Face dataset '{dataset_name}' for inference"
        + (f" (config='{dataset_config}')" if dataset_config else "")
    )
    ds_obj = load_dataset(dataset_name, name=dataset_config, token=hf_token)

    if isinstance(ds_obj, DatasetDict):
        available_splits = sorted(ds_obj.keys())
        if inference_split not in ds_obj:
            raise ValueError(
                f"Inference split '{inference_split}' not found in dataset '{dataset_name}'. "
                f"Available splits: {available_splits}"
            )

        return _load_hf_ner_split(
            split_dataset=ds_obj[inference_split],
            split_name=inference_split,
            text_column=text_column,
            tags_column=tags_column,
            selector_column=selector_column,
            selection=selection,
        )

    if isinstance(ds_obj, Dataset):
        print(
            f"Dataset '{dataset_name}' loaded as a single split Dataset. "
            "Using it directly for inference."
        )
        return _load_hf_ner_split(
            split_dataset=ds_obj,
            split_name=inference_split,
            text_column=text_column,
            tags_column=tags_column,
            selector_column=selector_column,
            selection=selection,
        )

    raise ValueError(
        f"Unexpected dataset type when loading '{dataset_name}' for inference: {type(ds_obj)}"
    )


def inference(
    corpus_data: List[Dict],
    model_path: str,
    output_dir: str,
    output_file_prefix: str = "",
    lang: str = "nl",
    max_word_per_chunk: int | None = 125,
    trust_remote_code: bool = True,
    strategy: Literal["simple", "average", "first", "max"] = "simple",
    pipe: Literal["dt4h", "hf"] = "hf",
    dt4h_post_hoc_cleaning=True,
    dt4h_min_confidence=0.65,
    dt4h_batch_size: int = 4,
    dt4h_allow_numeric_tags: Optional[List[str]] = None,
):
    """
    Run inference on a corpus using a pre-trained model.
    Uses the simple aggregation from the huggingface tokenclassification pipeline

    Args:
        corpus_data: List of documents with 'id' and 'text' fields, and potentially tagged entities
        model_path: Path to the trained model directory
        output_dir: Directory to save output TSV
        lang: Language code for tokenization
        max_word_per_chunk: Maximum words per chunk for processing. If None,
            automatically calculated as tokenizer.model_max_length // 2
        trust_remote_code: Whether to trust remote code when loading model

    Returns:
        List of prediction results
    """

    # Create output tsv path
    output_tsv_path = os.path.join(output_dir, f"{output_file_prefix}_predictions.tsv")
    os.makedirs(output_dir, exist_ok=True)

    ref_results = _extract_reference_results(corpus_data)

    if pipe == "hf":
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, use_fast=True
            )
            print("Using fast tokenizer for inference.")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, use_fast=False
            )
            print("Fast tokenizer not available; falling back to slow tokenizer.")

        # Auto-detect max_word_per_chunk from tokenizer if not provided
        if max_word_per_chunk is None:
            max_word_per_chunk = tokenizer.model_max_length // 2
            assert max_word_per_chunk < 10000, (
                f"Calculated max_word_per_chunk ({max_word_per_chunk}) is >= 10,000. "
                f"This seems too large. Please specify --max_word_per_chunk explicitly."
            )
            print(
                f"Auto-detected max_word_per_chunk: {max_word_per_chunk} (from tokenizer max_length: {tokenizer.model_max_length})"
            )
        else:
            print(f"Max word per chunk is set to \t\t {max_word_per_chunk}")
            print(f"Max model length is \t\t\t{tokenizer.model_max_length}")

        print(f"Loading model from {model_path}...")
        print(f"Processing {len(corpus_data)} samples with NER pipeline...")
        print(f"Output will be saved to {output_tsv_path}")

        # Load the model for NER
        ner_pipeline = pipeline(
            "token-classification",
            stride=max_word_per_chunk,
            model=model_path,
            tokenizer=tokenizer,
            aggregation_strategy=strategy,
            trust_remote_code=trust_remote_code,
        )

        pred_results = []
        for doc in tqdm(corpus_data, desc="Running inference"):
            doc_id = doc.get("id", "unknown")
            text = doc.get("text", "")

            if not text:
                continue

            # Get predictions
            entities = process_pipe(
                text, ner_pipeline, max_word_per_chunk=max_word_per_chunk, lang=lang
            )

            for entity in entities:
                # Extract needed information
                tag = entity.get("entity_group", "").replace("B-", "").replace("I-", "")
                start_span = entity.get("start", 0)
                end_span = entity.get("end", 0)
                entity_text = entity.get("word", text[start_span:end_span])

                # Add to results
                pred_results.append(
                    {
                        "filename": doc_id,
                        "label": tag,
                        "start_span": start_span,
                        "end_span": end_span,
                        "text": entity_text,
                    }
                )
    else:
        ner_pipe = predictor.PredictionNER(
            model_checkpoint=model_path,
            revision=None,
            stride=max_word_per_chunk,
            lang=lang,
            allow_numeric_tags=dt4h_allow_numeric_tags,
        )
        pred_results = []
        for doc in tqdm(corpus_data, desc="Running inference"):
            doc_id = doc.get("id", "unknown")
            text = doc.get("text", "")
            # text = text.replace("\n", " ").replace("\t", " ")

            if not text:
                continue

            res = ner_pipe.do_prediction_batch(
                text,
                batch_size=dt4h_batch_size,
                confidence_threshold=dt4h_min_confidence,
                post_hoc_cleaning=dt4h_post_hoc_cleaning,
                trim_trailing_cutoff_words=True,
            )
            if len(res) > 0:
                for _res in res:
                    pred_results.append(
                        {
                            "filename": doc_id,
                            "label": _res["tag"],
                            "start_span": _res["start"],
                            "end_span": _res["end"],
                            "text": _res["text"],
                        }
                    )

    # Create DataFrame and save to TSV
    pred_df: Optional[pd.DataFrame] = None
    if isinstance(pred_results, list) and len(pred_results) > 0:
        pred_df = pd.DataFrame(pred_results)
        pred_df.to_csv(output_tsv_path, sep="\t", index=False)
        print(f"Predictions saved to {output_tsv_path}")
        print(f"Total entities predicted: {len(pred_results)}")
    else:
        print("No entities found in the corpus")

    _write_reference_and_sequence_scores(
        ref_results=ref_results,
        pred_df=pred_df,
        output_dir=output_dir,
        output_file_prefix=output_file_prefix,
    )

    return pred_results


def inference_multihead_crf(
    corpus_data: List[Dict],
    model_path: str,
    output_dir: str,
    output_file_prefix: str = "",
    lang: str = "nl",
    max_word_per_chunk: int | None = None,
    trust_remote_code: bool = True,
):
    """
    Run inference on a corpus using a pre-trained MultiHead CRF model.
    Uses the simple aggregation from the huggingface tokenclassification pipeline

    Args:
        corpus_data: List of documents with 'id' and 'text' fields
        model_path: Path to the trained model directory
        output_dir: Directory to save output TSV
        lang: Language code for tokenization
        max_word_per_chunk: Maximum words per chunk for processing. If None,
            automatically calculated as tokenizer.model_max_length // 2
        trust_remote_code: Whether to trust remote code when loading model

    Returns:
        List of prediction results
    """
    import pandas as pd
    import torch

    from cardioner.multiclass.modeling import TokenClassificationModelMultiHeadCRF

    # Create output tsv path
    output_tsv_path = os.path.join(output_dir, f"{output_file_prefix}predictions.tsv")
    os.makedirs(output_dir, exist_ok=True)

    ref_results = _extract_reference_results(corpus_data)

    # Load tokenizer (prefer fast tokenizer, fallback to slow)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=True
        )
        print("Using fast tokenizer for inference.")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=False
        )
        print("Fast tokenizer not available; falling back to slow tokenizer.")

    # Auto-detect max_word_per_chunk from tokenizer if not provided
    if max_word_per_chunk is None:
        max_word_per_chunk = tokenizer.model_max_length // 2
        assert max_word_per_chunk < 10000, (
            f"Calculated max_word_per_chunk ({max_word_per_chunk}) is >= 10,000. "
            f"This seems too large. Please specify --max_word_per_chunk explicitly."
        )
        print(
            f"Auto-detected max_word_per_chunk: {max_word_per_chunk} (from tokenizer max_length: {tokenizer.model_max_length})"
        )

    print(f"Loading MultiHead CRF model from {model_path}...")
    model = TokenClassificationModelMultiHeadCRF.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Get entity types and id2label from model config
    entity_types = model.config.entity_types
    id2label = model.config.id2label
    print(f"Entity types: {entity_types}")
    print(f"Processing {len(corpus_data)} samples...")
    print(f"Output will be saved to {output_tsv_path}")

    results = []

    for doc in tqdm(corpus_data, desc="Running inference"):
        doc_id = doc.get("id", "unknown")
        text = doc.get("text", "")

        if not text:
            continue

        chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="o200k_base",
            separators=["\n\n\n", "\n\n", "\n", " .", " !", " ?", " ،", " ,", " ", ""],
            keep_separator=True,
            chunk_size=max_word_per_chunk,
            chunk_overlap=0,
        )
        chunks = _split_text_with_indices(text, chunker)

        for chunk_text, chunk_offset, _chunk_end in chunks:
            if not chunk_text.strip():
                continue

            word_tokens, word_spans = _get_word_spans(chunk_text)
            if len(word_tokens) == 0:
                continue

            encoding = tokenizer(
                word_tokens,
                return_tensors="pt",
                is_split_into_words=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding=True,
            )
            word_ids = encoding.word_ids(batch_index=0)
            inputs = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            sorted_entity_types = sorted(entity_types)
            for ent_idx, entity_type in enumerate(sorted_entity_types):
                predictions = outputs[ent_idx][0]
                seen_words = set()
                tagged_tokens = []

                for token_idx, word_idx in enumerate(word_ids):
                    if word_idx is None or word_idx in seen_words:
                        continue

                    seen_words.add(word_idx)
                    pred_id = int(predictions[token_idx])
                    label = id2label.get(str(pred_id), id2label.get(pred_id, "O"))
                    start, end = word_spans[word_idx]
                    tagged_tokens.append(
                        {
                            "tag": label,
                            "start": int(start),
                            "end": int(end),
                        }
                    )

                if tagged_tokens:
                    results.extend(
                        _aggregate_word_level_bio_predictions(
                            tagged_tokens=tagged_tokens,
                            chunk_text=chunk_text,
                            chunk_offset=chunk_offset,
                            doc_id=doc_id,
                            default_entity_type=entity_type,
                        )
                    )

    # Create DataFrame and save to TSV
    pred_df: Optional[pd.DataFrame] = None
    if results:
        pred_df = pd.DataFrame(results)
        pred_df.to_csv(output_tsv_path, sep="\t", index=False)
        print(f"Predictions saved to {output_tsv_path}")
        print(f"Total entities found: {len(results)}")
    else:
        print("No entities found in the corpus")

    _write_reference_and_sequence_scores(
        ref_results=ref_results,
        pred_df=pred_df,
        output_dir=output_dir,
        output_file_prefix=output_file_prefix,
    )

    return results


def inference_multihead(
    corpus_data: List[Dict],
    model_path: str,
    output_dir: str,
    output_file_prefix: str = "",
    lang: str = "nl",
    max_word_per_chunk: int | None = None,
    trust_remote_code: bool = True,
):
    """
    Run inference on a corpus using a pre-trained MultiHead model (no CRF).

    Args:
        corpus_data: List of documents with 'id' and 'text' fields
        model_path: Path to the trained model directory
        output_dir: Directory to save output TSV
        lang: Language code for tokenization
        max_word_per_chunk: Maximum words per chunk for processing. If None,
            automatically calculated as tokenizer.model_max_length // 2
        trust_remote_code: Whether to trust remote code when loading model

    Returns:
        List of prediction results
    """
    import pandas as pd
    import torch

    from cardioner.multiclass.modeling import TokenClassificationModelMultiHead

    # Create output tsv path
    output_tsv_path = os.path.join(output_dir, f"{output_file_prefix}predictions.tsv")
    os.makedirs(output_dir, exist_ok=True)

    ref_results = _extract_reference_results(corpus_data)

    # Load tokenizer (prefer fast tokenizer, fallback to slow)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=True
        )
        print("Using fast tokenizer for inference.")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=False
        )
        print("Fast tokenizer not available; falling back to slow tokenizer.")

    # Auto-detect max_word_per_chunk from tokenizer if not provided
    if max_word_per_chunk is None:
        max_word_per_chunk = tokenizer.model_max_length // 2
        assert max_word_per_chunk < 10000, (
            f"Calculated max_word_per_chunk ({max_word_per_chunk}) is >= 10,000. "
            f"This seems too large. Please specify --max_word_per_chunk explicitly."
        )
        print(
            f"Auto-detected max_word_per_chunk: {max_word_per_chunk} (from tokenizer max_length: {tokenizer.model_max_length})"
        )

    print(f"Loading MultiHead model (no CRF) from {model_path}...")
    model = TokenClassificationModelMultiHead.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Get entity types and id2label from model config
    entity_types = sorted(model.config.entity_types)
    id2label = model.config.id2label
    if isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}

    print(f"Entity types: {entity_types}")
    print(f"ID to label mapping: {id2label}")

    # Create text splitter for chunking documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        separators=["\n\n\n", "\n\n", "\n", " .", " !", " ?", " ،", " ,", " ", ""],
        keep_separator=True,
        chunk_size=max_word_per_chunk,
        chunk_overlap=0,
    )

    results = []

    for doc in tqdm(corpus_data, desc="Processing documents"):
        doc_id = doc["id"]
        text = doc["text"]

        # Chunk the document with robust offset mapping
        chunks = _split_text_with_indices(text, text_splitter)

        for chunk_text, chunk_offset, _chunk_end in chunks:
            if not chunk_text.strip():
                continue

            word_tokens, word_spans = _get_word_spans(chunk_text)
            if len(word_tokens) == 0:
                continue

            encoding = tokenizer(
                word_tokens,
                return_tensors="pt",
                is_split_into_words=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding=True,
            )

            word_ids = encoding.word_ids(batch_index=0)

            # Move to device
            encoding = {k: v.to(device) for k, v in encoding.items()}

            # Get predictions (argmax decoding)
            with torch.no_grad():
                predictions = model(**encoding)

            # Process predictions for each entity type
            for ent_idx, entity_type in enumerate(entity_types):
                preds = predictions[ent_idx][0].cpu().tolist()
                seen_words = set()
                tagged_tokens = []

                for token_idx, word_idx in enumerate(word_ids):
                    if word_idx is None or word_idx in seen_words:
                        continue

                    seen_words.add(word_idx)
                    pred = preds[token_idx]
                    tag = id2label.get(pred, "O")
                    start, end = word_spans[word_idx]
                    tagged_tokens.append(
                        {
                            "tag": tag,
                            "start": int(start),
                            "end": int(end),
                        }
                    )

                if tagged_tokens:
                    results.extend(
                        _aggregate_word_level_bio_predictions(
                            tagged_tokens=tagged_tokens,
                            chunk_text=chunk_text,
                            chunk_offset=chunk_offset,
                            doc_id=doc_id,
                            default_entity_type=entity_type,
                        )
                    )

    # Create DataFrame and save to TSV
    pred_df: Optional[pd.DataFrame] = None
    if results:
        pred_df = pd.DataFrame(results)
        pred_df.to_csv(output_tsv_path, sep="\t", index=False)
        print(f"Predictions saved to {output_tsv_path}")
        print(f"Total entities found: {len(results)}")
    else:
        print("No entities found in the corpus")

    _write_reference_and_sequence_scores(
        ref_results=ref_results,
        pred_df=pred_df,
        output_dir=output_dir,
        output_file_prefix=output_file_prefix,
    )

    return results


def filter_tags(iob_data, tags, entity_types, multi_class) -> tuple | None:
    print(
        f"Entity types provided: {entity_types},\nuse these as a filter for the tokenized data/tags"
    )
    assert tags is not None, "Entity types provided, but no tags found."
    # filter tags list
    entity_types = [entity_type.upper() for entity_type in entity_types]
    tags = [tag.upper() for tag in tags]
    print(f"Extracted tags are {tags}, checking against {entity_types}")
    tags = ["O"] + [
        tag for tag in tags if any([entity_type in tag for entity_type in entity_types])
    ]

    if len(tags) == 1:
        return False

    # filter tokenized_data
    #
    filtered_tokenized_data = []
    for doc in iob_data:
        temp_dict = {
            "id": doc["id"],
            "gid": doc["gid"],
            "batch": doc.get("batch", "cardioccc"),
        }
        temp_tokens = []
        if multi_class:
            temp_tags = []
            for chindex, _tag in enumerate(doc["tags"]):
                temp_tokens.append(doc["tokens"][chindex])
                if any([entity_type in _tag for entity_type in entity_types]):
                    temp_tags.append(_tag)
                else:
                    # Token doesn't match entity filter, assign O
                    temp_tags.append("O")
        else:
            temp_tags = []
            for chindex, _tags in enumerate(doc["tags"]):
                # Filter the token's actual tags to only include matching entity types
                _temp_tags = [
                    t
                    for t in _tags
                    if any(entity_type in t.upper() for entity_type in entity_types)
                ]
                # Always keep the token, even if it has no entity tags (it will get O label)
                temp_tokens.append(doc["tokens"][chindex])
                temp_tags.append(
                    _temp_tags
                )  # Empty list will become O in tokenize_and_align_labels

        if len(temp_tags) > 0:
            temp_dict["tokens"] = temp_tokens
            temp_dict["tags"] = temp_tags
            filtered_tokenized_data.append(temp_dict)

    return filtered_tokenized_data, tags


def prepare(
    Model: str = "CLTL/MedRoBERTa.nl",
    corpus_train: List[Dict] | List[List[Dict]] | None = None,
    corpus_validation: List[Dict] | None = None,
    corpus_test: List[Dict] | List[List[Dict]] | None = None,
    annotation_loc: Optional[str] = None,
    label2id: Optional[Dict[str, int]] = None,
    id2label: Optional[Dict[int, str]] = None,
    chunk_size: int = 256,
    max_length: int = 514,
    chunk_type: Literal["standard", "centered", "paragraph"] = "standard",
    multi_class: bool = False,
    use_iob: bool = True,
    hf_token: str | None = None,
    use_multihead_crf: bool = False,
    entity_types: List[str] | None = None,
    strict_tag_set: bool = False,
    lang: str = "nl",
):
    # Multi-head CRF preparation
    if use_multihead_crf:
        from cardioner.multiclass.loader import (
            annotate_corpus_multihead,
            annotate_corpus_multihead_centered,
            get_entity_types_from_corpus,
            tokenize_and_align_labels_multihead,
        )

        return prepare_multihead(
            Model=Model,
            corpus_train=corpus_train,
            corpus_validation=corpus_validation,
            corpus_test=corpus_test,
            chunk_size=chunk_size,
            max_length=max_length,
            chunk_type=chunk_type,
            hf_token=hf_token,
            entity_types=entity_types,
            lang=lang,
        )

    if multi_class:
        from cardioner.multiclass.loader import (
            annotate_corpus_centered,
            annotate_corpus_paragraph,
            annotate_corpus_standard,
            tokenize_and_align_labels,
        )
    else:
        from cardioner.multilabel.loader import (
            annotate_corpus_centered,
            annotate_corpus_paragraph,
            annotate_corpus_standard,
            count_tokens_with_multiple_labels,
            tokenize_and_align_labels,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        Model,
        add_prefix_space=True,
        token=hf_token,
        rust_remote_code=False,
        force_download=True,
        local_files_only=False,
    )
    tokenizer.model_max_length = max_length

    model_config = AutoConfig.from_pretrained(
        Model, token=hf_token, trust_remote_code=True
    )
    max_model_length = model_config.max_position_embeddings  # 514
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)  # 2
    max_allowed_chunk_size = max_model_length - num_special_tokens  # 512

    # Run the transformation
    annotate_functions = {
        "standard": annotate_corpus_standard,
        "centered": annotate_corpus_centered,
        "paragraph": annotate_corpus_paragraph,
    }

    annotation_lang = "xx" if lang == "multi" else ("cs" if lang == "cz" else lang)

    annotate_kwargs = {
        "standard": {
            "chunk_size": chunk_size,
            "max_allowed_chunk_size": max_allowed_chunk_size,
            "IOB": use_iob,
            "lang": annotation_lang,
        },
        "paragraph": {
            "chunk_size": chunk_size,
            "max_allowed_chunk_size": max_allowed_chunk_size,
            "IOB": use_iob,
            "lang": annotation_lang,
        },
        "centered": {
            "chunk_size": chunk_size,
            "IOB": use_iob,
            "lang": annotation_lang,
        },
    }

    datasets = {
        "train": corpus_train,
        "test": corpus_test,
        "validation": corpus_validation,
    }

    # Remove datasets that are None
    datasets = {k: v for k, v in datasets.items() if v is not None}

    # Initialize variables
    iob_data_train = iob_data_test = iob_data_validation = []
    train_unique_tags: List[str] | None = None
    split_unique_tags: Dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    annotate_func = annotate_functions[chunk_type]
    kwargs = annotate_kwargs[chunk_type]

    skipped_count = 0
    processed_batches = 0
    for batch_id, corpus in datasets.items():
        processed_batches += 1
        iob_data, _unique_tags = annotate_func(corpus, batch_id=batch_id, **kwargs)

        if isinstance(entity_types, list):
            assert all([isinstance(s, str) for s in entity_types]), (
                f"Not all entity_types are strings {entity_types}"
            )
            res = filter_tags(iob_data, _unique_tags, entity_types, multi_class)
            if not res:
                skipped_count += 1
                continue
            else:
                iob_data, _unique_tags = res

        if _unique_tags is not None:
            split_unique_tags[batch_id] = set(_unique_tags)

        if batch_id == "train":
            iob_data_train = iob_data
            train_unique_tags = _unique_tags
        elif batch_id == "test":
            iob_data_test = iob_data
        elif batch_id == "validation":
            iob_data_validation = iob_data

    print(f"{skipped_count}/{processed_batches} batches skipped ")

    if train_unique_tags is None:
        raise ValueError("No train tags could be extracted from the training corpus.")

    train_tag_set = set(train_unique_tags)
    val_extra_tags = sorted(split_unique_tags.get("validation", set()) - train_tag_set)
    test_extra_tags = sorted(split_unique_tags.get("test", set()) - train_tag_set)

    if len(val_extra_tags) > 0:
        print(
            "Validation has tags not present in train. "
            f"Count={len(val_extra_tags)}. Tags={val_extra_tags}"
        )
    if len(test_extra_tags) > 0:
        print(
            "Test has tags not present in train. "
            f"Count={len(test_extra_tags)}. Tags={test_extra_tags}"
        )

    if strict_tag_set and (len(val_extra_tags) > 0 or len(test_extra_tags) > 0):
        raise ValueError(
            "strict_tag_set=True and non-train splits contain unseen tags. "
            f"validation_only={val_extra_tags}, test_only={test_extra_tags}"
        )

    # Default behavior: use train tag space only.
    unique_tags = list(train_unique_tags)

    # Ensure O stays at index 0 for compatibility.
    if "O" in unique_tags and unique_tags[0] != "O":
        unique_tags = ["O"] + [t for t in unique_tags if t != "O"]

    if (not strict_tag_set) and (len(val_extra_tags) > 0 or len(test_extra_tags) > 0):
        allowed_tags = set(unique_tags)

        def _coerce_non_train_tags_to_train_space(
            iob_data_split: List[Dict],
        ) -> List[Dict]:
            if multi_class:
                for doc in iob_data_split:
                    doc["tags"] = [
                        tag if tag in allowed_tags else "O" for tag in doc["tags"]
                    ]
            else:
                for doc in iob_data_split:
                    cleaned_token_tags = []
                    for token_tags in doc["tags"]:
                        if isinstance(token_tags, list):
                            cleaned_token_tags.append(
                                [tag for tag in token_tags if tag in allowed_tags]
                            )
                        else:
                            cleaned_token_tags.append([])
                    doc["tags"] = cleaned_token_tags
            return iob_data_split

        iob_data_validation = _coerce_non_train_tags_to_train_space(iob_data_validation)
        iob_data_test = _coerce_non_train_tags_to_train_space(iob_data_test)
        print(
            "Ignoring non-train tags in validation/test (mapped/removed to O space). "
            "Use --strict_tag_set to fail instead."
        )

    label2id = {l: int(c) for c, l in enumerate(unique_tags)}
    id2label = {int(c): l for c, l in enumerate(unique_tags)}

    print("Unique tags (train tag space): ", unique_tags)

    iob_data = iob_data_train + iob_data_test + iob_data_validation

    if multi_class == False:
        count_tokens_with_multiple_labels(iob_data)

    partial_tokenize_and_align_labels = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )

    iob_data_dataset = Dataset.from_list(iob_data)
    iob_data_dataset_tokenized = iob_data_dataset.map(
        partial_tokenize_and_align_labels,
        batched=True,
    )
    # given a max_length tokens we want to center the context around all spans in the documents and extract them as a seperate documents. Each separate extraction needs to get a separate sub_id.

    max_seq_length = max(
        len(entry["input_ids"]) for entry in iob_data_dataset_tokenized
    )
    print(f"Maximum sequence length after tokenization: {max_seq_length}")

    iob_data_dataset_tokenized_with_labels = []
    for entry in iob_data_dataset_tokenized:
        entry.update({"label2id": label2id, "id2label": id2label})
        iob_data_dataset_tokenized_with_labels.append(entry)

    if annotation_loc is not None:
        annotation_loc = annotation_loc.replace(
            ".jsonl", f"_chunk{ChunkSize}_{ChunkType}.jsonl"
        )
        with open(annotation_loc, "w", encoding="utf-8") as fw:
            for entry in iob_data_dataset_tokenized_with_labels:
                json.dump(entry, fw)
                fw.write("\n")

    return iob_data_dataset_tokenized_with_labels, unique_tags


def prepare_multihead(
    Model: str = "CLTL/MedRoBERTa.nl",
    corpus_train: List[Dict] | None = None,
    corpus_validation: List[Dict] | None = None,
    corpus_test: List[Dict] | None = None,
    chunk_size: int = 256,
    max_length: int = 514,
    chunk_type: Literal["standard", "centered", "paragraph"] = "standard",
    hf_token: str | None = None,
    entity_types: List[str] | None = None,
    lang: str = "nl",
):
    """
    Prepare data for Multi-Head CRF training.

    Each entity type gets its own BIO label sequence.
    """
    from cardioner.multiclass.loader import (
        annotate_corpus_multihead,
        annotate_corpus_multihead_centered,
        get_entity_types_from_corpus,
        tokenize_and_align_labels_multihead,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        Model,
        add_prefix_space=True,
        token=hf_token,
        trust_remote_code=False,
        force_download=True,
        local_files_only=False,
    )
    tokenizer.model_max_length = max_length

    model_config = AutoConfig.from_pretrained(Model, token=hf_token)
    max_model_length = model_config.max_position_embeddings
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    max_allowed_chunk_size = max_model_length - num_special_tokens

    # Auto-detect entity types if not provided
    if entity_types is None and corpus_train is not None:
        entity_types = get_entity_types_from_corpus(corpus_train)
        print(f"Auto-detected entity types: {entity_types}")

    if not entity_types:
        raise ValueError("entity_types must be provided or detectable from corpus")

    # BIO labels for each head
    bio_label2id = {"O": 0, "B": 1, "I": 2}
    bio_id2label = {0: "O", 1: "B", 2: "I"}

    datasets = {
        "train": corpus_train,
        "test": corpus_test,
        "validation": corpus_validation,
    }
    datasets = {k: v for k, v in datasets.items() if v is not None}

    iob_data_train = []
    iob_data_test = []
    iob_data_validation = []

    # Select annotation function based on chunk type
    annotation_lang = "xx" if lang == "multi" else ("cs" if lang == "cz" else lang)

    if chunk_type == "centered":
        annotate_func = annotate_corpus_multihead_centered
        kwargs = {"chunk_size": chunk_size, "lang": annotation_lang}
    else:
        annotate_func = annotate_corpus_multihead
        kwargs = {
            "chunk_size": chunk_size,
            "max_allowed_chunk_size": max_allowed_chunk_size,
            "lang": annotation_lang,
        }

    for batch_id, corpus in datasets.items():
        iob_data, _ = annotate_func(
            corpus, entity_types=entity_types, batch_id=batch_id, **kwargs
        )

        if batch_id == "train":
            iob_data_train = iob_data
        elif batch_id == "test":
            iob_data_test = iob_data
        elif batch_id == "validation":
            iob_data_validation = iob_data

    print(f"Entity types for Multi-Head CRF: {entity_types}")
    print(f"BIO labels per head: {bio_label2id}")

    iob_data = iob_data_train + iob_data_test + iob_data_validation

    partial_tokenize = partial(
        tokenize_and_align_labels_multihead,
        tokenizer=tokenizer,
        entity_types=entity_types,
        max_length=max_length,
    )

    iob_data_dataset = Dataset.from_list(iob_data)
    iob_data_dataset_tokenized = iob_data_dataset.map(
        partial_tokenize,
        batched=True,
    )

    max_seq_length = max(
        len(entry["input_ids"]) for entry in iob_data_dataset_tokenized
    )
    print(f"Maximum sequence length after tokenization: {max_seq_length}")

    # Add metadata to each entry
    iob_data_dataset_tokenized_with_labels = []
    for entry in iob_data_dataset_tokenized:
        entry.update(
            {
                "label2id": bio_label2id,
                "id2label": bio_id2label,
                "entity_types": entity_types,
                "is_multihead": True,
            }
        )
        iob_data_dataset_tokenized_with_labels.append(entry)

    return iob_data_dataset_tokenized_with_labels, entity_types


def train(
    tokenized_data_train: List[Dict],
    tokenized_data_test: List[Dict] | None,
    tokenized_data_validation: List[Dict],
    force_splitter: bool = False,
    only_first_split: bool = False,
    Model: str = "CLTL/MedRoBERTa.nl",
    Splits: List[Tuple[List[str], List[str]]] | int | None = 5,
    output_dir: str = "../output",
    max_length: int = 514,
    num_epochs: int = 10,
    batch_size: int = 20,
    profile: bool = False,
    multi_class: bool = False,
    use_crf: bool = False,
    weight_decay: float = 0.001,
    learning_rate: float = 1e-4,
    accumulation_steps: int = 1,
    hf_token: str = None,
    freeze_backbone: bool = False,
    classifier_hidden_layers: tuple | None = None,
    classifier_dropout: float = 0.1,
    use_class_weights: bool = False,
    use_multihead_crf: bool = False,
    use_multihead: bool = False,
    number_of_layers_per_head: int = 1,
    crf_reduction: str = "mean",
    word_level: bool = False,
):
    # Check if this is multi-head data
    is_multihead = (
        tokenized_data_train[0].get("is_multihead", False)
        if tokenized_data_train
        else False
    )

    if is_multihead or use_multihead_crf or use_multihead:
        return train_multihead(
            tokenized_data_train=tokenized_data_train,
            tokenized_data_test=tokenized_data_test,
            tokenized_data_validation=tokenized_data_validation,
            force_splitter=force_splitter,
            only_first_split=only_first_split,
            Model=Model,
            Splits=Splits,
            output_dir=output_dir,
            max_length=max_length,
            num_epochs=num_epochs,
            batch_size=batch_size,
            profile=profile,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            accumulation_steps=accumulation_steps,
            hf_token=hf_token,
            freeze_backbone=freeze_backbone,
            classifier_dropout=classifier_dropout,
            number_of_layers_per_head=number_of_layers_per_head,
            crf_reduction=crf_reduction,
            use_crf=use_multihead_crf and not use_multihead,
            use_class_weights=use_class_weights,
        )
    label2id = tokenized_data_train[0]["label2id"]
    id2label = tokenized_data_train[0]["id2label"]

    label2id_tr = tokenized_data_train[0]["label2id"]
    if (tokenized_data_validation is not None) & (len(tokenized_data_validation) > 0):
        label2id_vl = tokenized_data_validation[0]["label2id"]
        assert label2id_tr == label2id_vl, (
            "Label2id mismatch between train, validation."
        )

    if (tokenized_data_test is not None) & (len(tokenized_data_test) > 0):
        label2d_test = tokenized_data_train[0]["label2id"]
        assert label2id_tr == label2d_test, "Label2id mismatch between train, test."

    label2id = {str(k): int(v) for k, v in label2id.items()}
    id2label = {int(k): str(v) for k, v in id2label.items()}

    # extract class weights
    if use_class_weights:
        class_weights = calculate_class_weights(
            tokenized_data_train, label2id, multi_class
        )
    else:
        class_weights = None

    # Ensure labels are correct
    num_labels = len(label2id)
    for entry in tokenized_data_train:
        labels = entry["labels"]
        if multi_class == False:
            for token_labels in labels:
                if isinstance(token_labels, list):
                    assert len(token_labels) == num_labels, (
                        "Mismatch in label dimensions, in train set."
                    )
                else:
                    assert token_labels == -100, "Labels should be lists or -100."
        else:
            assert all(
                [label in [-100] + list(range(num_labels)) for label in labels]
            ), f"Labels should be in range (0,{num_labels})  or -100."

    for entry in tokenized_data_validation:
        labels = entry["labels"]
        if multi_class == False:
            for token_labels in labels:
                if isinstance(token_labels, list):
                    assert len(token_labels) == num_labels, (
                        "Mismatch in label dimensions, in validation set."
                    )
                else:
                    assert token_labels == -100, "Labels should be lists or -100."
        else:
            assert all(
                [label in [-100] + list(range(num_labels)) for label in labels]
            ), f"Labels should be in range (0,{num_labels})  or -100."

    if (
        (tokenized_data_validation is None)
        | (isinstance(Splits, list))
        | ((tokenized_data_validation is not None) & (force_splitter == True))
    ):
        print("Using cross-validation for model training and validation..")
        groups = [entry["gid"] for entry in tokenized_data_train]
        shuffled_data, shuffled_groups = shuffle(
            tokenized_data_train, groups, random_state=42
        )
        if isinstance(Splits, int):
            splitter = GroupKFold(n_splits=Splits)
            SplitList = list(splitter.split(shuffled_data, groups=shuffled_groups))
        elif Splits is not None:
            # we have to turn the List[Tuple[List['id'], List['id']]] into List[Tuple[List[int], List[int]]] based on shuffled_data which is a list of {'gid':.. }
            # where gid=id
            gid_id = defaultdict(list)
            for k, d in enumerate(shuffled_data):
                gid_id[d["gid"]].append(k)
            SplitList = [
                tuple(
                    [
                        [idx for ind in T[0] for idx in gid_id[ind]],
                        [jdx for jind in T[1] for jdx in gid_id[jind]],
                    ],
                )
                for T in Splits
            ]

        print(f"Splitting data into {len(SplitList)} folds")
        if only_first_split:
            SplitList = SplitList[:1]
            print("Only running the first split (only_first_split=True)")
        for k, (train_idx, test_idx) in enumerate(SplitList):
            if multi_class:
                TrainClass = MultiClassModelTrainer(
                    label2id=label2id,
                    id2label=id2label,
                    tokenizer=None,
                    model=Model,
                    use_crf=use_crf,
                    output_dir=f"{output_dir}/fold_{k}",
                    max_length=max_length,
                    num_train_epochs=num_epochs,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    learning_rate=learning_rate,
                    gradient_accumulation_steps=accumulation_steps,
                    freeze_backbone=freeze_backbone,
                    classifier_hidden_layers=classifier_hidden_layers,
                    classifier_dropout=classifier_dropout,
                    hf_token=hf_token,
                    class_weights=class_weights,
                )
            else:
                TrainClass = MultiLabelModelTrainer(
                    label2id=label2id,
                    id2label=id2label,
                    tokenizer=None,
                    model=Model,
                    output_dir=f"{output_dir}/fold_{k}",
                    max_length=max_length,
                    num_train_epochs=num_epochs,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    learning_rate=learning_rate,
                    gradient_accumulation_steps=accumulation_steps,
                    hf_token=hf_token,
                    freeze_backbone=freeze_backbone,
                    classifier_hidden_layers=classifier_hidden_layers,
                    classifier_dropout=classifier_dropout,
                    class_weights=class_weights,
                    word_level=word_level,
                )

            print(f"Training on split {k}")
            train_data = [shuffled_data[i] for i in train_idx]
            test_data = [shuffled_data[i] for i in test_idx]

            fold_output_dir = os.path.join(output_dir, f"fold_{k}")
            os.makedirs(fold_output_dir, exist_ok=True)
            split_export = {
                "train_ids": [d["id"] for d in train_data],
                "train_gids": [d["gid"] for d in train_data],
                "test_ids": [d["id"] for d in test_data],
                "test_gids": [d["gid"] for d in test_data],
            }
            if (tokenized_data_validation is not None) and (
                len(tokenized_data_validation) > 0
            ):
                split_export["validation_ids"] = [
                    d["id"] for d in tokenized_data_validation
                ]
                split_export["validation_gids"] = [
                    d["gid"] for d in tokenized_data_validation
                ]
            split_path = os.path.join(fold_output_dir, "split.json")
            with open(split_path, "w", encoding="utf-8") as fw:
                json.dump(split_export, fw, indent=2)

            if (tokenized_data_validation is not None) and (
                len(tokenized_data_validation) > 0
            ):
                TrainClass.train(
                    train_data=train_data,
                    test_data=test_data,
                    eval_data=tokenized_data_validation,
                    profile=profile,
                )
            else:
                TrainClass.train(
                    train_data=train_data,
                    test_data=test_data,
                    eval_data=test_data,
                    profile=profile,
                )
        # perform model merger
        # ######################
        print(100 * "=")
        print(
            f"Performing SLERP model merging using the chordal method in {output_dir}"
        )
        list_of_model_locations = model_merger.path_parser(output_dir)

        # remove checkpoint-xx folders
        list_of_model_locations = [
            path
            for path in list_of_model_locations
            if not path.split("/")[-1].startswith("checkpoint-")
        ]
        try:
            print(
                f"Merging models from the following folders: {list_of_model_locations}"
            )

            new_state_dict = model_merger.average_state_dict_advanced(
                list_of_model_locations, method="chordal"
            )
            model_config = AutoConfig.from_pretrained(
                list_of_model_locations[0], trust_remote_code=True
            )
            model_averaged = AutoModelForTokenClassification.from_config(
                config=model_config,
                trust_remote_code=True,
            )
            missing, unexpected = model_averaged.load_state_dict(
                new_state_dict, strict=False
            )
            if missing:
                print("[load_state_dict] Missing keys:", missing)
            if unexpected:
                print("[load_state_dict] Unexpected keys:", unexpected)

            model_averaged = model_averaged.to(bfloat16)

            merged_path = os.path.join(output_dir, "merged_model")
            os.makedirs(merged_path, exist_ok=True)
            model_averaged.save_pretrained(merged_path)
        except Exception as e:
            print(f"An error occurred during model merging: {e}")

        # get performance aggregations
        #
        collected_dict = parse_performance_json.parse_dir(output_dir)
        json.dump(
            collected_dict,
            open(
                os.path.join(output_dir, "collected_results.json"),
                "w",
                encoding="latin1",
            ),
        )
        aggregated_dict = parse_performance_json.get_aggregates(collected_dict)
        json.dump(
            aggregated_dict,
            open(
                os.path.join(output_dir, "aggregated_results.json"),
                "w",
                encoding="latin1",
            ),
        )

    elif (tokenized_data_validation is not None) and (tokenized_data_test is not None):
        print(
            "Using preset train/test/validation split for model training and validation.."
        )
        if multi_class:
            TrainClass = MultiClassModelTrainer(
                label2id=label2id,
                id2label=id2label,
                tokenizer=None,
                model=Model,
                use_crf=use_crf,
                output_dir=output_dir,
                max_length=max_length,
                num_train_epochs=num_epochs,
                batch_size=batch_size,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                gradient_accumulation_steps=accumulation_steps,
                freeze_backbone=freeze_backbone,
                classifier_hidden_layers=classifier_hidden_layers,
                classifier_dropout=classifier_dropout,
                hf_token=hf_token,
                class_weights=class_weights,
            )
        else:
            TrainClass = MultiLabelModelTrainer(
                label2id=label2id,
                id2label=id2label,
                tokenizer=None,
                model=Model,
                output_dir=output_dir,
                max_length=max_length,
                num_train_epochs=num_epochs,
                batch_size=batch_size,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                gradient_accumulation_steps=accumulation_steps,
                hf_token=hf_token,
                freeze_backbone=freeze_backbone,
                classifier_hidden_layers=classifier_hidden_layers,
                classifier_dropout=classifier_dropout,
                class_weights=class_weights,
            )

        print("Training on full dataset")
        TrainClass.train(
            train_data=tokenized_data_train,
            test_data=tokenized_data_test,
            eval_data=tokenized_data_validation,
            profile=profile,
        )
    else:
        raise ValueError(
            "No validation data provided, and no cross-validation splits provided. Please provide either a validation set or a split file."
        )


def train_multihead(
    tokenized_data_train: List[Dict],
    tokenized_data_test: List[Dict] | None,
    tokenized_data_validation: List[Dict],
    force_splitter: bool = False,
    only_first_split: bool = False,
    Model: str = "CLTL/MedRoBERTa.nl",
    Splits: List[Tuple[List[str], List[str]]] | int | None = 5,
    output_dir: str = "../output",
    max_length: int = 514,
    num_epochs: int = 10,
    batch_size: int = 20,
    profile: bool = False,
    weight_decay: float = 0.001,
    learning_rate: float = 1e-4,
    accumulation_steps: int = 1,
    hf_token: str = None,
    freeze_backbone: bool = False,
    classifier_dropout: float = 0.1,
    number_of_layers_per_head: int = 1,
    crf_reduction: str = "mean",
    use_crf: bool = True,
    use_class_weights: bool = False,
):
    """
    Train a Multi-Head model for multiple entity types.

    Each entity type gets its own classification head.
    If use_crf=True, also adds a CRF layer per head.
    """
    # Extract entity types and labels from data
    entity_types = tokenized_data_train[0].get("entity_types", [])
    label2id = tokenized_data_train[0]["label2id"]
    id2label = tokenized_data_train[0]["id2label"]

    if not entity_types:
        raise ValueError(
            "entity_types not found in tokenized data. Did you use prepare() with use_multihead_crf=True?"
        )

    print("=" * 60)
    print(f"Training Multi-Head {'CRF ' if use_crf else ''}Model")
    print(f"Entity types: {entity_types}")
    print(f"BIO labels: {label2id}")
    print(f"Use CRF: {use_crf}")
    print("=" * 60)

    # Validate labels
    label2id = {str(k): int(v) for k, v in label2id.items()}
    id2label = {int(k): str(v) for k, v in id2label.items()}

    if (
        (tokenized_data_validation is None)
        | (isinstance(Splits, list))
        | ((tokenized_data_validation is not None) & (force_splitter == True))
    ):
        print(
            "Using cross-validation for multi-head CRF model training and validation.."
        )
        groups = [entry["gid"] for entry in tokenized_data_train]
        shuffled_data, shuffled_groups = shuffle(
            tokenized_data_train, groups, random_state=42
        )
        if isinstance(Splits, int):
            splitter = GroupKFold(n_splits=Splits)
            SplitList = list(splitter.split(shuffled_data, groups=shuffled_groups))
        elif Splits is not None:
            # we have to turn the List[Tuple[List['id'], List['id']]] into List[Tuple[List[int], List[int]]] based on shuffled_data which is a list of {'gid':.. }
            # where gid=id
            gid_id = defaultdict(list)
            for k, d in enumerate(shuffled_data):
                gid_id[d["gid"]].append(k)
            SplitList = [
                tuple(
                    [
                        [idx for ind in T[0] for idx in gid_id[ind]],
                        [jdx for jind in T[1] for jdx in gid_id[jind]],
                    ],
                )
                for T in Splits
            ]

        print(f"Splitting data into {len(SplitList)} folds")
        if only_first_split:
            SplitList = SplitList[:1]
            print("Only running the first split (only_first_split=True)")
        for k, (train_idx, test_idx) in enumerate(SplitList):
            if use_crf:
                trainer = MultiHeadCRFTrainer(
                    entity_types=entity_types,
                    label2id=label2id,
                    id2label=id2label,
                    tokenizer=None,
                    model=Model,
                    batch_size=batch_size,
                    max_length=max_length,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    num_train_epochs=num_epochs,
                    output_dir=f"{output_dir}/fold_{k}",
                    hf_token=hf_token,
                    freeze_backbone=freeze_backbone,
                    gradient_accumulation_steps=accumulation_steps,
                    number_of_layers_per_head=number_of_layers_per_head,
                    classifier_dropout=classifier_dropout,
                    crf_reduction=crf_reduction,
                )
            else:
                trainer = MultiHeadTrainer(
                    entity_types=entity_types,
                    label2id=label2id,
                    id2label=id2label,
                    tokenizer=None,
                    model=Model,
                    batch_size=batch_size,
                    max_length=max_length,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    num_train_epochs=num_epochs,
                    output_dir=f"{output_dir}/fold_{k}",
                    hf_token=hf_token,
                    freeze_backbone=freeze_backbone,
                    gradient_accumulation_steps=accumulation_steps,
                    number_of_layers_per_head=number_of_layers_per_head,
                    classifier_dropout=classifier_dropout,
                    use_class_weights=use_class_weights,
                )

            print(f"Training on split {k}")
            train_data = [shuffled_data[i] for i in train_idx]
            test_data = [shuffled_data[i] for i in test_idx]

            fold_output_dir = os.path.join(output_dir, f"fold_{k}")
            os.makedirs(fold_output_dir, exist_ok=True)
            split_export = {
                "train_ids": [d["id"] for d in train_data],
                "train_gids": [d["gid"] for d in train_data],
                "test_ids": [d["id"] for d in test_data],
                "test_gids": [d["gid"] for d in test_data],
            }
            if (
                (tokenized_data_validation is not None)
                and (len(tokenized_data_validation) > 0)
                and (force_splitter == True)
            ):
                split_export["validation_ids"] = [
                    d["id"] for d in tokenized_data_validation
                ]
                split_export["validation_gids"] = [
                    d["gid"] for d in tokenized_data_validation
                ]
            split_path = os.path.join(fold_output_dir, "split.json")
            with open(split_path, "w", encoding="utf-8") as fw:
                json.dump(split_export, fw, indent=2)

            if (
                (tokenized_data_validation is not None)
                and (len(tokenized_data_validation) > 0)
                and (force_splitter == True)
            ):
                trainer.train(
                    train_data=train_data,
                    test_data=test_data,
                    eval_data=tokenized_data_validation,
                    profile=profile,
                )
            else:
                trainer.train(
                    train_data=train_data,
                    test_data=[],
                    eval_data=tokenized_data_validation
                    if (tokenized_data_validation is not None)
                    and (len(tokenized_data_validation) > 0)
                    else test_data,
                    profile=profile,
                )

        # get performance aggregations
        #
        collected_dict = parse_performance_json.parse_dir(output_dir)
        json.dump(
            collected_dict,
            open(
                os.path.join(output_dir, "collected_results.json"),
                "w",
                encoding="latin1",
            ),
        )
        aggregated_dict = parse_performance_json.get_aggregates(collected_dict)
        json.dump(
            aggregated_dict,
            open(
                os.path.join(output_dir, "aggregated_results.json"),
                "w",
                encoding="latin1",
            ),
        )

    elif (tokenized_data_validation is not None) and (tokenized_data_test is not None):
        print(
            f"Using preset train/test/validation split for multi-head {'CRF ' if use_crf else ''}model training and validation.."
        )
        # Create trainer
        if use_crf:
            trainer = MultiHeadCRFTrainer(
                entity_types=entity_types,
                label2id=label2id,
                id2label=id2label,
                tokenizer=None,
                model=Model,
                batch_size=batch_size,
                max_length=max_length,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_train_epochs=num_epochs,
                output_dir=output_dir,
                hf_token=hf_token,
                freeze_backbone=freeze_backbone,
                gradient_accumulation_steps=accumulation_steps,
                number_of_layers_per_head=number_of_layers_per_head,
                classifier_dropout=classifier_dropout,
                crf_reduction=crf_reduction,
            )
        else:
            trainer = MultiHeadTrainer(
                entity_types=entity_types,
                label2id=label2id,
                id2label=id2label,
                tokenizer=None,
                model=Model,
                batch_size=batch_size,
                max_length=max_length,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_train_epochs=num_epochs,
                output_dir=output_dir,
                hf_token=hf_token,
                freeze_backbone=freeze_backbone,
                gradient_accumulation_steps=accumulation_steps,
                number_of_layers_per_head=number_of_layers_per_head,
                classifier_dropout=classifier_dropout,
                use_class_weights=use_class_weights,
            )

        # Train
        print("Training on full dataset")
        trainer.train(
            train_data=tokenized_data_train,
            test_data=tokenized_data_test,
            eval_data=tokenized_data_validation,
            profile=profile,
        )

        print(f"Multi-Head {'CRF ' if use_crf else ''}model saved to {output_dir}")
    else:
        raise ValueError(
            "No validation data provided, and no cross-validation splits provided. Please provide either a validation set or a split file."
        )


if __name__ == "__main__":
    """
        take in .jsonl with:
        {'tags': [{'start': int, 'end': int, 'tag': str}], 'text':str, 'id': str}

        and output .jsonl with tokenized and aligned data
    """
    # TODO: add gradient accumulation
    argparsers = argparse.ArgumentParser()
    argparsers.add_argument("--model", type=str, default="CLTL/MedRoBERTa.nl")
    argparsers.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["es", "nl", "en", "it", "ro", "sv", "cz", "multi"],
    )
    argparsers.add_argument("--corpus_train", type=str, required=False)
    argparsers.add_argument("--corpus_validation", type=str, required=False)
    argparsers.add_argument(
        "--hf_dataset",
        type=str,
        required=False,
        help="Hugging Face dataset id (e.g. ai4privacy/pii-masking-openpii-1.5m)",
    )
    argparsers.add_argument(
        "--hf_dataset_config",
        type=str,
        required=False,
        default=None,
        help="Optional Hugging Face dataset config/subset name.",
    )
    argparsers.add_argument(
        "--hf_train_split",
        type=str,
        default="train",
        help="HF split to use as training data.",
    )
    argparsers.add_argument(
        "--hf_validation_split",
        type=str,
        default="validation",
        help="HF split to use as validation data (if present); also used as default inference split for HF inference corpora.",
    )
    argparsers.add_argument(
        "--hf_test_split",
        type=str,
        default="test",
        help="HF split to use as test data (if present).",
    )
    argparsers.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Required with --hf_dataset: column containing raw text.",
    )
    argparsers.add_argument(
        "--tags_column",
        type=str,
        default=None,
        help="Required with --hf_dataset: column containing span tags JSON/JSONL.",
    )
    argparsers.add_argument(
        "--selector_column",
        type=str,
        default=None,
        help="Optional HF column used to filter rows (e.g. language).",
    )
    argparsers.add_argument(
        "--selection",
        type=str,
        nargs="+",
        default=None,
        help="Allowed values for --selector_column (required when selector_column is set).",
    )
    argparsers.add_argument("--split_file", type=str, required=False)
    argparsers.add_argument("--annotation_loc", type=str, required=False)
    argparsers.add_argument("--output_dir", type=str, default="output")
    argparsers.add_argument("--parse_annotations", action="store_true", default=False)
    argparsers.add_argument("--train_model", action="store_true", default=False)
    argparsers.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the transformer backbone and train only the classifier head",
    )
    argparsers.add_argument("--chunk_size", type=int, default=None)
    argparsers.add_argument(
        "--chunk_type",
        type=str,
        default="standard",
        choices=["standard", "centered", "paragraph"],
    )
    argparsers.add_argument("--max_token_length", type=int, default=514)
    argparsers.add_argument("--num_epochs", type=int, default=10)
    argparsers.add_argument("--num_labels", type=int, default=9)
    argparsers.add_argument("--learning_rate", type=float, default=1e-4)
    argparsers.add_argument("--weight_decay", type=float, default=0.01)
    argparsers.add_argument("--batch_size", type=int, default=16)
    argparsers.add_argument("--accumulation_steps", type=int, default=1)
    argparsers.add_argument("--num_splits", type=int, default=5)
    argparsers.add_argument("--hf_token", type=str, default=None)
    argparsers.add_argument("--multiclass", action="store_true", default=False)
    argparsers.add_argument("--use_crf", action="store_true", default=False)
    argparsers.add_argument("--profile", action="store_true", default=False)
    argparsers.add_argument("--force_splitter", action="store_true", default=False)
    argparsers.add_argument(
        "--only_first_split",
        action="store_true",
        default=False,
        help="If using a splitter/cross-validation, run only the first split",
    )
    argparsers.add_argument("--write_annotations", action="store_true", default=False)
    argparsers.add_argument("--without_iob_tagging", action="store_true", default=False)
    argparsers.add_argument(
        "--classifier_hidden_layers", type=int, nargs="+", default=None
    )
    argparsers.add_argument("--classifier_dropout", type=float, default=0.1)
    argparsers.add_argument("--use_class_weights", action="store_true", default=False)
    argparsers.add_argument(
        "--strict_tag_set",
        action="store_true",
        default=False,
        help="Require validation/test to use exactly the train tag set. If not set, unseen val/test tags are ignored (mapped to O space).",
    )
    argparsers.add_argument("--word_level", action="store_true", default=False)
    argparsers.add_argument("--output_test_tsv", action="store_true", default=False)
    argparsers.add_argument(
        "--use_multihead_crf",
        action="store_true",
        default=False,
        help="Use Multi-Head CRF model with separate heads per entity type",
    )
    argparsers.add_argument(
        "--use_multihead",
        action="store_true",
        default=False,
        help="Use Multi-Head model (no CRF) with separate heads per entity type",
    )
    argparsers.add_argument(
        "--entity_types",
        type=str,
        nargs="+",
        default=None,
        help="Entity types to use (e.g., DRUG DISEASE SYMPTOM). Used for filtering tags and for Multi-Head models. Auto-detected if not provided.",
    )
    argparsers.add_argument(
        "--number_of_layers_per_head",
        type=int,
        default=1,
        help="Number of dense layers per head in Multi-Head CRF",
    )
    argparsers.add_argument(
        "--crf_reduction",
        type=str,
        default="mean",
        choices=["mean", "sum", "token_mean", "none"],
        help="CRF loss reduction mode for Multi-Head CRF",
    )
    argparsers.add_argument(
        "--inference_only",
        action="store_true",
        default=False,
        help="Run inference only using a pre-trained model (no training)",
    )
    argparsers.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a pre-trained model for inference (required if --inference_only is set)",
    )
    argparsers.add_argument(
        "--corpus_inference",
        type=str,
        default=None,
        help="Path to corpus file/directory for inference, or a Hugging Face dataset id (uses corpus_validation if not set)",
    )
    argparsers.add_argument(
        "--inference_filter_file",
        type=str,
        default=None,
        help="Path to a .txt file containing one document ID per line. If provided, only these IDs are kept for inference.",
    )
    argparsers.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Trust remote code when loading custom models",
    )
    argparsers.add_argument(
        "--inference_strategy",
        choices=["simple", "average", "first", "max"],
        default="simple",
        help="Inference strategy for the model",
    )
    argparsers.add_argument(
        "--inference_pipe",
        choices=["hf", "dt4h"],
        default="hf",
        help="Use the standard pipeline [hf] for token classification or our custom inference class [dt4h], not that in the latter case the inference strategy is moot. For now [dt4h] only works for multilabel/multiclass, not CRF",
    )
    argparsers.add_argument(
        "--inference_batch_size",
        type=int,
        default=4,
        help="Batch size for dt4h inference (PredictionNER).",
    )
    argparsers.add_argument(
        "--allow_numeric_tags",
        type=str,
        nargs="*",
        default=["AGE"],
        help="Entity tags allowed to be numeric-only in dt4h post-hoc cleaning (default: AGE).",
    )
    argparsers.add_argument(
        "--inference_stride",
        type=int,
        default=125,
        help="Stride for inference.",
    )
    argparsers.add_argument(
        "--output_file_prefix", default="inference_", type=str, help="Prefix for the inference results file."
    )

    args = argparsers.parse_args()

    tokenized_data = None
    tags = None

    _model = args.model
    corpus_train = args.corpus_train
    corpus_validation = args.corpus_validation
    hf_dataset = args.hf_dataset
    hf_dataset_config = args.hf_dataset_config
    hf_train_split = args.hf_train_split
    hf_validation_split = args.hf_validation_split
    hf_test_split = args.hf_test_split
    text_column = args.text_column
    tags_column = args.tags_column
    selector_column = args.selector_column
    selection = args.selection
    split_file = args.split_file
    force_splitter = args.force_splitter
    only_first_split = args.only_first_split
    _annotation_loc = args.annotation_loc
    parse_annotations = args.parse_annotations
    train_model = args.train_model
    hf_token = args.hf_token
    freeze_backbone = args.freeze_backbone
    classifier_hidden_layers = (
        tuple(args.classifier_hidden_layers) if args.classifier_hidden_layers else None
    )
    classifier_dropout = args.classifier_dropout
    lang = args.lang

    if args.without_iob_tagging:
        use_iob = False
        print(
            "WARNING: you are training without the IOB-tagging scheme. Ensure this is correct."
        )
    else:
        use_iob = True

    if not args.split_file:
        print("WARNING: you are training without a split file. Ensure this is correct.")

    if (args.word_level) and (args.multiclass):
        print("WARNING: word_level on works for multilabel, ignoring..")

    # Handle inference-only mode
    if args.inference_only:
        assert args.model_path is not None, (
            "Model path (--model_path) is required for inference-only mode"
        )
        assert os.path.isdir(args.model_path), (
            f"Model path {args.model_path} does not exist or is not a directory"
        )

        # Determine corpus for inference
        # Preference order: explicit inference corpus -> HF dataset arg -> validation -> train
        corpus_inference = args.corpus_inference or hf_dataset or corpus_validation or corpus_train
        assert corpus_inference is not None, (
            "Corpus for inference is required (--corpus_inference, --hf_dataset, --corpus_validation, or --corpus_train)"
        )

        # Load corpus from local path or Hugging Face dataset id
        if os.path.isdir(corpus_inference):
            corpus_inference_list = merge_annotations(corpus_inference)
        elif os.path.isfile(corpus_inference):
            corpus_inference_list = []
            with open(corpus_inference, "r", encoding="utf-8") as fr:
                content = fr.read()

            # Check if file is JSON or JSONL
            content = content.strip()
            if content.startswith("["):
                # JSON format: load entire file and extract test_ids/val_ids
                corpus_data = json.loads(content)
                if isinstance(corpus_data, dict):
                    if isinstance(corpus_data.get("documents"), list):
                        corpus_inference_list = corpus_data["documents"]
                    else:
                        raise ValueError(
                            f"Inference corpus file '{corpus_inference}' contains a JSON object, "
                            "but a list of documents was expected. "
                            "If this is a split file, pass it via --split_file instead."
                        )
                else:
                    corpus_inference_list = corpus_data
            else:
                # JSONL format: parse line by line
                for line in content.split("\n"):
                    if line.strip():
                        corpus_inference_list.append(json.loads(line))
        else:
            if text_column is None:
                raise ValueError(
                    "When --corpus_inference references a Hugging Face dataset, "
                    "--text_column is required."
                )

            print(
                f"Interpreting --corpus_inference='{corpus_inference}' as a Hugging Face "
                f"dataset id and loading split '{hf_validation_split}'."
            )
            try:
                corpus_inference_list = load_hf_inference_corpus(
                    dataset_name=corpus_inference,
                    dataset_config=hf_dataset_config,
                    text_column=text_column,
                    tags_column=tags_column,
                    hf_token=hf_token,
                    selector_column=selector_column,
                    selection=selection,
                    inference_split=hf_validation_split,
                )
            except Exception as e:
                raise ValueError(
                    f"Corpus reference '{corpus_inference}' is neither an existing local path "
                    f"nor a loadable Hugging Face dataset id. Original error: {e}"
                ) from e

            if tags_column is None:
                print(
                    "No --tags_column provided for HF inference corpus; reference scoring "
                    "will be skipped."
                )

        # If split_file is provided, filter to only test_files
        if split_file is not None:
            assert os.path.isfile(split_file), (
                f"Split file {split_file} does not exist."
            )
            with open(split_file, "r", encoding="utf-8") as fr:
                split_data = json.load(fr)

            _test_ids = split_data.get("test_gids", []) or split_data.get(
                "test_ids", []
            )
            test_file_ids = [entry.strip(".txt") for entry in _test_ids]

            original_count = len(corpus_inference_list)
            corpus_inference_list = [
                entry for entry in corpus_inference_list if entry["id"] in test_file_ids
            ]

            print(
                f"Filtered corpus using split_file test_files: {original_count} -> {len(corpus_inference_list)} documents"
            )

            if len(corpus_inference_list) == 0:
                print(
                    f"WARNING: No documents matched test_files from split_file. "
                    f"Expected IDs (first 5): {test_file_ids[:5]}"
                )

        # Optional ID-based filtering for inference corpus
        if args.inference_filter_file is not None:
            assert args.inference_filter_file.endswith(".txt"), (
                f"Inference filter file must be a .txt file, got: {args.inference_filter_file}"
            )
            assert os.path.isfile(args.inference_filter_file), (
                f"Inference filter file {args.inference_filter_file} does not exist."
            )

            with open(args.inference_filter_file, "r", encoding="utf-8") as fr:
                filter_ids = {
                    line.strip().removesuffix(".txt")
                    for line in fr
                    if line.strip() != ""
                }

            original_count = len(corpus_inference_list)
            corpus_inference_list = [
                entry for entry in corpus_inference_list if entry["id"] in filter_ids
            ]

            print(
                f"Filtered corpus using inference_filter_file: {original_count} -> {len(corpus_inference_list)} documents"
            )

            if len(corpus_inference_list) == 0:
                preview_ids = list(filter_ids)[:5]
                print(
                    f"WARNING: No documents matched IDs from inference_filter_file. "
                    f"Expected IDs (first 5): {preview_ids}"
                )

        print(
            f"Running inference-only mode with {len(corpus_inference_list)} documents"
        )

        # Detect if model is MultiHead CRF by checking config
        config_path = os.path.join(args.model_path, "config.json")
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

        if is_multihead_crf:
            print(50 * "=", flush=True)
            print(
                "Detected MultiHead CRF model, using specialized inference...",
                flush=True,
            )
            print(50 * "=", flush=True)
            inference_multihead_crf(
                corpus_data=corpus_inference_list,
                model_path=args.model_path,
                output_dir=args.output_dir,
                output_file_prefix=args.output_file_prefix,
                lang=lang,
                max_word_per_chunk=None,  # Auto-detect from tokenizer
                trust_remote_code=True,  # Always true for multihead CRF
            )
        elif is_multihead:
            print(50 * "=", flush=True)
            print(
                "Detected MultiHead model (no CRF), using specialized inference...",
                flush=True,
            )
            print(50 * "=", flush=True)
            inference_multihead(
                corpus_data=corpus_inference_list,
                model_path=args.model_path,
                output_dir=args.output_dir,
                output_file_prefix=args.output_file_prefix,
                lang=lang,
                max_word_per_chunk=None,  # Auto-detect from tokenizer
                trust_remote_code=True,  # Always true for multihead
            )
        else:
            # Run standard inference
            print(50 * "=")
            print("Running standard inference multiclass or multilabel")
            print(50 * "=")
            inference(
                corpus_data=corpus_inference_list,
                model_path=args.model_path,
                output_dir=args.output_dir,
                output_file_prefix=args.output_file_prefix,
                lang=lang,
                max_word_per_chunk=args.inference_stride,  # Auto-detect from tokenizer
                trust_remote_code=args.trust_remote_code,
                strategy=args.inference_strategy,
                pipe=args.inference_pipe,
                dt4h_batch_size=args.inference_batch_size,
                dt4h_allow_numeric_tags=args.allow_numeric_tags,
            )

        print("Inference completed!")
        exit(0)

    assert (_annotation_loc is not None) | (parse_annotations is not None), (
        "Either provide an annotation location or set parse_annotations to True"
    )
    assert (train_model is True) | (parse_annotations is True), (
        "Either parse annotations or train the model, or both..do something!"
    )

    use_hf_dataset = hf_dataset is not None

    if use_hf_dataset and corpus_train is not None:
        raise ValueError("Provide either --hf_dataset or --corpus_train, not both.")

    if use_hf_dataset:
        if text_column is None or tags_column is None:
            raise ValueError(
                "When --hf_dataset is used, both --text_column and --tags_column are required."
            )

        corpus_train_list, corpus_validation_list, corpus_test_list = load_hf_corpora(
            dataset_name=hf_dataset,
            dataset_config=hf_dataset_config,
            text_column=text_column,
            tags_column=tags_column,
            hf_token=hf_token,
            selector_column=selector_column,
            selection=selection,
            train_split=hf_train_split,
            validation_split=hf_validation_split,
            test_split=hf_test_split,
        )

        if (
            (split_file is None)
            and (not force_splitter)
            and (len(corpus_validation_list) == 0)
        ):
            print(
                "No validation split available for HF dataset and no split file provided. "
                "Enabling --force_splitter automatically."
            )
            force_splitter = True

    else:
        assert corpus_train is not None, (
            "Corpus_train is required unless --hf_dataset is provided"
        )
        assert (
            ((corpus_train is not None) and (corpus_validation is not None))
            | ((split_file is not None) and (corpus_train is not None))
            | ((corpus_train is not None) and (force_splitter))
        ), "Either provide a split file or a train and validation corpus"

        corpus_train_path = corpus_train
        if corpus_train_path is None:
            raise ValueError("Corpus_train is required unless --hf_dataset is provided")

        if os.path.isdir(corpus_train_path):
            # go through jsons and merge into one
            corpus_train_list = merge_annotations(corpus_train_path)
        else:
            assert os.path.isfile(corpus_train_path), (
                f"Corpus_train file {corpus_train_path} does not exist."
            )
            # read jsonl
            corpus_train_list = []
            with open(corpus_train_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    corpus_train_list.append(json.loads(line))

        corpus_validation_list = None
        if corpus_validation is not None:
            if os.path.isdir(corpus_validation):
                corpus_validation_list = merge_annotations(corpus_validation)
            else:
                assert os.path.isfile(corpus_validation), (
                    f"Corpus_validation file {corpus_validation} does not exist."
                )
                corpus_validation_list = []
                with open(corpus_validation, "r", encoding="utf-8") as fr:
                    for line in fr:
                        corpus_validation_list.append(json.loads(line))

        corpus_test_list = []

    if split_file is not None:
        assert os.path.isfile(split_file), f"Split_file {split_file} does not exist."

    ################################################
    ################################################

    has_validation_corpus = (corpus_validation_list is not None) and (
        len(corpus_validation_list) > 0
    )

    if (split_file is not None) and has_validation_corpus:
        print(
            "Split file and validation corpus provided, ignoring validation corpus in favor of split file."
        )
        corpus_validation_list = None
        corpus_validation = None
        has_validation_corpus = False

    if force_splitter:
        if has_validation_corpus:
            print(
                "Force splitter is on, and validation set is provided, therefore, ignoring split file."
            )
            split_file = None
        elif split_file is not None:
            print(
                "Force splitter is on, and split file is provided, therefore, ignoring the test in the split file (if present)."
            )

    if split_file is not None:
        with open(split_file, "r", encoding="utf-8") as fr:
            split_data = json.load(fr)
            # TODO: check if the split file is correct, seems redundant to have separate entries for class/language.
            corpus_folds = split_data["folds"]
            corpus_validation_ids = [
                entry.strip(".txt") for entry in split_data.get("test_files", [])
            ]  # split_data #[lang]["validation"]["symp"]

        corpus_train_id_lists = [
            [entry.strip(".txt") for entry in fold["train_files"]]
            for fold in corpus_folds
        ]

        corpus_test_id_lists = [
            [entry.strip(".txt") for entry in fold["val_files"]]
            for fold in corpus_folds
        ]

        splits = list(zip(corpus_train_id_lists, corpus_test_id_lists))

        # corpus_training_lists = [
        #     [entry for entry in corpus_train_list if entry["id"] in corpus_train_ids]
        #     for corpus_train_ids in corpus_train_id_lists
        # ]

        # corpus_test_lists = [
        #     [entry for entry in corpus_train_list if entry["id"] in corpus_test_ids]
        #     for corpus_test_ids in corpus_test_id_lists
        # ]
        corpus_test_list = []
        corpus_validation_list = [
            entry for entry in corpus_train_list if entry["id"] in corpus_validation_ids
        ]

        # print overview of counts per fold
        print(100 * "=")
        print("Overview of counts per fold:")
        print(100 * "=")
        for k, fold in enumerate(corpus_folds):
            print(f"Fold {k}:")
            print(f"  Train: {len(fold['train_files'])}")
            print(f"  Validation: {len(fold['val_files'])}")
        print(f"  Test: {len(corpus_validation_list)}")
        print(100 * "=")
        print(100 * "=")
    else:
        splits = None
        if not use_hf_dataset:
            corpus_test_list = []

    OutputDir = args.output_dir
    ChunkSize = args.max_token_length if args.chunk_size is None else args.chunk_size
    max_length = args.max_token_length
    ChunkType = args.chunk_type
    num_labels = args.num_labels
    num_splits = args.num_splits
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    profile = args.profile
    multi_class = args.multiclass
    use_crf = args.use_crf
    weight_decay = args.weight_decay
    learning_rate = args.learning_rate
    accumulation_steps = args.accumulation_steps
    use_multihead_crf = args.use_multihead_crf
    use_multihead = args.use_multihead
    strict_tag_set = args.strict_tag_set
    entity_types = args.entity_types
    number_of_layers_per_head = args.number_of_layers_per_head
    crf_reduction = args.crf_reduction

    if not splits:
        splits = num_splits

    if args.write_annotations == False:
        _annotation_loc = None

    if parse_annotations:
        print("Loading and prepping data..")
        tokenized_data, tags = prepare(
            Model=_model,
            corpus_train=corpus_train_list,
            corpus_validation=corpus_validation_list,
            corpus_test=corpus_test_list,
            annotation_loc=_annotation_loc,
            chunk_size=ChunkSize,
            chunk_type=ChunkType,
            max_length=max_length,
            multi_class=multi_class,
            use_iob=use_iob,
            hf_token=hf_token,
            use_multihead_crf=use_multihead_crf or use_multihead,
            entity_types=entity_types,
            strict_tag_set=strict_tag_set,
            lang=lang,
        )

    if train_model:
        print("Training the model..")
        if tokenized_data is None:
            with open(_annotation_loc, "r", encoding="utf-8") as fr:
                tokenized_data = [json.loads(line) for line in fr]

        # Prefer the label space inferred from parsed data over CLI --num_labels.
        if (
            isinstance(tokenized_data, list)
            and len(tokenized_data) > 0
            and (not (use_multihead_crf or use_multihead))
            and isinstance(tokenized_data[0], dict)
            and ("label2id" in tokenized_data[0])
            and isinstance(tokenized_data[0]["label2id"], dict)
        ):
            inferred_num_labels = len(tokenized_data[0]["label2id"])
            if inferred_num_labels != num_labels:
                print(
                    f"Overriding --num_labels={num_labels} with inferred label count={inferred_num_labels} from tokenized data."
                )
                num_labels = inferred_num_labels

        # check if input_ids and labels have the same length, are smaller than max_length and if the labels are within range
        for entry in tokenized_data:
            if not (use_multihead_crf or use_multihead):
                assert len(entry["input_ids"]) == len(entry["labels"]), (
                    f"Input_ids and labels have different lengths for entry {entry['id']}. {len(entry['input_ids'])}/{len(entry['labels'])}"
                )
            else:
                for entry_labels in entry["labels"].values():
                    assert len(entry["input_ids"]) == len(entry_labels), (
                        f"Input_ids and labels have different lengths for entry {entry['id']}. {len(entry['input_ids'])}/{len(entry_labels)}"
                    )

            assert len(entry["input_ids"]) <= max_length, (
                f"Input_ids are longer than max_length for entry {entry['id']}"
            )

            # NOTE: we intentionally skip strict label-range assertions here.
            # Label spaces are derived during parsing/tokenization and can differ
            # from static CLI expectations when using external datasets.

        # TODO: this is extremely ugly and needs to be refactored
        tokenized_data_train = [
            entry for entry in tokenized_data if entry["batch"] == "train"
        ]
        tokenized_data_test = [
            entry for entry in tokenized_data if entry["batch"] == "test"
        ]
        tokenized_data_validation = [
            entry for entry in tokenized_data if entry["batch"] == "validation"
        ]

        print(
            f"We have {len(tokenized_data_train)} docs in the train set, {len(tokenized_data_test)} in the test set, and {len(tokenized_data_validation)} in the validation set"
        )

        if len(tokenized_data_train) == 0:
            tokenized_data_train = tokenized_data

        if (len(tokenized_data_validation) == 0) and (len(tokenized_data_test) > 0):
            tokenized_data_validation = tokenized_data_test

        # train-> split-> validation (if train is available, validation is available, and force_splitter = True)
        # train-> test -> validation (if train is available, test is available, and validation is available)
        # [x] train-> validation (if train is available, and validation is available)
        train(
            tokenized_data_train,
            tokenized_data_test,
            tokenized_data_validation,
            force_splitter=force_splitter,
            only_first_split=only_first_split,
            Model=_model,
            Splits=splits,
            output_dir=OutputDir,
            max_length=max_length,
            num_epochs=num_epochs,
            batch_size=batch_size,
            profile=profile,
            multi_class=multi_class,
            use_crf=use_crf,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            accumulation_steps=accumulation_steps,
            hf_token=hf_token,
            freeze_backbone=freeze_backbone,
            classifier_hidden_layers=classifier_hidden_layers,
            classifier_dropout=classifier_dropout,
            use_class_weights=args.use_class_weights,
            use_multihead_crf=use_multihead_crf,
            use_multihead=use_multihead,
            number_of_layers_per_head=number_of_layers_per_head,
            crf_reduction=crf_reduction,
            word_level=args.word_level,
        )

        # Create output tsv for test
        #
        # name	tag	start_span	end_span	text
        # casos_clinicos_cardiologia286	MEDICATION	429	438	warfarine
        # casos_clinicos_cardiologia286	MEDICATION	905	914	warfarine
        # ...
        if args.output_test_tsv:
            # Run inference on validation corpus using the trained model
            if corpus_validation_list is not None:
                inference(
                    corpus_data=corpus_validation_list,
                    model_path=OutputDir,
                    output_dir=OutputDir,
                    output_file_prefix=args.output_file_prefix,
                    lang=lang,
                    max_word_per_chunk=args.inference_stride,  # Auto-detect from tokenizer
                    trust_remote_code=args.trust_remote_code,
                    dt4h_allow_numeric_tags=args.allow_numeric_tags,
                )
            else:
                print("No validation corpus available for test TSV output")
