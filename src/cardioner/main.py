import os
from os import environ

import spacy

# Load a spaCy model for tokenization
nlp = spacy.blank("nl")
environ["WANDB_MODE"] = "disabled"
environ["WANDB_DISABLED"] = "true"
environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple

import evaluate
import pandas as pd
import transformers
from datasets import Dataset
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
    output_tsv_path = os.path.join(output_dir, f"{output_file_prefix}predictions.tsv")
    ref_tsv_path = os.path.join(output_dir, f"{output_file_prefix}reference.tsv")
    os.makedirs(output_dir, exist_ok=True)

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
        ref_results = []
        for doc in tqdm(corpus_data, desc="Running inference"):
            doc_id = doc.get("id", "unknown")
            text = doc.get("text", "")

            # if doc.get("tags", None) is not None add ref_results
            ref_tags = doc.get("tags", None)
            if isinstance(ref_tags, list):
                for _tag in ref_tags:
                    entity_text = text[_tag["start"] : _tag["end"]]
                    ref_results.append(
                        {
                            "filename": doc_id,
                            "label": _tag["tag"],
                            "start_span": _tag["start"],
                            "end_span": _tag["end"],
                            "text": entity_text,
                        }
                    )

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
        )
        pred_results = []
        ref_results = []
        for doc in tqdm(corpus_data, desc="Running inference"):
            doc_id = doc.get("id", "unknown")
            text = doc.get("text", "")
            # text = text.replace("\n", " ").replace("\t", " ")

            # if doc.get("tags", None) is not None add ref_results
            ref_tags = doc.get("tags", None)
            if isinstance(ref_tags, list):
                for _tag in ref_tags:
                    entity_text = text[_tag["start"] : _tag["end"]]
                    ref_results.append(
                        {
                            "filename": doc_id,
                            "label": _tag["tag"],
                            "start_span": _tag["start"],
                            "end_span": _tag["end"],
                            "text": entity_text,
                        }
                    )

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
    if isinstance(pred_results, list) and len(pred_results) > 0:
        df = pd.DataFrame(pred_results)
        df.to_csv(output_tsv_path, sep="\t", index=False)
        print(f"Predictions saved to {output_tsv_path}")
        print(f"Total entities predicted: {len(pred_results)}")
    else:
        print("No entities found in the corpus")

    if isinstance(ref_results, list) and len(ref_results) > 0:
        df_ref = pd.DataFrame(ref_results)
        df_ref.to_csv(ref_tsv_path, sep="\t", index=False)

        # remove entity classes that are not present in the prediction df
        df_ref = df_ref.loc[df_ref.label.isin(df.label.unique())]

        print(f"Reference results saved to {ref_tsv_path}")
        print(f"Total entities in reference data: {len(df_ref)}")
    else:
        print("No entities found in the corpus")

    # scoring, if possible
    if len(ref_results) > 0:
        print(
            f"Performing sequence scoring and writing to {output_dir}/{output_file_prefix}sequence_result.json"
        )
        res_by_cat_strict, micro_summ_strict, macro_summ_strict = (
            evaluation.calculate_metrics_strict(df_ref, df)
        )
        res_by_cat_relaxed, micro_summ_relaxed, macro_summ_relaxed = (
            evaluation.calculate_metrics_relaxed(df_ref, df)
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
        json.dump(
            final_dict,
            open(f"{output_dir}/{output_file_prefix}sequence_result.json", "w"),
        )

    return pred_results


def inference_multihead_crf(
    corpus_data: List[Dict],
    model_path: str,
    output_dir: str,
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
    from multiclass.modeling import TokenClassificationModelMultiHeadCRF

    # Create output tsv path
    output_tsv_path = os.path.join(output_dir, "predictions.tsv")
    os.makedirs(output_dir, exist_ok=True)

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

        # Tokenize with offset mapping
        # Process in chunks if text is too long
        words = text.split()
        chunks = []
        current_chunk_words = []
        current_start = 0

        for i, word in enumerate(words):
            current_chunk_words.append(word)
            if len(current_chunk_words) >= max_word_per_chunk:
                chunk_text = " ".join(current_chunk_words)
                chunk_start = text.find(chunk_text, current_start)
                chunks.append((chunk_text, chunk_start))
                current_start = chunk_start + len(chunk_text)
                current_chunk_words = []

        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunk_start = text.find(chunk_text, current_start)
            chunks.append((chunk_text, chunk_start))

        if not chunks:
            chunks = [(text, 0)]

        for chunk_text, chunk_offset in chunks:
            # Tokenize
            inputs = tokenizer(
                chunk_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )

            offset_mapping = inputs.pop("offset_mapping")[0]
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)

            # outputs is a list of predictions per entity type (sorted by entity type)
            sorted_entity_types = sorted(entity_types)

            for ent_idx, entity_type in enumerate(sorted_entity_types):
                predictions = outputs[ent_idx][0]  # Get first batch item

                # Convert predictions to entities
                current_entity = None
                current_start = None
                current_end = None

                for token_idx, (pred_id, (start, end)) in enumerate(
                    zip(predictions, offset_mapping)
                ):
                    if start == end:  # Skip special tokens
                        continue

                    pred_id = int(pred_id)
                    label = id2label.get(str(pred_id), id2label.get(pred_id, "O"))

                    if label.startswith("B-"):
                        # Save previous entity if exists
                        if current_entity is not None:
                            entity_text = chunk_text[current_start:current_end]
                            results.append(
                                {
                                    "filename": doc_id,
                                    "label": current_entity,
                                    "start_span": chunk_offset + current_start,
                                    "end_span": chunk_offset + current_end,
                                    "text": entity_text,
                                }
                            )
                        # Start new entity
                        current_entity = label[2:]  # Remove B- prefix
                        current_start = int(start)
                        current_end = int(end)

                    elif label.startswith("I-") and current_entity is not None:
                        tag = label[2:]  # Remove I- prefix
                        if tag == current_entity:
                            current_end = int(end)
                        else:
                            # Different entity type, save previous and start new
                            entity_text = chunk_text[current_start:current_end]
                            results.append(
                                {
                                    "filename": doc_id,
                                    "label": current_entity,
                                    "start_span": chunk_offset + current_start,
                                    "end_span": chunk_offset + current_end,
                                    "text": entity_text,
                                }
                            )
                            current_entity = tag
                            current_start = int(start)
                            current_end = int(end)

                    else:  # O tag or I- without B-
                        if current_entity is not None:
                            entity_text = chunk_text[current_start:current_end]
                            results.append(
                                {
                                    "filename": doc_id,
                                    "label": current_entity,
                                    "start_span": chunk_offset + current_start,
                                    "end_span": chunk_offset + current_end,
                                    "text": entity_text,
                                }
                            )
                            current_entity = None
                            current_start = None
                            current_end = None

                # Don't forget the last entity
                if current_entity is not None:
                    entity_text = chunk_text[current_start:current_end]
                    results.append(
                        {
                            "filename": doc_id,
                            "label": current_entity,
                            "start_span": chunk_offset + current_start,
                            "end_span": chunk_offset + current_end,
                            "text": entity_text,
                        }
                    )

    # Create DataFrame and save to TSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_tsv_path, sep="\t", index=False)
        print(f"Predictions saved to {output_tsv_path}")
        print(f"Total entities found: {len(results)}")
    else:
        print("No entities found in the corpus")

    return results


def inference_multihead(
    corpus_data: List[Dict],
    model_path: str,
    output_dir: str,
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
    from multiclass.modeling import TokenClassificationModelMultiHead

    # Create output tsv path
    output_tsv_path = os.path.join(output_dir, "predictions.tsv")
    os.makedirs(output_dir, exist_ok=True)

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

        # Chunk the document
        chunks = text_splitter.split_text(text)

        chunk_offset = 0
        for chunk_text in chunks:
            if not chunk_text.strip():
                chunk_offset += len(chunk_text)
                continue

            # Tokenize
            encoding = tokenizer(
                chunk_text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                return_tensors="pt",
            )

            offset_mapping = encoding.pop("offset_mapping")[0].tolist()

            # Move to device
            encoding = {k: v.to(device) for k, v in encoding.items()}

            # Get predictions (argmax decoding)
            with torch.no_grad():
                predictions = model(**encoding)

            # Process predictions for each entity type
            for ent_idx, entity_type in enumerate(entity_types):
                preds = predictions[ent_idx][0].cpu().tolist()

                # Extract entities using BIO tags
                current_entity = None
                current_start = None
                current_end = None

                for token_idx, (pred, (start, end)) in enumerate(
                    zip(preds, offset_mapping)
                ):
                    if start == 0 and end == 0:
                        continue

                    tag = id2label.get(pred, "O")

                    if tag.startswith("B"):
                        if current_entity is not None:
                            entity_text = chunk_text[current_start:current_end]
                            results.append(
                                {
                                    "filename": doc_id,
                                    "label": current_entity,
                                    "start_span": chunk_offset + current_start,
                                    "end_span": chunk_offset + current_end,
                                    "text": entity_text,
                                }
                            )
                        current_entity = entity_type
                        current_start = int(start)
                        current_end = int(end)

                    elif tag.startswith("I"):
                        if current_entity == entity_type:
                            current_end = int(end)
                        elif current_entity is not None:
                            entity_text = chunk_text[current_start:current_end]
                            results.append(
                                {
                                    "filename": doc_id,
                                    "label": current_entity,
                                    "start_span": chunk_offset + current_start,
                                    "end_span": chunk_offset + current_end,
                                    "text": entity_text,
                                }
                            )
                            current_entity = entity_type
                            current_start = int(start)
                            current_end = int(end)

                    else:  # O tag
                        if current_entity is not None:
                            entity_text = chunk_text[current_start:current_end]
                            results.append(
                                {
                                    "filename": doc_id,
                                    "label": current_entity,
                                    "start_span": chunk_offset + current_start,
                                    "end_span": chunk_offset + current_end,
                                    "text": entity_text,
                                }
                            )
                            current_entity = None
                            current_start = None
                            current_end = None

                # Don't forget the last entity
                if current_entity is not None:
                    entity_text = chunk_text[current_start:current_end]
                    results.append(
                        {
                            "filename": doc_id,
                            "label": current_entity,
                            "start_span": chunk_offset + current_start,
                            "end_span": chunk_offset + current_end,
                            "text": entity_text,
                        }
                    )

            chunk_offset += len(chunk_text)

    # Create DataFrame and save to TSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_tsv_path, sep="\t", index=False)
        print(f"Predictions saved to {output_tsv_path}")
        print(f"Total entities found: {len(results)}")
    else:
        print("No entities found in the corpus")

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

    annotate_kwargs = {
        "standard": {
            "chunk_size": chunk_size,
            "max_allowed_chunk_size": max_allowed_chunk_size,
            "IOB": use_iob,
        },
        "paragraph": {
            "chunk_size": chunk_size,
            "max_allowed_chunk_size": max_allowed_chunk_size,
            "IOB": use_iob,
        },
        "centered": {"chunk_size": chunk_size, "IOB": use_iob},
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
    unique_tags = None

    annotate_func = annotate_functions[chunk_type]
    kwargs = annotate_kwargs[chunk_type]

    skipped_count = 0
    for k, (batch_id, corpus) in enumerate(datasets.items()):
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

        if batch_id == "train":
            iob_data_train = iob_data
            unique_tags = _unique_tags
        elif batch_id == "test":
            iob_data_test = iob_data
        elif batch_id == "validation":
            iob_data_validation = iob_data

        if (batch_id != "train") & (len(corpus) > 0):
            assert unique_tags == _unique_tags, "Tags are not the same in train/val"
    # paragraph?
    print(f"{skipped_count}/{k} batches skipped ")

    label2id = {l: int(c) for c, l in enumerate(unique_tags)}
    id2label = {int(c): l for c, l in enumerate(unique_tags)}

    print("Unique tags: ", unique_tags)

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
    if chunk_type == "centered":
        annotate_func = annotate_corpus_multihead_centered
        kwargs = {"chunk_size": chunk_size}
    else:
        annotate_func = annotate_corpus_multihead
        kwargs = {
            "chunk_size": chunk_size,
            "max_allowed_chunk_size": max_allowed_chunk_size,
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
                TrainClass.train(
                    train_data=train_data,
                    test_data=test_data,
                    eval_data=tokenized_data_validation,
                    profile=profile,
                )
            else:
                TrainClass.train(
                    train_data=train_data,
                    test_data=[],
                    eval_data=tokenized_data_validation
                    if (tokenized_data_validation is not None)
                    and (len(tokenized_data_validation) > 0)
                    else test_data,
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
                config=model_config, trust_remote_code=True, use_safetensors=True
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
        help="Path to corpus file/directory for inference (uses corpus_validation if not set)",
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
        "--inference_stride",
        type=int,
        default=125,
        help="Stride for inference.",
    )
    argparsers.add_argument(
        "--output_file_prefix", type=str, help="Prefix for the inference results file."
    )

    args = argparsers.parse_args()

    tokenized_data = None
    tags = None

    _model = args.model
    corpus_train = args.corpus_train
    corpus_validation = args.corpus_validation
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
        corpus_inference = args.corpus_inference or corpus_validation or corpus_train
        assert corpus_inference is not None, (
            "Corpus for inference is required (--corpus_inference, --corpus_validation, or --corpus_train)"
        )

        # Load corpus
        if os.path.isdir(corpus_inference):
            corpus_inference_list = merge_annotations(corpus_inference)
        else:
            assert os.path.isfile(corpus_inference), (
                f"Corpus file {corpus_inference} does not exist."
            )
            corpus_inference_list = []
            with open(corpus_inference, "r", encoding="utf-8") as fr:
                for line in fr:
                    corpus_inference_list.append(json.loads(line))

        # If split_file is provided, filter to only test_files
        if split_file is not None:
            assert os.path.isfile(split_file), (
                f"Split file {split_file} does not exist."
            )
            with open(split_file, "r", encoding="utf-8") as fr:
                split_data = json.load(fr)

            test_file_ids = [entry.strip(".txt") for entry in split_data["test_gids"]]

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
            print("Detected MultiHead CRF model, using specialized inference...")
            inference_multihead_crf(
                corpus_data=corpus_inference_list,
                model_path=args.model_path,
                output_dir=args.output_dir,
                lang=lang,
                max_word_per_chunk=None,  # Auto-detect from tokenizer
                trust_remote_code=True,  # Always true for multihead CRF
            )
        elif is_multihead:
            print("Detected MultiHead model (no CRF), using specialized inference...")
            inference_multihead(
                corpus_data=corpus_inference_list,
                model_path=args.model_path,
                output_dir=args.output_dir,
                lang=lang,
                max_word_per_chunk=None,  # Auto-detect from tokenizer
                trust_remote_code=True,  # Always true for multihead
            )
        else:
            # Run standard inference
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
            )

        print("Inference completed!")
        exit(0)

    assert (
        ((corpus_train is not None) and (corpus_validation is not None))
        | ((split_file is not None) and (corpus_train is not None))
        | ((corpus_train is not None) and (force_splitter))
    ), "Either provide a split file or a train and validation corpus"
    assert (_annotation_loc is not None) | (parse_annotations is not None), (
        "Either provide an annotation location or set parse_annotations to True"
    )
    assert (train_model is True) | (parse_annotations is True), (
        "Either parse annotations or train the model, or both..do something!"
    )

    assert corpus_train is not None, "Corpus_train is required"

    if os.path.isdir(corpus_train):
        # go through jsons and merge into one
        corpus_train_list = merge_annotations(corpus_train)
    else:
        assert os.path.isfile(corpus_train), (
            f"Corpus_train file {corpus_train} does not exist."
        )
        # read jsonl
        corpus_train_list = []
        with open(corpus_train, "r", encoding="utf-8") as fr:
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
            with open(corpus_train, "r", encoding="utf-8") as fr:
                for line in fr:
                    corpus_validation_list.append(json.loads(line))
    if split_file is not None:
        assert os.path.isfile(split_file), f"Split_file {split_file} does not exist."

    ################################################
    ################################################

    if (split_file is not None) and (corpus_validation is not None):
        print(
            "Split file and validation corpus provided, ignoring validation corpus file in favor of split file."
        )
        corpus_validation_list = None
        corpus_validation = None

    if force_splitter:
        if corpus_validation is not None:
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
                entry.strip(".txt") for entry in split_data["test_files"]
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
        )

    if train_model:
        print("Training the model..")
        if tokenized_data is None:
            with open(_annotation_loc, "r", encoding="utf-8") as fr:
                tokenized_data = [json.loads(line) for line in fr]

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

            # assert that all labels are within range, >=0 and < num_labels
            # Skip label validation for multi-head models (labels are dicts)
            if not (use_multihead_crf or use_multihead):
                for label in entry["labels"]:
                    if multi_class == False:
                        assert all(
                            [
                                ((_label >= 0) and (_label < num_labels))
                                | (_label == -100)
                                for _label in label
                            ]
                        ), (
                            f"Label {label}, {type(label)} is not within range for entry {entry['id']}"
                        )
                    else:
                        assert label in [-100] + list(range(num_labels)), (
                            f"Label {label} is not within range for entry {entry['id']}"
                        )

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
                )
            else:
                print("No validation corpus available for test TSV output")
