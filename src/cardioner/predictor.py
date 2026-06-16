import argparse
import copy
import os
import re
import unicodedata
import warnings
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

# from dotenv import find_dotenv, load_dotenv
# print(load_dotenv(find_dotenv(".env")))
import regex  # Note: This is NOT the built-in 're' module
import torch
import torch.nn.functional as F
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    TokenClassificationPipeline,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# Import custom model classes for loading legacy models
try:
    from cardioner.multilabel.modeling import MultiLabelTokenClassificationModelCustom
    from cardioner.multilabel.trainer import MultiLabelTokenClassificationModelHF
except ImportError:
    # Fallback for different import contexts
    try:
        from multilabel.modeling import MultiLabelTokenClassificationModelCustom
        from multilabel.trainer import MultiLabelTokenClassificationModelHF
    except ImportError:
        MultiLabelTokenClassificationModelHF = None
        MultiLabelTokenClassificationModelCustom = None
from transformers.pipelines.token_classification import AggregationStrategy

from cardioner.utils import clean_spans


def split_sentence_with_indices(text):
    pattern = r"""
        (?:
            \p{N}+[.,]?\p{N}*\s*[%$€]?           # Numbers with optional decimal/currency
        )
        |
        \p{L}+(?:-\p{L}+)*                      # Words with optional hyphens (letters from any language)
        |
        [()\[\]{}]                              # Parentheses and brackets
        |
        [^\p{L}\p{N}\s]                         # Other single punctuation marks
    """
    return list(regex.finditer(pattern, text, flags=regex.VERBOSE))


def write_annotations_to_file(data, file_path):
    """
    Writes annotation data to a TSV file.

    Parameters:
        data (list of dict): Each dict should have keys:
            'filename', 'ann_id', 'label', 'start_span', 'end_span', 'text'
        file_path (str): Path to the output file
    """
    header = ["filename", "ann_id", "label", "start_span", "end_span", "text"]

    with open(file_path, "w", encoding="utf-8") as f:
        # Write the header
        f.write("\t".join(header) + "\n")
        # Write each row
        for entry in data:
            row = [str(entry[key]) for key in header]
            f.write("\t".join(row) + "\n")


def load_tsv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads a TSV file with specific columns into a pandas DataFrame.

    Expected columns:
        filename, label, start_span, end_span, text, note

    Args:
        file_path (str): Path to the TSV file.

    Returns:
        pd.DataFrame: DataFrame containing the TSV data.
    """
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype={
            "filename": str,
            "label": str,
            "start_span": int,
            "end_span": int,
            "text": str,
            "note": str,
        },
        keep_default_na=False,  # Prevents empty strings being converted to NaN
    )
    return df


class PredictionNER:
    def __init__(
        self,
        model_checkpoint: str,
        revision: Optional[str],
        stride: int | None = 250,
        overlap: int = 0,
        device: Optional[str] = None,
        lang: Literal["es", "nl", "en", "it", "ro", "sv", "cz", "multi"] = "nl",
        trim_trailing_cutoff_words: bool = False,
    ) -> None:

        MAX_TOKENS_IOB_SENT = stride
        OVERLAPPING_LEN = overlap

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            revision=revision,
            is_split_into_words=True,
            truncation=False,
        )
        if MAX_TOKENS_IOB_SENT is None:
            MAX_TOKENS_IOB_SENT = self.tokenizer.model_max_length // 2

        self.model = self._load_model(model_checkpoint, revision)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.lang = lang
        self.trim_trailing_cutoff_words = trim_trailing_cutoff_words
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="o200k_base",
            separators=["\n\n\n", "\n\n", "\n", " .", " !", " ?", " ،", " ,", " ", ""],
            keep_separator=True,
            chunk_size=MAX_TOKENS_IOB_SENT,
            chunk_overlap=OVERLAPPING_LEN,
        )

        ner_labels = list(self.model.config.id2label.values())
        self.base_entity_types = sorted(
            set(label[2:] for label in ner_labels if label != "O")
        )

    def _load_model(self, model_checkpoint: str, revision: Optional[str]):
        """
        Smart model loader that handles both HF standard models and custom model types.
        """
        import json
        import os

        config_path = os.path.join(model_checkpoint, "config.json")

        # Check if it's a custom model by looking at the config
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            architectures = config_dict.get("architectures", [])
            auto_map = config_dict.get("auto_map", {})
            print("Architecture:", architectures)

            # Case 1: MultiLabelTokenClassificationModelHF (standard HF-style, no auto_map)
            if "MultiLabelTokenClassificationModelHF" in architectures and not auto_map:
                if MultiLabelTokenClassificationModelHF is None:
                    raise ImportError(
                        "MultiLabelTokenClassificationModelHF not available. "
                        "Make sure cardioner.multilabel.trainer is importable."
                    )
                print(
                    f"Loading MultiLabelTokenClassificationModelHF from {model_checkpoint}"
                )

                # Suppress misleading "weights not initialized" warning during backbone loading
                # The backbone weights get overwritten by the saved checkpoint anyway
                import logging as py_logging

                from transformers import logging as hf_logging

                prev_verbosity = hf_logging.get_verbosity()
                hf_logging.set_verbosity_error()
                py_logging.getLogger("transformers.modeling_utils").setLevel(
                    py_logging.ERROR
                )

                config = AutoConfig.from_pretrained(model_checkpoint, revision=revision)
                # Set backbone_model_name for proper loading
                if (
                    not hasattr(config, "backbone_model_name")
                    or config.backbone_model_name is None
                ):
                    config.backbone_model_name = config._name_or_path
                model = MultiLabelTokenClassificationModelHF(config)

                # Restore logging verbosity
                hf_logging.set_verbosity(prev_verbosity)
                py_logging.getLogger("transformers.modeling_utils").setLevel(
                    py_logging.WARNING
                )

                # Load state dict
                model_file = os.path.join(model_checkpoint, "model.safetensors")
                if not os.path.exists(model_file):
                    model_file = os.path.join(model_checkpoint, "pytorch_model.bin")
                if os.path.exists(model_file):
                    if model_file.endswith(".safetensors"):
                        from safetensors.torch import load_file

                        state_dict = load_file(model_file)
                    else:
                        import torch

                        state_dict = torch.load(model_file, map_location="cpu")
                    model.load_state_dict(state_dict)
                return model

            # Case 2: Custom model with auto_map (trust_remote_code)
            if auto_map:
                print(
                    f"Loading custom model with trust_remote_code from {model_checkpoint}"
                )
                try:
                    return AutoModelForTokenClassification.from_pretrained(
                        model_checkpoint, revision=revision, trust_remote_code=True
                    )
                except ValueError as e:
                    error_text = str(e)
                    model_ref = auto_map.get("AutoModelForTokenClassification")
                    if isinstance(model_ref, str):
                        model_cls = get_class_from_dynamic_module(
                            model_ref, model_checkpoint
                        )
                        fallback_config = PretrainedConfig.from_dict(config_dict)
                        return model_cls.from_pretrained(
                            model_checkpoint,
                            revision=revision,
                            trust_remote_code=True,
                            config=fallback_config,
                        )
                    if (
                        "model type" not in error_text
                        and "model_type" not in error_text
                        and "config_class" not in error_text
                    ):
                        raise
                    fallback_config = PretrainedConfig.from_dict(config_dict)
                    return AutoModelForTokenClassification.from_pretrained(
                        model_checkpoint,
                        revision=revision,
                        trust_remote_code=True,
                        config=fallback_config,
                    )

        # Default: standard HuggingFace model
        print(f"Loading standard HuggingFace model from {model_checkpoint}")
        return AutoModelForTokenClassification.from_pretrained(
            model_checkpoint, revision=revision, trust_remote_code=True
        )

    def split_text_with_indices(self, text):
        raw_chunks = self.text_splitter.split_text(text)

        # Align each chunk manually by finding its first occurrence in text
        used_indices = set()
        # chunks = []

        for chunk_text in raw_chunks:
            # Find the first unique match position in text to use as a start index
            start_index = text.find(chunk_text)

            # Prevent collisions if chunk_text repeats (naïve fallback)
            while start_index in used_indices:
                start_index = text.find(chunk_text, start_index + 1)
            used_indices.add(start_index)

            end_index = start_index + len(chunk_text)

            yield chunk_text, start_index, end_index

    def predict_text(self, text: str, o_confidence_threshold: float = 0.70):
        # 1. Split text into words and punctuation using regex
        text_matches = split_sentence_with_indices(text)

        # 2. Strip and filter out empty or whitespace-only tokens
        text_words = [m.group().strip() for m in text_matches if m.group().strip()]

        if not text_words:
            return []

        # 3. Tokenize with word alignment
        inputs = self.tokenizer(
            text_words, return_tensors="pt", is_split_into_words=True, truncation=False
        )
        inputs = inputs.to(self.device)
        word_ids = inputs.word_ids()

        # 4. Predict
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)

        predictions = torch.argmax(logits, dim=2)[0]

        # 5. Map predictions back to original stripped words
        results = []
        seen = set()
        non_empty_matches = [m for m in text_matches if m.group().strip()]
        id2label = self.model.config.id2label

        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen:
                continue
            seen.add(word_idx)

            word = text_words[word_idx]
            start = non_empty_matches[word_idx].start()
            end = non_empty_matches[word_idx].end()

            tag_id = predictions[i].item()
            tag = id2label[tag_id]
            score = probs[0, i, tag_id].item()

            # If the tag is "O" and its confidence is low, try to find the next best non-"O" label
            if tag == "O" and score < o_confidence_threshold:
                sorted_probs = torch.argsort(probs[0, i], descending=True)
                for alt_id in sorted_probs:
                    alt_tag = id2label[alt_id.item()]
                    if alt_tag != "O":
                        tag_id = alt_id.item()
                        tag = alt_tag
                        score = probs[0, i, tag_id].item()
                        break  # take the first non-"O" alternative
            _res = {
                "word": word,
                "tag": tag,
                "start": start,
                "end": end,
                "score": score,
            }
            results.append(_res)
        return results

    def aggregate_entities(
        self,
        tagged_tokens,
        original_text,
        confidence_threshold=0.5,
        no_iob=False,
        post_hoc_cleaning=True,
        trim_trailing_cutoff_words_enabled: Optional[bool] = True,
        pre_merge_rule_1: bool = False,
        pre_merge_rule_2: bool = False,
        pre_merge_rule_3: bool = False,
    ):
        """
        Aggregates token-level predictions into entity-level predictions.

        This method processes a list of tagged tokens and combines them into coherent entities
        based on IOB (Inside-Outside-Begin) tagging scheme. It applies correction rules to
        fix common tagging inconsistencies and then aggregates consecutive tokens of the
        same entity type.

        Args:
            tagged_tokens (list): List of dictionaries containing token predictions.
                Each dict should have keys: 'word', 'tag', 'start', 'end', 'score'
            original_text (str): The original text from which tokens were extracted
            confidence_threshold (float, optional): Minimum confidence score required
                for an entity to be included in results. Defaults to 0.3.
            no_iob (bool, optional): If True, disables IOB-specific processing.
                Defaults to False. TODO: NOT IMPLEMENTED YET
            post_hoc_cleaning (bool, optional): If True, applies post-hoc cleaning to the aggregated entities.
                This includes span boundary management, text cleaning (removing unwanted prefixes/suffixes),
                and entity validation. Defaults to False.

        Returns:
            list: List of dictionaries representing aggregated entities.
                Each dict contains: 'start', 'end', 'tag', 'text', 'score'

        Note:
            The method applies three correction rules:
            1. Fixes "O" tags between "B-" and "I-" tags of the same type
            2. Enforces IOB2 compliance by converting any "I-" that does not
               follow a "B-" or "I-" of the same type to "B-"
            3. Splits entities on tokens containing "." by setting that token to "O"
               and restarting continuation with a "B-" tag

            Entities are only included if all constituent tokens meet the confidence threshold
            and the resulting text is not composed entirely of special characters.

            When post_hoc_cleaning is enabled, additional cleaning is applied:
            - Removes trailing space + punctuation (if no opening parenthesis)
            - Removes trailing closing parenthesis (if no opening parenthesis)
            - Removes leading whitespace
            - Validates entities (non-empty, not just "de", not numeric, not special chars only)
        """

        def is_special_char(text):
            return bool(re.fullmatch(r"\W+", text.strip()))

        # TODO: add non-IOB version
        def finalize_entity(entity):
            scores = np.array(entity["scores"])
            median_score = np.median(scores)
            min_score = np.min(scores)
            
            if (median_score >= confidence_threshold) and min_score>0.1:
                entity_text = original_text[entity["start"] : entity["end"]]
                if not is_special_char(entity_text):
                    entity["text"] = entity_text
                    entity["score"] = sum(entity["scores"]) / len(entity["scores"])
                    del entity["scores"]
                    return entity
            return None

        corrected_tokens = copy.deepcopy(tagged_tokens)

        if pre_merge_rule_1:
            # Rule 1: Fix "O" between "B-" and "I-" of same type
            for i in range(1, len(corrected_tokens) - 1):
                prev_tag = corrected_tokens[i - 1]["tag"]
                curr_tag = corrected_tokens[i]["tag"]
                next_tag = corrected_tokens[i + 1]["tag"]

                if (
                    curr_tag == "O"
                    and prev_tag.startswith("B-")
                    and next_tag.startswith("I-")
                ):
                    prev_type = prev_tag[2:]
                    next_type = next_tag[2:]
                    if prev_type == next_type:
                        corrected_tokens[i]["tag"] = "I-" + prev_type

        if pre_merge_rule_2:
            # Rule 2: Enforce IOB2 compliance for I- tags
            prev_tag = "O"
            for i in range(len(corrected_tokens)):
                tag = corrected_tokens[i]["tag"]
                if tag.startswith("I-"):
                    tag_type = tag[2:]
                    if not (
                        (prev_tag.startswith("B-") or prev_tag.startswith("I-"))
                        and prev_tag[2:] == tag_type
                    ):
                        tag = "B-" + tag_type
                        corrected_tokens[i]["tag"] = tag
                    prev_tag = tag
                elif tag.startswith("B-"):
                    prev_tag = tag
                else:
                    prev_tag = "O"

        # Rule 3: Split entities on ".", ":", ";", "/" tokens and restart following continuation with B-
        if pre_merge_rule_3:
            for i in range(len(corrected_tokens)):
                tag = corrected_tokens[i]["tag"]
                if tag.startswith("O"):
                    continue

                token_text = original_text[
                    corrected_tokens[i]["start"] : corrected_tokens[i]["end"]
                ]
                if token_text.strip() in {".", ":", ";", "/"}:
                    corrected_tokens[i]["tag"] = "O"

                    # If the next token was continuing an entity, restart it with B-
                    if i + 1 < len(corrected_tokens) and corrected_tokens[i + 1][
                        "tag"
                    ].startswith("I-"):
                        next_type = corrected_tokens[i + 1]["tag"][2:]
                        corrected_tokens[i + 1]["tag"] = "B-" + next_type

        # Step 2: Aggregate entities
        entities = []
        current_entity = None

        for idx, item in enumerate(corrected_tokens):
            tag = item["tag"]
            start = item["start"]
            end = item["end"]
            score = item["score"]

            if tag.startswith("B-"):
                if current_entity:
                    # Check if current should be merged (same type and touching
                    # or separated by only whitespace)
                    between_text = original_text[current_entity["end"] : start]
                    if current_entity["tag"] == tag[2:] and (
                        current_entity["end"] == start or between_text.isspace()
                    ):
                        # Merge
                        current_entity["end"] = end
                        current_entity["scores"].append(score)
                        continue
                    else:
                        finalized = finalize_entity(current_entity)
                        if finalized:
                            entities.append(finalized)
                current_entity = {
                    "start": start,
                    "end": end,
                    "tag": tag[2:],
                    "scores": [score],
                }

            elif tag.startswith("I-"):
                tag_type = tag[2:]
                if current_entity and current_entity["tag"] == tag_type:
                    current_entity["end"] = end
                    current_entity["scores"].append(score)
                elif (
                    current_entity
                    and idx + 1 < len(corrected_tokens)
                    and corrected_tokens[idx + 1]["tag"] == f"I-{current_entity['tag']}"
                    and tag_type != current_entity["tag"]
                ):
                    # Bridge a single-token class intrusion inside an active entity sequence.
                    # Example: B-X I-X I-Y I-X  -> keep X sequence contiguous.
                    continue
                else:
                    if current_entity:
                        finalized = finalize_entity(current_entity)
                        if finalized:
                            entities.append(finalized)
                    current_entity = {
                        "start": start,
                        "end": end,
                        "tag": tag_type,
                        "scores": [score],
                    }

            else:  # tag == "O"
                if current_entity:
                    finalized = finalize_entity(current_entity)
                    if finalized:
                        entities.append(finalized)
                    current_entity = None

        # Finalize last entity
        if current_entity:
            finalized = finalize_entity(current_entity)
            if finalized:
                entities.append(finalized)

        # Apply post-hoc cleaning if requested
        if post_hoc_cleaning:
            trim_cutoff = (
                self.trim_trailing_cutoff_words
                if trim_trailing_cutoff_words_enabled is None
                else trim_trailing_cutoff_words_enabled
            )
            entities = clean_spans(
                entities,
                original_text,
                lang=self.lang,
                trim_trailing_cutoff_words_enabled=trim_cutoff,
            )

        return entities

    def predict_text_batch(
        self, texts: List[str], o_confidence_threshold: float = 0.70
    ) -> List[List[Dict[str, str]]]:
        batch_text_matches = [split_sentence_with_indices(t) for t in texts]
        batch_text_words = [
            [m.group().strip() for m in matches if m.group().strip()]
            for matches in batch_text_matches
        ]

        valid_indices = [i for i, words in enumerate(batch_text_words) if words]
        if not valid_indices:
            return [[] for _ in texts]

        valid_words = [batch_text_words[i] for i in valid_indices]
        inputs = self.tokenizer(
            valid_words,
            return_tensors="pt",
            is_split_into_words=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=True,
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)

        predictions = torch.argmax(logits, dim=2)
        id2label = self.model.config.id2label

        results_per_valid = []
        for batch_idx, words in enumerate(valid_words):
            word_ids = inputs.word_ids(batch_index=batch_idx)
            text_matches = batch_text_matches[valid_indices[batch_idx]]
            non_empty_matches = [m for m in text_matches if m.group().strip()]
            seen = set()
            results = []

            for i, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx in seen:
                    continue
                seen.add(word_idx)

                word = words[word_idx]
                start = non_empty_matches[word_idx].start()
                end = non_empty_matches[word_idx].end()

                tag_id = predictions[batch_idx, i].item()
                tag = id2label[tag_id]
                score = probs[batch_idx, i, tag_id].item()

                if tag == "O" and score < o_confidence_threshold:
                    sorted_probs = torch.argsort(probs[batch_idx, i], descending=True)
                    for alt_id in sorted_probs:
                        alt_tag = id2label[alt_id.item()]
                        alt_score = probs[batch_idx, i, alt_id.item()].item()
                        if alt_tag != "O" and alt_score >= 0.5:
                            tag_id = alt_id.item()
                            tag = alt_tag
                            score = probs[batch_idx, i, tag_id].item()
                            break

                results.append(
                    {
                        "word": word,
                        "tag": tag,
                        "start": start,
                        "end": end,
                        "score": score,
                    }
                )

            results_per_valid.append(results)

        output = [[] for _ in texts]
        for out_idx, valid_idx in enumerate(valid_indices):
            output[valid_idx] = results_per_valid[out_idx]
        return output

    def do_prediction_batch(
        self,
        text,
        batch_size: int = 8,
        confidence_threshold: float = 0.6,
        post_hoc_cleaning: bool = True,
        o_confidence_threshold: float = 0.7,
        trim_trailing_cutoff_words: Optional[bool] = None,
    ):
        final_prediction = []
        batch_texts = []
        batch_offsets = []

        for sub_text, sub_text_start, sub_text_end in self.split_text_with_indices(
            text
        ):
            batch_texts.append(sub_text)
            batch_offsets.append(sub_text_start)

            if len(batch_texts) >= batch_size:
                batch_tokens = self.predict_text_batch(
                    batch_texts, o_confidence_threshold=o_confidence_threshold
                )
                for tokens, sub_text, sub_text_start in zip(
                    batch_tokens, batch_texts, batch_offsets
                ):
                    predictions = self.aggregate_entities(
                        tokens,
                        sub_text,
                        confidence_threshold=confidence_threshold,
                        post_hoc_cleaning=post_hoc_cleaning,
                        trim_trailing_cutoff_words_enabled=trim_trailing_cutoff_words,
                    )
                    for pred in predictions:
                        pred["start"] += sub_text_start
                        pred["end"] += sub_text_start
                        pred["entity"] = pred["tag"]
                        final_prediction.append(pred)

                batch_texts = []
                batch_offsets = []

        if batch_texts:
            batch_tokens = self.predict_text_batch(
                batch_texts, o_confidence_threshold=o_confidence_threshold
            )
            for tokens, sub_text, sub_text_start in zip(
                batch_tokens, batch_texts, batch_offsets
            ):
                predictions = self.aggregate_entities(
                    tokens,
                    sub_text,
                    confidence_threshold=confidence_threshold,
                    post_hoc_cleaning=post_hoc_cleaning,
                    trim_trailing_cutoff_words_enabled=trim_trailing_cutoff_words,
                )
                for pred in predictions:
                    pred["start"] += sub_text_start
                    pred["end"] += sub_text_start
                    pred["entity"] = pred["tag"]
                    final_prediction.append(pred)

        return final_prediction

    def do_prediction(
        self,
        text,
        confidence_threshold=0.6,
        post_hoc_cleaning=True,
        trim_trailing_cutoff_words: Optional[bool] = None,
    ):
        final_prediction = []
        # final_prediction_2 = []
        for sub_text, sub_text_start, sub_text_end in self.split_text_with_indices(
            text
        ):
            tokens = self.predict_text(text=sub_text)
            predictions = self.aggregate_entities(
                tokens,
                sub_text,
                confidence_threshold=confidence_threshold,
                post_hoc_cleaning=post_hoc_cleaning,
                trim_trailing_cutoff_words_enabled=trim_trailing_cutoff_words,
            )

            for pred in predictions:
                pred["start"] += sub_text_start
                pred["end"] += sub_text_start
                pred["entity"] = pred["tag"]
                final_prediction.append(pred)
        return final_prediction


def evaluate(
    model_checkpoint,
    revision,
    root_path,
    lang,
    cat,
    device: Optional[str] = None,
    batch_size: int = 8,
):
    ner = PredictionNER(
        model_checkpoint=model_checkpoint,
        revision=revision,
        device=device,
        lang=lang,
    )
    # conver the predictions to ann format
    test_files_root = os.path.join(root_path, "txt")
    tsv_file_path_test = os.path.join(root_path, f"test_cardioccc_{lang}_{cat}.tsv")
    test_df = load_tsv_to_dataframe(tsv_file_path_test)
    prd_ann = []

    for fn in tqdm(test_df["filename"].unique()):
        # fn = "casos_clinicos_cardiologia508"
        with open(
            os.path.join(test_files_root, fn + ".txt"), "r", encoding="utf-8"
        ) as f:
            document_text = f.read()
            prds = ner.do_prediction_batch(
                document_text,
                batch_size=batch_size,
                confidence_threshold=0.35,
            )
            for prd in prds:
                prd_ann.append(
                    {
                        "filename": fn,
                        "label": prd["tag"],
                        "ann_id": "NA",
                        "start_span": prd["start"],
                        "end_span": prd["end"],
                        "text": prd["text"].replace("\n", " ").replace("\t", " "),
                    }
                )
        # break

    output_tsv_path = os.path.join(
        root_path, f"pre_{model_checkpoint.split('/')[1]}_{revision}.tsv"
    )
    write_annotations_to_file(prd_ann, output_tsv_path)
    print(f"output_tsv_path {output_tsv_path}")


# Add Manuela predictor
#


class PrefixAwareTokenClassificationPipeline(TokenClassificationPipeline):
    """
    A TokenClassificationPipeline that:
      • Checks for tokenizer._tokenizer.model.continuing_subword_prefix, and if present
        uses `is_subword = not word.startswith(prefix)` instead of `len(word) != len(word_ref)`.
      • Falls back to the original “space-based” heuristic only if no prefix is set.
    """

    def __init__(self, prefix: str = "##", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = prefix

    @staticmethod
    def fix_misdecoded_string(
        s,
        incorrect_encoding: Literal["latin-1", "cp1252"] = "cp1252",
        correct_encoding="utf-8",
    ):
        """Attempts to fix a string that was likely misdecoded."""
        try:
            # Encode the string using the assumed incorrect encoding
            byte_representation = s.encode(incorrect_encoding)
            # Decode the bytes using the correct encoding
            corrected_s = byte_representation.decode(correct_encoding)
            return corrected_s
        except (UnicodeEncodeError, UnicodeDecodeError):
            # If encoding/decoding fails, return the original string
            return s

    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: np.ndarray,
        scores: np.ndarray,
        offset_mapping: Optional[List[Tuple[int, int]]],
        special_tokens_mask: np.ndarray,
        aggregation_strategy: AggregationStrategy,
    ) -> List[Dict]:
        """
        Copy of the original gather_pre_entities, except for the one block:
            is_subword = len(word) != len(word_ref)
        is replaced by
            is_subword = not word.startswith(prefix)
        whenever prefix := tokenizer._tokenizer.model.continuing_subword_prefix is not None.
        """
        pre_entities = []

        # Extract the prefix (if any) from the BPE model
        prefix = None
        if getattr(self.tokenizer, "_tokenizer", None) is not None and getattr(
            self.tokenizer._tokenizer.model, "continuing_subword_prefix", None
        ):
            prefix = self.tokenizer._tokenizer.model.continuing_subword_prefix

        for idx, token_scores in enumerate(scores):
            # 1) Skip special- tokens right away
            if special_tokens_mask[idx]:
                continue

            # 2) Get the text form of this subtoken
            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))

            # MAYBE REMOVE?
            if word == prefix:
                continue

            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                # Convert from torch tensors if needed
                if not isinstance(start_ind, int):
                    if self.framework == "pt":
                        start_ind = start_ind.item()
                        end_ind = end_ind.item()

                # The raw slice of the original sentence that this subtoken covers:
                word_ref = sentence[start_ind:end_ind]

                # print(f'|{word}|,|{word_ref}|')

                # If the tokenizer has a continuing_subword_prefix, use prefix‐based logic:
                if prefix is not None and prefix != "":
                    # No warning here, because we trust prefix for subword detection
                    is_subword = not word.startswith(prefix)
                else:
                    word_fixed = self.fix_misdecoded_string(word)
                    if word.startswith(self.prefix) and word_ref != "":
                        is_subword = False
                    elif len(word_fixed) != len(word_ref) and word_ref != "":
                        # Fallback heuristic exactly as in the original pipeline:
                        if aggregation_strategy in {
                            AggregationStrategy.FIRST,
                            AggregationStrategy.AVERAGE,
                            AggregationStrategy.MAX,
                        }:
                            warnings.warn(
                                f"\nTokenizer does not support real words or there is an encoding problem\n:\n\t word from tokenizer {word},\n\t word from text {word_ref}\n Using fallback heuristic",
                                UserWarning,
                            )
                        is_subword = True  # (start_ind > 0 and " " not in sentence[start_ind - 1 : start_ind + 1])
                    else:
                        is_subword = True
                # If this token is actually an <unk> token, force it to be non‐subword
                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                start_ind = None
                end_ind = None
                is_subword = False

            # Debug print for each token
            # max_score_idx = np.argmax(token_scores)
            # max_score = token_scores[max_score_idx]
            # predicted_label = self.model.config.id2label[max_score_idx]

            # print(f"DEBUG Token {idx}: word='{word}',\
            # word_fixed='{self.fix_misdecoded_string(word)}',\
            # word_ref='{word_ref}', is_subword={is_subword},\
            # predicted_label='{predicted_label}', max_score={max_score:.4f}")

            pre_entities.append(
                {
                    "word": word,
                    "scores": token_scores,
                    "start": start_ind,
                    "end": end_ind,
                    "index": idx,
                    "is_subword": is_subword,
                }
            )
        # print(f"DEBUG: Total pre_entities: {len(pre_entities)}")
        return pre_entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model with specified configuration."
    )

    parser.add_argument(
        "--model_checkpoint", "-m", type=str, help="Model checkpoint to use."
    )
    parser.add_argument(
        "--revision", "-r", type=str, default="main", help="Model revision or version."
    )
    parser.add_argument(
        "--root", "-p", type=str, help="Path to the dataset root directory."
    )
    parser.add_argument(
        "--lang", "-l", type=str, help="Language code (e.g., 'es', 'en', 'multi')."
    )
    parser.add_argument(
        "--cat", "-c", type=str, help="Category (e.g., 'med' for medication)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda', 'cpu'). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for batched chunk inference.",
    )

    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint
    revision = args.revision
    root = args.root
    lang = args.lang
    cat = args.cat

    lang = lang.upper()
    cat = cat.upper()

    evaluate(
        model_checkpoint,
        revision,
        root,
        lang,
        cat,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Now use these variables below as needed
    print(f"Using model: {model_checkpoint} (revision: {revision})")
    print(f"Dataset root: {root} | Language: {lang} | Category: {cat}")
