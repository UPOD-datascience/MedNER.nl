"""
Incoming format is :
[
{"tags": [{"start": xx, "end":xx, "tag": "DISEASE"},...],
 "id": xxx,
 "text": xxx}

]
"""

from ast import parse
from os import environ, truncate

import spacy

# Load a spaCy model for tokenization
environ["WANDB_MODE"] = "disabled"
environ["WANDB_DISABLED"] = "true"

import argparse
import json
import re
from collections import defaultdict
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from datasets import Dataset, DatasetDict
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


def flatten_token_list(l: list) -> List[List[str]]:
    """
    Recursively flatten a nested list of tokens into a list of token sequences.
    Stops when it reaches a list of strings (a single split).
    """
    if isinstance(l, list):
        if l and isinstance(l[0], str):
            return [l]
    return [item for sublist in l for item in flatten_token_list(sublist)]


def split_tokens_hierarchical(
    tokens: List[str],
    max_chunk_size: int,
) -> List[List[str]]:
    """
    Hierarchically split tokens into smaller units based on:
    1. Paragraphs (double newlines)
    2. Lines (single newlines)
    3. Sentences (., !, ?)
    4. Individual words (as last resort)

    This mirrors the logic in light_ner_word_level.py's split_tokens function.
    """
    # Split by paragraphs (double newlines)
    paragraph_breaks = (
        [0]
        + [i + 1 for i in range(len(tokens)) if re.match(r".*\n\n+.*", tokens[i])]
        + [len(tokens)]
    )
    paragraphs = [
        tokens[paragraph_breaks[i] : paragraph_breaks[i + 1]]
        for i in range(len(paragraph_breaks) - 1)
    ]

    for p_idx, para_tokens in enumerate(paragraphs):
        if len(para_tokens) > max_chunk_size:
            # Split by lines (single newlines, excluding double newlines)
            line_breaks = (
                [0]
                + [
                    i + 1
                    for i in range(len(para_tokens))
                    if re.sub(r"\n\n+", "", para_tokens[i]).count("\n") == 1
                ]
                + [len(para_tokens)]
            )
            lines = [
                para_tokens[line_breaks[i] : line_breaks[i + 1]]
                for i in range(len(line_breaks) - 1)
            ]

            for l_idx, line_tokens in enumerate(lines):
                if len(line_tokens) > max_chunk_size:
                    # Split by sentences (., !, ?)
                    sentence_breaks = (
                        [0]
                        + [
                            i + 1
                            for i in range(len(line_tokens))
                            if re.match(r"[\.!\?]", line_tokens[i])
                        ]
                        + [len(line_tokens)]
                    )
                    sentences = [
                        line_tokens[sentence_breaks[i] : sentence_breaks[i + 1]]
                        for i in range(len(sentence_breaks) - 1)
                    ]

                    for s_idx, sentence_tokens in enumerate(sentences):
                        if len(sentence_tokens) > max_chunk_size:
                            # Split by individual words as last resort
                            words = [[token] for token in sentence_tokens]
                            sentences[s_idx] = words

                    lines[l_idx] = sentences

            paragraphs[p_idx] = lines

    splits = flatten_token_list(paragraphs)
    return splits


def merge_splits_into_chunks_multilabel(
    tokens: List[str],
    token_tags: List[List[str]],
    splits: List[List[str]],
    max_chunk_size: int,
) -> List[Tuple[List[str], List[List[str]]]]:
    """
    Merge the hierarchical splits back into chunks that fit within max_chunk_size,
    while keeping token_tags aligned.

    Returns a list of (chunk_tokens, chunk_tags) tuples.
    """
    chunks = []
    current_chunk_tokens = []
    current_chunk_tags = []
    current_token_idx = 0

    for split in splits:
        split_len = len(split)

        # Check if adding this split would exceed the limit
        if (
            len(current_chunk_tokens) + split_len > max_chunk_size
            and current_chunk_tokens
        ):
            # Save current chunk and start a new one
            chunks.append((current_chunk_tokens, current_chunk_tags))
            current_chunk_tokens = []
            current_chunk_tags = []

        # Add the split to current chunk
        current_chunk_tokens.extend(split)
        current_chunk_tags.extend(
            token_tags[current_token_idx : current_token_idx + split_len]
        )
        current_token_idx += split_len

    # Don't forget the last chunk
    if current_chunk_tokens:
        chunks.append((current_chunk_tokens, current_chunk_tags))

    return chunks


# def annotate_corpus_paragraph, use split_text
def annotate_corpus_paragraph(
    corpus,
    batch_id: str = "b1",
    lang: str = "nl",
    chunk_size: int = 256,
    max_allowed_chunk_size: int = 300,
    paragraph_boundary: str = "\n\n",
    min_token_len: int = 8,
    IOB: bool = True,
):
    """
    Annotate corpus using hierarchical chunking strategy:
    1. First try to split by paragraphs (double newlines)
    2. If a paragraph is too long, split by lines (single newlines)
    3. If a line is still too long, split by sentences (., !, ?)
    4. If a sentence is still too long, split by individual words
    5. Then greedily merge splits back into chunks up to max_allowed_chunk_size

    This approach mirrors the logic in light_ner_word_level.py.
    """
    annotated_data = []
    unique_tags = set()

    nlp = spacy.blank(lang)

    for entry in tqdm(corpus):
        text = entry["text"]
        tags = entry["tags"]

        # Tokenize the text using spaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]
        token_offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

        # Initialize tags for each token with empty list
        token_tags = [[] for _ in range(len(doc))]

        # Annotate each token with labels
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            unique_tags.add(tag_type)

            # Find tokens that fall within the span
            is_first_token = True
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue  # Token is before the entity
                if token_start >= end:
                    break  # Token is after the entity
                if (
                    (token_start >= start and token_end <= end)
                    or (token_start < start and token_end > start)
                    or (token_start < end and token_end > end)
                ):
                    # Token overlaps with entity boundary
                    if is_first_token and IOB:
                        tag_label = f"B-{tag_type}"
                        is_first_token = False
                    elif IOB:
                        tag_label = f"I-{tag_type}"
                    else:
                        tag_label = tag_type
                    token_tags[i].append(tag_label)

        # Skip empty documents
        if not tokens:
            continue

        # Respect requested chunk_size while staying within model constraints.
        effective_chunk_size = min(chunk_size, max_allowed_chunk_size)

        # Hierarchically split tokens into manageable units
        splits = split_tokens_hierarchical(tokens, effective_chunk_size)

        # Merge splits back into chunks that fit within effective_chunk_size
        chunks = merge_splits_into_chunks_multilabel(
            tokens, token_tags, splits, effective_chunk_size
        )

        # Create annotated data entries for each chunk
        for chunk_idx, (chunk_tokens, chunk_tags) in enumerate(chunks):
            # Skip chunks that are too small
            if len(chunk_tokens) < min_token_len:
                continue

            annotated_data.append(
                {
                    "gid": entry["id"],
                    "id": entry["id"] + f"_{chunk_idx}",
                    "batch": batch_id,
                    "tokens": chunk_tokens,
                    "tags": chunk_tags,
                }
            )

    if IOB:
        tag_list = ["O"] + [f"B-{tag},I-{tag}" for tag in unique_tags]
        tag_list = [tag for sublist in tag_list for tag in sublist.split(",")]
    else:
        tag_list = ["O"] + [tag for tag in unique_tags]

    return annotated_data, tag_list


# def annotate_corpus_sentence, use pysbd


def annotate_corpus_standard(
    corpus,
    batch_id: str = "b1",
    lang: str = "nl",
    chunk_size: int = 256,
    max_allowed_chunk_size: int = 450,
    IOB: bool = True,
):
    annotated_data = []
    unique_tags = set()

    nlp = spacy.blank(lang)

    for entry in tqdm(corpus):
        text = entry["text"]
        tags = entry["tags"]

        # Tokenize the text using spaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]
        token_offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

        # Initialize tags for each token with empty list
        token_tags = [[] for _ in range(len(doc))]

        # Annotate each token with labels
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            unique_tags.add(tag_type)

            # Find tokens that fall within the span
            is_first_token = True
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue  # Token is before the entity
                if token_start >= end:
                    break  # Token is after the entity
                if (
                    (token_start >= start and token_end <= end)
                    or (token_start < start and token_end > start)
                    or (token_start < end and token_end > end)
                ):
                    # Token overlaps with entity boundary
                    if is_first_token and IOB:
                        tag_label = f"B-{tag_type}"
                        is_first_token = False
                    elif IOB:
                        tag_label = f"I-{tag_type}"
                    else:
                        tag_label = tag_type
                    token_tags[i].append(tag_label)

        # Split tokens and tags into chunks of max_tokens without splitting entities
        i = 0
        while i < len(tokens):
            end_index = min(i + chunk_size, len(tokens))
            # Adjust end_index to avoid splitting entities
            while (
                end_index < len(tokens)
                and any(label.startswith("I-") for label in token_tags[end_index])
                and (end_index - i) < max_allowed_chunk_size
            ):
                end_index += 1

            # Ensure the chunk does not exceed max_allowed_chunk_size
            if (end_index - i) > max_allowed_chunk_size:
                end_index = i + max_allowed_chunk_size
            chunk_tokens = tokens[i:end_index]
            chunk_tags = token_tags[i:end_index]

            annotated_data.append(
                {
                    "gid": entry["id"],
                    "id": entry["id"]
                    + f"_{i // chunk_size}",  # Modify ID to reflect chunk
                    "batch": batch_id,
                    "tokens": chunk_tokens,
                    "tags": chunk_tags,
                }
            )

            i = end_index

    if IOB:
        tag_list = ["O"] + [f"B-{tag},I-{tag}" for tag in unique_tags]
        tag_list = [tag for sublist in tag_list for tag in sublist.split(",")]
    else:
        tag_list = ["O"] + [tag for tag in unique_tags]

    return annotated_data, tag_list


def annotate_corpus_centered(
    corpus,
    batch_id: str = "b1",
    lang: str = "nl",
    chunk_size: int = 512,
    IOB: bool = True,
):
    annotated_data = []
    unique_tags = set()

    nlp = spacy.blank(lang)

    for entry in tqdm(corpus):
        text = entry["text"]
        tags = entry["tags"]

        # Tokenize the text using spaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]
        token_offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

        # Initialize tags for each token as an empty list to allow multiple labels
        token_tags = [[] for _ in range(len(doc))]

        # Annotate each token with IOB tags per label
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            unique_tags.add(tag_type)

            # Find tokens that fall within the span
            is_first_token = True
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue  # Token is before the entity
                if token_start >= end:
                    break  # Token is after the entity
                if (
                    (token_start >= start and token_end <= end)
                    or (token_start < start and token_end > start)
                    or (token_start < end and token_end > end)
                ):
                    # Token overlaps with entity boundary

                    if is_first_token and IOB:
                        tag_label = f"B-{tag_type}"
                        is_first_token = False
                    elif IOB:
                        tag_label = f"I-{tag_type}"
                    else:
                        tag_label = tag_type

                    token_tags[i].append(tag_label)

        # Create chunks centered around each span
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]

            # Find the token indices for the span
            span_start_idx = None
            span_end_idx = None
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end > start and span_start_idx is None:
                    span_start_idx = i
                if token_start >= end:
                    span_end_idx = i
                    break

            if span_start_idx is None:
                print(
                    f"Warning: Could not find token indices for span in document ID {entry['id']}"
                )
                print(f"Span start: {start}, Span end: {end}")
                continue

            if span_end_idx is None:
                span_end_idx = len(tokens)  # Span goes to the end of the text

            # Adjust left_context and right_context to avoid splitting entities
            left_context = span_start_idx - (chunk_size // 2)
            right_context = span_end_idx + (chunk_size // 2)

            # Ensure contexts are within bounds
            left_context = max(0, left_context)
            right_context = min(len(tokens), right_context)

            # Adjust left_context to avoid starting in the middle of an entity
            while (
                left_context > 0
                and any(label.startswith("I-") for label in token_tags[left_context])
                and (right_context - left_context) < chunk_size * 2
            ):
                left_context -= 1

            # Adjust right_context to avoid ending in the middle of an entity
            while (
                right_context < len(tokens)
                and any(
                    label.startswith("I-") for label in token_tags[right_context - 1]
                )
                and (right_context - left_context) < chunk_size * 2
            ):
                right_context += 1

            # Limit the chunk size to avoid excessively long sequences
            if (right_context - left_context) > chunk_size * 2:
                right_context = left_context + chunk_size * 2

            chunk_tokens = tokens[left_context:right_context]
            chunk_tags = token_tags[left_context:right_context]

            annotated_data.append(
                {
                    "gid": entry["id"],
                    "id": entry["id"] + f"_span_{start}_{end}",
                    "batch": batch_id,
                    "tokens": chunk_tokens,
                    "tags": chunk_tags,
                }
            )

    if IOB:
        tag_list = ["O"] + [f"B-{tag},I-{tag}" for tag in unique_tags]
        tag_list = [tag for sublist in tag_list for tag in sublist.split(",")]
    else:
        tag_list = ["O"] + [tag for tag in unique_tags]
    return annotated_data, tag_list


def count_tokens_with_multiple_labels(annotated_data):
    total_tokens = 0
    total_labeled_tokens = 0
    tokens_with_multiple_labels = 0

    for entry in annotated_data:
        token_tags = entry["tags"]
        for token_labels in token_tags:
            total_tokens += 1
            if len(token_labels) > 1:
                tokens_with_multiple_labels += 1
                total_labeled_tokens += 1
            elif (len(token_labels) == 1) and (token_labels[0] != "O"):
                total_labeled_tokens += 1

    print(f"Total tokens: {total_tokens}")
    print(f"Total labeled tokens: {total_labeled_tokens}")
    print(f"Tokens with multiple labels: {tokens_with_multiple_labels}")

    if total_tokens > 0:
        percentage_multilabeled = (tokens_with_multiple_labels / total_tokens) * 100
        percentage_lab_multi = (
            tokens_with_multiple_labels / total_labeled_tokens
        ) * 100
        print(
            f"Percentage of tokens with multiple labels: {percentage_multilabeled:.2f}%"
        )
        print(
            f"Percentage of labeled tokens with multiple labels: {percentage_lab_multi:.2f}%"
        )
    else:
        print("No tokens found.")


def align_labels_with_tokens(labels, word_ids, num_labels, id2label):
    new_labels = []
    current_word = None

    label2id = {label: idx for idx, label in id2label.items()}
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            # Special token
            new_labels.append([-100] * num_labels)
        elif word_id != current_word:
            # Start of a new word
            current_word = word_id
            if word_id < len(labels):
                label_ids = labels[word_id]
                label_vector = [
                    1 if idx in label_ids else 0 for idx in range(num_labels)
                ]
            else:
                label_vector = [0] * num_labels
            new_labels.append(label_vector)
        else:
            # Same word as previous token, convert B- to I-
            if word_id < len(labels):
                label_ids = labels[word_id]
                label_names = [id2label[idx] for idx in label_ids]
                # Convert B- labels to I- labels
                converted_label_ids = [
                    label2id[label.replace("B-", "I-")]
                    if label.startswith("B-")
                    else label2id[label]
                    for label in label_names
                ]
                label_vector = [
                    1 if idx in converted_label_ids else 0 for idx in range(num_labels)
                ]
            else:
                label_vector = [0] * num_labels
            new_labels.append(label_vector)
    return new_labels


def tokenize_and_align_labels(docs, tokenizer, label2id, max_length=None):
    # padding is handled during the collation
    #
    tokenized_inputs = tokenizer(
        docs["tokens"],
        is_split_into_words=True,
        max_length=tokenizer.model_max_length if max_length is None else max_length,
        padding=False,
        truncation=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    id2label = {idx: label for label, idx in label2id.items()}

    all_labels = []
    for token_tags in docs["tags"]:
        # Convert tag names to IDs, defaulting to 'O' if empty
        labels = [
            [label2id[tag] for tag in tags] if tags else [label2id["O"]]
            for tags in token_tags
        ]
        all_labels.append(labels)

    new_labels = []
    all_word_ids = []  # Store word_ids for word-level training
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        all_word_ids.append(word_ids)  # Collect word_ids for each example
        num_labels = len(label2id)
        aligned_labels = align_labels_with_tokens(
            labels, word_ids, num_labels, id2label
        )
        new_labels.append(aligned_labels)

    tokenized_inputs["labels"] = new_labels
    tokenized_inputs["word_ids"] = (
        all_word_ids  # Add word_ids to output for word-level training
    )
    return tokenized_inputs
