"""
Incoming format is :
[
{"tags": [{"start": xx, "end":xx, "tag": "DISEASE"},...],
 "id": xxx,
 "text": xxx}

]

For Multi-Head CRF, each entity type gets its own BIO label sequence.
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


def merge_splits_into_chunks_multiclass(
    tokens: List[str],
    token_tags: List[str],
    splits: List[List[str]],
    max_chunk_size: int,
) -> List[Tuple[List[str], List[str]]]:
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

        # Initialize tags for each token with "O" (outside)
        token_tags = ["O"] * len(doc)

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
                        token_tags[i] = f"B-{tag_type}"
                        is_first_token = False
                    elif IOB:
                        token_tags[i] = f"I-{tag_type}"
                    else:
                        token_tags[i] = tag_type

        # Skip empty documents
        if not tokens:
            continue

        # Respect requested chunk_size while staying within model constraints.
        effective_chunk_size = min(chunk_size, max_allowed_chunk_size)

        # Hierarchically split tokens into manageable units
        splits = split_tokens_hierarchical(tokens, effective_chunk_size)

        # Merge splits back into chunks that fit within effective_chunk_size
        chunks = merge_splits_into_chunks_multiclass(
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
        # tag_list = ['O'] + [f'B-{tag},I-{tag}' for tag in sorted(unique_tags)]
        # tag_list = [tag for sublist in tag_list for tag in sublist.split(',')]
        tag_list = ["O"] + [
            x for tag in sorted(unique_tags) for x in (f"B-{tag}", f"I-{tag}")
        ]
    else:
        tag_list = ["O"] + [tag for tag in unique_tags]

    return annotated_data, tag_list


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

    for entry in corpus:
        text = entry["text"]
        tags = entry["tags"]

        # Tokenize the text using spaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]
        token_offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

        # Initialize tags for each token with "O" (outside)
        token_tags = ["O"] * len(doc)

        # Annotate each token with IOB tags
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            unique_tags.add(tag_type)

            # Find tokens that fall within the span
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue  # Token is before the entity
                if token_start >= end:
                    break  # Token is after the entity
                if token_start >= start and token_end <= end:
                    # Token is inside the entity
                    if token_tags[i] == "O":
                        if IOB:
                            token_tags[i] = f"B-{tag_type}"
                        else:
                            token_tags[i] = tag_type
                    else:
                        if IOB:
                            token_tags[i] = f"I-{tag_type}"
                        else:
                            token_tags[i] = tag_type

                elif (token_start < start and token_end > start) or (
                    token_start < end and token_end > end
                ):
                    # Token overlaps with entity boundary
                    if IOB:
                        token_tags[i] = f"I-{tag_type}"
                    else:
                        token_tags[i] = tag_type

        # Split tokens and tags into chunks of max_tokens without splitting entities
        i = 0
        while i < len(tokens):
            end_index = min(i + chunk_size, len(tokens))
            # Adjust end_index to avoid splitting entities
            while (
                end_index < len(tokens)
                and token_tags[end_index].startswith("I-")
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
    lang: str = "nl",
    batch_id: str = "b1",
    chunk_size: int = 512,
    IOB: bool = True,
):
    annotated_data = []
    unique_tags = set()

    nlp = spacy.blank(lang)

    for entry in corpus:
        text = entry["text"]
        tags = entry["tags"]

        # Tokenize the text using spaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]
        token_offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

        # Initialize tags for each token with "O" (outside)
        token_tags = ["O"] * len(doc)

        # Annotate each token with IOB tags
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            unique_tags.add(tag_type)

            # Find tokens that fall within the span
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue  # Token is before the entity
                if token_start >= end:
                    break  # Token is after the entity
                if token_start >= start and token_end <= end:
                    # Token is inside the entity
                    if token_tags[i] == "O":
                        if IOB:
                            token_tags[i] = f"B-{tag_type}"
                        else:
                            token_tags[i] = tag_type
                    else:
                        if IOB:
                            token_tags[i] = f"I-{tag_type}"
                        else:
                            token_tags[i] = tag_type
                elif (token_start < start and token_end > start) or (
                    token_start < end and token_end > end
                ):
                    # Token overlaps with entity boundary
                    if IOB:
                        token_tags[i] = f"I-{tag_type}"
                    else:
                        token_tags[i] = tag_type

        # Create chunks centered around each span
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]

            # Find the token indices for the span
            span_start_idx = None
            span_end_idx = None
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end >= start and span_start_idx is None:
                    span_start_idx = i
                if token_start >= end:
                    span_end_idx = i
                    break

            if span_start_idx is None:
                print(
                    f"Warning: Could not find token indices for span in document ID {entry['id']}"
                )
                print(f"Span start: {start}, Span end: {end}")
                # print(f"Token offsets: {token_offsets}")
                continue

            if span_end_idx is None:
                span_end_idx = len(tokens)  # Span goes to the end of the text

            # Center the chunk around the span
            left_context = max(0, span_start_idx - (chunk_size // 2))
            right_context = min(len(tokens), span_start_idx + (chunk_size // 2))

            # Adjust if the span is near the start or end of the document
            if right_context - left_context < chunk_size:
                if left_context == 0:
                    right_context = min(len(tokens), left_context + chunk_size)
                elif right_context == len(tokens):
                    left_context = max(0, right_context - chunk_size)

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


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    max_label_idx = len(labels) - 1  # Maximum valid index for labels
    for word_id in word_ids:
        if word_id is None:
            # Special token
            new_labels.append(-100)
        elif word_id != current_word:
            # Start of a new word!
            current_word = word_id
            # Ensure word_id does not exceed labels length
            if word_id > max_label_idx:
                label = -100  # Assign -100 if word_id is beyond labels
            else:
                label = labels[word_id]
            new_labels.append(label)
        else:
            # Same word as previous token
            if word_id > max_label_idx:
                label = -100
            else:
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(
    docs,
    tokenizer,
    label2id: Optional[Dict[str, int]] = None,
    max_length: Optional[int] = None,
):
    """
    Tokenizes and aligns labels with tokens.
    """
    tokenized_inputs = tokenizer(
        docs["tokens"],
        is_split_into_words=True,
        max_length=tokenizer.model_max_length if max_length is None else max_length,
        padding="max_length",  # Pad sequences to max_length
        truncation=True,  # Truncate sequences longer than max_length
        return_offsets_mapping=True,
    )
    if label2id is not None:
        # Corrected this line to handle lists of lists
        all_labels = [[label2id[tag] for tag in tags] for tags in docs["tags"]]
    else:
        all_labels = docs["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# =============================================================================
# Multi-Head CRF Support Functions
# =============================================================================


def annotate_corpus_multihead(
    corpus,
    entity_types: List[str],
    batch_id: str = "b1",
    lang: str = "nl",
    chunk_size: int = 256,
    max_allowed_chunk_size: int = 450,
):
    """
    Annotate corpus for Multi-Head CRF with separate BIO labels per entity type.

    Each entity type gets its own sequence of BIO tags (O=0, B=1, I=2).
    This allows for overlapping entities of different types.

    Args:
        corpus: List of documents with tags
        entity_types: List of entity type names to extract (e.g., ["DRUG", "DISEASE"])
        batch_id: Batch identifier
        lang: Language for spaCy tokenizer
        chunk_size: Target chunk size in tokens
        max_allowed_chunk_size: Maximum chunk size

    Returns:
        annotated_data: List of annotated chunks with multi-head labels
        entity_types: List of entity types (for consistency)
    """
    annotated_data = []
    nlp = spacy.blank(lang)

    # BIO label mapping: O=0, B=1, I=2
    bio_label2id = {"O": 0, "B": 1, "I": 2}

    for entry in tqdm(corpus, desc="Annotating for Multi-Head CRF"):
        text = entry["text"]
        tags = entry["tags"]

        # Tokenize the text using spaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]
        token_offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

        # Initialize tags for each entity type with "O" (outside)
        # Structure: {entity_type: ["O", "O", ...]}
        entity_token_tags = {ent: ["O"] * len(tokens) for ent in entity_types}

        # Annotate each token with BIO tags for each entity type
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]

            # Skip if this entity type is not in our list
            if tag_type not in entity_types:
                continue

            # Find tokens that fall within the span
            is_first_token = True
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue  # Token is before the entity
                if token_start >= end:
                    break  # Token is after the entity

                # Token overlaps with entity
                if (
                    (token_start >= start and token_end <= end)
                    or (token_start < start and token_end > start)
                    or (token_start < end and token_end > end)
                ):
                    if is_first_token:
                        entity_token_tags[tag_type][i] = "B"
                        is_first_token = False
                    else:
                        entity_token_tags[tag_type][i] = "I"

        # Split tokens and tags into chunks
        i = 0
        while i < len(tokens):
            end_index = min(i + chunk_size, len(tokens))

            # Adjust end_index to avoid splitting entities (check all entity types)
            while end_index < len(tokens) and (end_index - i) < max_allowed_chunk_size:
                # Check if any entity type has an I tag at end_index
                any_inside = any(
                    entity_token_tags[ent][end_index] == "I" for ent in entity_types
                )
                if not any_inside:
                    break
                end_index += 1

            # Ensure the chunk does not exceed max_allowed_chunk_size
            if (end_index - i) > max_allowed_chunk_size:
                end_index = i + max_allowed_chunk_size

            chunk_tokens = tokens[i:end_index]

            # Create chunk tags for each entity type
            chunk_tags = {
                ent: entity_token_tags[ent][i:end_index] for ent in entity_types
            }

            annotated_data.append(
                {
                    "gid": entry["id"],
                    "id": entry["id"] + f"_{i // chunk_size}",
                    "batch": batch_id,
                    "tokens": chunk_tokens,
                    "tags": chunk_tags,  # Dict[entity_type, List[BIO_tag]]
                }
            )

            i = end_index

    return annotated_data, entity_types


def annotate_corpus_multihead_centered(
    corpus,
    entity_types: List[str],
    batch_id: str = "b1",
    lang: str = "nl",
    chunk_size: int = 512,
):
    """
    Annotate corpus for Multi-Head CRF with chunks centered around entities.

    Args:
        corpus: List of documents with tags
        entity_types: List of entity type names
        batch_id: Batch identifier
        lang: Language for spaCy tokenizer
        chunk_size: Target chunk size in tokens

    Returns:
        annotated_data: List of annotated chunks
        entity_types: List of entity types
    """
    annotated_data = []
    nlp = spacy.blank(lang)

    for entry in tqdm(corpus, desc="Annotating centered for Multi-Head CRF"):
        text = entry["text"]
        tags = entry["tags"]

        # Tokenize
        doc = nlp(text)
        tokens = [token.text for token in doc]
        token_offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

        # Initialize tags for each entity type
        entity_token_tags = {ent: ["O"] * len(tokens) for ent in entity_types}

        # Annotate tokens
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            if tag_type not in entity_types:
                continue

            is_first_token = True
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue
                if token_start >= end:
                    break
                if (
                    (token_start >= start and token_end <= end)
                    or (token_start < start and token_end > start)
                    or (token_start < end and token_end > end)
                ):
                    if is_first_token:
                        entity_token_tags[tag_type][i] = "B"
                        is_first_token = False
                    else:
                        entity_token_tags[tag_type][i] = "I"

        # Create chunks centered around each entity span
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            if tag_type not in entity_types:
                continue

            # Find token indices for the span
            span_start_idx = None
            span_end_idx = None
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end >= start and span_start_idx is None:
                    span_start_idx = i
                if token_start >= end:
                    span_end_idx = i
                    break

            if span_start_idx is None:
                continue
            if span_end_idx is None:
                span_end_idx = len(tokens)

            # Center the chunk around the span
            left_context = max(0, span_start_idx - (chunk_size // 2))
            right_context = min(len(tokens), span_start_idx + (chunk_size // 2))

            if right_context - left_context < chunk_size:
                if left_context == 0:
                    right_context = min(len(tokens), left_context + chunk_size)
                elif right_context == len(tokens):
                    left_context = max(0, right_context - chunk_size)

            chunk_tokens = tokens[left_context:right_context]
            chunk_tags = {
                ent: entity_token_tags[ent][left_context:right_context]
                for ent in entity_types
            }

            annotated_data.append(
                {
                    "gid": entry["id"],
                    "id": entry["id"] + f"_span_{start}_{end}",
                    "batch": batch_id,
                    "tokens": chunk_tokens,
                    "tags": chunk_tags,
                }
            )

    return annotated_data, entity_types


def align_labels_with_tokens_multihead(
    labels_dict: Dict[str, List[str]], word_ids, bio_label2id: Dict[str, int]
):
    """
    Align multi-head labels with subword tokens.

    Args:
        labels_dict: Dict mapping entity types to BIO tag lists
        word_ids: Word IDs from tokenizer
        bio_label2id: Mapping of BIO tags to IDs {"O": 0, "B": 1, "I": 2}

    Returns:
        Dict mapping entity types to aligned label ID lists
    """
    aligned_labels = {}

    for entity_type, labels in labels_dict.items():
        new_labels = []
        current_word = None
        max_label_idx = len(labels) - 1

        for word_id in word_ids:
            if word_id is None:
                # Special token
                new_labels.append(-100)
            elif word_id != current_word:
                # Start of a new word
                current_word = word_id
                if word_id > max_label_idx:
                    new_labels.append(-100)
                else:
                    label_str = labels[word_id]
                    new_labels.append(bio_label2id.get(label_str, 0))
            else:
                # Same word as previous token (subword)
                if word_id > max_label_idx:
                    new_labels.append(-100)
                else:
                    label_str = labels[word_id]
                    label_id = bio_label2id.get(label_str, 0)
                    # If label is B (1), change to I (2) for continuation
                    if label_id == 1:  # B tag
                        label_id = 2  # Change to I tag
                    new_labels.append(label_id)

        aligned_labels[entity_type] = new_labels

    return aligned_labels


def tokenize_and_align_labels_multihead(
    docs, tokenizer, entity_types: List[str], max_length: Optional[int] = None
):
    """
    Tokenize documents and align multi-head labels with subword tokens.

    Args:
        docs: Dataset batch with "tokens" and "tags" (dict of entity -> BIO tags)
        tokenizer: HuggingFace tokenizer
        entity_types: List of entity type names
        max_length: Maximum sequence length

    Returns:
        Tokenized inputs with multi-head labels
    """
    bio_label2id = {"O": 0, "B": 1, "I": 2}

    tokenized_inputs = tokenizer(
        docs["tokens"],
        is_split_into_words=True,
        max_length=tokenizer.model_max_length if max_length is None else max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )

    # Process labels for each document
    all_labels = []
    for i in range(len(docs["tokens"])):
        word_ids = tokenized_inputs.word_ids(i)
        doc_tags = docs["tags"][i]  # Dict[entity_type, List[BIO_tag]]

        aligned = align_labels_with_tokens_multihead(doc_tags, word_ids, bio_label2id)
        all_labels.append(aligned)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def get_entity_types_from_corpus(corpus: List[Dict]) -> List[str]:
    """
    Extract unique entity types from a corpus.

    Args:
        corpus: List of documents with "tags" field

    Returns:
        Sorted list of unique entity types
    """
    entity_types = set()
    for entry in corpus:
        for tag in entry.get("tags", []):
            entity_types.add(tag["tag"])
    return sorted(list(entity_types))
