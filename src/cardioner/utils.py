import hashlib
import json
import os
import pprint
import warnings
from collections import defaultdict
from typing import Dict, List, Literal, Optional

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from spacy.lang.cs import Czech
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.it import Italian
from spacy.lang.nl import Dutch
from spacy.lang.ro import Romanian
from spacy.lang.sv import Swedish
from spacy.lang.xx import MultiLanguage
from tqdm import tqdm
from transformers import pipeline

lang_dict = {
    "es": Spanish,
    "nl": Dutch,
    "en": English,
    "it": Italian,
    "ro": Romanian,
    "sv": Swedish,
    "cz": Czech,
    "multi": MultiLanguage,
}


def _offset_tags(tags, offset):
    replacement = []
    for _tags in tags:
        _tags["start"] += offset
        _tags["end"] += offset
        replacement.append(_tags)
    return replacement


def pipe_with_progress(pipe, texts, batch_size=16, **kwargs):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Running pipeline"):
        batch = texts[i : i + batch_size]
        batch_results = pipe(batch, **kwargs)
        results.extend(batch_results)
    return results


def process_pipe(
    text: str | Dataset | List[str],
    pipe: pipeline,
    max_word_per_chunk: int = 256,
    hf_stride: bool = True,
    batch_size: int = 16,
    lang: Literal["es", "nl", "en", "it", "ro", "sv", "cz", "multi"] = "en",
) -> List[Dict[str, str]] | List[List[Dict[str, str]]]:
    """
    text: The text to process
    pipe: The transformers pipeline to use
    max_word_per_chunk: The maximum number of words per chunk,
      we need this to avoid exceeding the maximum input size of the model
    lang: The language of the text
    batch_size: batch_size in case the input is a Dataset
    hf_stride: use stride that is part of the huggingface pipe
    """
    assert lang in ["es", "nl", "en", "it", "ro", "sv", "cz", "multi"], (
        f"Language {lang} not supported"
    )
    assert (
        isinstance(text, str) or isinstance(text, Dataset) or isinstance(text, List)
    ), f"Text must be of type str or Dataset, got {type(text)}"

    if not hf_stride and isinstance(text, Dataset):
        hf_stride = True
        warnings.warn(
            "The dataset loader only works with hf_stride==True, continuing with HF stride"
        )

    if hf_stride:
        if isinstance(text, Dataset):
            # Extract texts from the Dataset
            texts = text["text"]
            named_ents = pipe_with_progress(
                pipe, texts, batch_size, stride=max_word_per_chunk
            )
        else:
            if isinstance(text, list):
                named_ents = []
                for _text in text:
                    assert isinstance(_text, str), (
                        f"_text should be a string here but instead is {type(_text)}"
                    )
                    _ents = pipe(_text)
                    named_ents.extend(_ents)
            else:
                named_ents = pipe(text, stride=max_word_per_chunk)
    else:
        raise NotImplementedError(
            "..Use hf_stride=True for now..and probably forever :D"
        )
        nlp = lang_dict[lang]()
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        # only necessary if no FastTokenizer is available or aggregation_strategy is None
        sentence_bag = []
        word_count = 0
        named_ents = []
        offset = 0
        for sent in doc.sents:
            word_count += len(sent)
            if word_count > max_word_per_chunk:
                txt = " ".join(sentence_bag)
                _named_ents = pipe(txt)
                # add offsets
                if offset > 0:
                    _named_ents = _offset_tags(_named_ents, offset)
                named_ents.extend(_named_ents)
                sentence_bag = []
                word_count = len(sent)
                offset += len(txt)
            sentence_bag.append(sent.text)
        if len(sentence_bag) > 0:
            _named_ents = pipe(".".join(sentence_bag))
            named_ents.extend(_named_ents)
            if offset > 0:
                _named_ents = _offset_tags(_named_ents, offset)
    return named_ents


def pretty_print_classifier(classifier):
    """Pretty print a PyTorch module with indentation and structure details."""
    import pprint

    import torch.nn as nn

    pp = pprint.PrettyPrinter(indent=4, width=100)

    if isinstance(classifier, nn.Linear):
        return f"Linear(in_features={classifier.in_features}, out_features={classifier.out_features})"

    if isinstance(classifier, nn.Sequential):
        # Create a structured representation as a dictionary
        layers_info = {}
        for i, layer in enumerate(classifier):
            layer_info = {"type": layer.__class__.__name__}

            # Add details based on layer type
            if isinstance(layer, nn.Linear):
                layer_info["in_features"] = layer.in_features
                layer_info["out_features"] = layer.out_features
            elif isinstance(layer, nn.Dropout):
                layer_info["p"] = layer.p
            elif isinstance(layer, nn.BatchNorm1d):
                layer_info["num_features"] = layer.num_features
            elif hasattr(layer, "in_channels") and hasattr(layer, "out_channels"):
                layer_info["in_channels"] = layer.in_channels
                layer_info["out_channels"] = layer.out_channels

            layers_info[f"layer_{i}"] = layer_info

        # Use pprint to format the dictionary
        return f"Sequential with {len(classifier)} layers:\n{pp.pformat(layers_info)}"

    # For other module types, convert to dict and pretty print
    try:
        # Try to extract meaningful attributes
        module_info = {
            "type": classifier.__class__.__name__,
            "parameters": {
                name: param.size() for name, param in classifier.named_parameters()
            },
        }
        return pp.pformat(module_info)
    except:
        # Fallback
        return str(classifier)


def calculate_class_weights(
    dataset, label2id, multiclass=False, smoothing_factor=0.001, max_weight=50.0
):
    """
    Calculate class weights based on label frequency in the dataset.

    Args:
        dataset: Training dataset containing token labels
        label2id: Dictionary mapping label names to ids
        multilabel: Whether labels are multilabel (one-hot encoded)
        smoothing_factor: Smoothing factor to avoid extreme weights (unused, kept for compatibility)
        max_weight: Maximum weight to prevent numerical instability (default: 50.0)

    Returns:
        List of class weights
    """
    label_counts = defaultdict(int)
    total_tokens = 0

    print("Extracting class weights from training set...")
    # Count label occurrences
    for example in tqdm(dataset):
        labels = example["labels"]

        if multiclass == False:
            # For multilabel: labels are one-hot encoded [seq_length, num_labels]
            # Convert to tensor if it's a list
            labels_tensor = (
                torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
            )

            # Skip padding tokens (assuming padding tokens have all zeros or special value)
            valid_tokens = (
                (labels_tensor.sum(dim=-1) > 0)
                if labels_tensor.ndim > 1
                else (labels_tensor != -100)
            )

            # Count each class occurrence
            if labels_tensor.ndim > 1:  # One-hot encoded
                for i in range(labels_tensor.shape[1]):  # For each class
                    class_occurrences = labels_tensor[:, i].sum().item()
                    label_counts[i] += class_occurrences
                    total_tokens += valid_tokens.sum().item()
            else:
                # Handle case where labels might be indices instead of one-hot
                for label_idx in labels_tensor[valid_tokens]:
                    label_counts[label_idx.item()] += 1
                    total_tokens += 1
        else:
            # For single-label classification
            for label in labels:
                if label != -100:  # Skip padding tokens
                    label_counts[label] += 1
                    total_tokens += 1

    # Calculate weights inversely proportional to frequency
    weights = []
    num_classes = len(label2id)

    for i in range(num_classes):
        count = label_counts.get(i, 1)  # Use 1 instead of 0 to avoid div by zero
        # Simple inverse frequency: total / (num_classes * count)
        weight = total_tokens / (num_classes * count)
        weights.append(weight)

    # Normalize relative to O (class 0) - keeps O at 1.0, scales others proportionally
    # This avoids numerical instability from very small O weights
    o_weight = weights[0] if weights[0] > 0 else 1.0
    weights = [w / o_weight for w in weights]

    # Cap weights to prevent numerical instability (gradient explosion -> nan loss)
    original_weights = weights.copy()
    weights = [min(w, max_weight) for w in weights]

    id2label = {i: label for i, label in enumerate(label2id)}
    print(
        f"Extracted class weights (before cap): {[f'{id2label[i]}_{w:.2f}' for i, w in enumerate(original_weights)]}"
    )
    print(
        f"Extracted class weights (capped at {max_weight}): {[f'{id2label[i]}_{w:.2f}' for i, w in enumerate(weights)]}"
    )
    return weights


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


def fix_tokenizer(
    vocab_file: str,
    initial_encoding: Literal["latin-1", "cp1252"] = "cp1252",
    prefix="Ġ",
):
    """
    Given a vocab.json we go through the tokens and correct the encoding garbling with fix_misdecoded_string
    We save the original vocab.json as vocab_bak.json and write the corrected one to vocab.json

    We assume we want to convert to utf8
    """
    import shutil

    # check if vocab_file is an existing json file
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

    if not vocab_file.endswith(".json"):
        raise ValueError(f"Expected a JSON file, got: {vocab_file}")

    try:
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {vocab_file}. Error: {e}")

    # Create backup of original vocab file
    backup_file = vocab_file.replace(".json", "_bak.json")
    shutil.copy2(vocab_file, backup_file)
    print(f"Created backup: {backup_file}")

    # Fix encoding issues in the vocabulary
    fixed_vocab = {}
    vocab_map = {}
    fixed_count = 0

    broken_tokens = []
    for token, token_id in vocab.items():
        # Try to fix the token encoding
        ignore = False
        if token.startswith(prefix):
            add_prefix = True
            if sum([c == prefix for c in token]) > 1:
                ignore = True
        else:
            add_prefix = False

        if add_prefix:
            _token = token[1:]
        else:
            _token = token

        if ignore == True:
            fixed_token = _token
        else:
            fixed_token = fix_misdecoded_string(_token)

        # Check if the token was actually changed
        if (fixed_token != _token) and (ignore == False):
            fixed_count += 1
            broken_tokens.append(token)
            print(f"Fixed token: '{token}' -> '{fixed_token}'")

        if token == "ĠĠ":
            print(fixed_token, "|", token, "|", ignore, "|", add_prefix, "|", token_id)

        if add_prefix:
            fixed_token = prefix + fixed_token

        if token == "ĠĠ":
            print(fixed_token, "|", token, "|", ignore, "|", add_prefix, "|", token_id)
        vocab_map[token] = fixed_token
        fixed_vocab[fixed_token] = token_id

    # Write the corrected vocabulary back to the original file
    with open(f"{vocab_file}.fix", "w", encoding="utf-8") as f:
        json.dump(fixed_vocab, f, ensure_ascii=False, indent=2)

    print(f"Fixed {fixed_count} tokens in vocabulary")
    print(f"Corrected vocabulary saved to: {vocab_file}")

    # Given the fixed vocab, we now need to change the merges.txt
    # We can do this by iterating over the vocab and checking if the token is a merge
    # If it is, we can add the merge to the merges.txt file
    vocab_folder = os.path.dirname(vocab_file)
    merges_new = []
    # go through lines, split by " ", check if [0] is in fixed_vocab and replace with that term
    with open(f"{vocab_folder}/merges.txt", "r", encoding="utf-8") as fr:
        fr.readline()
        for line in fr:
            token, merge = line.strip().split(" ")

            new_token = vocab_map.get(token, token)
            new_merge = vocab_map.get(merge, merge)
            merges_new.append((new_token, new_merge))

    # write to new merges.txt file
    with open(f"{vocab_folder}/merges.txt.fix", "w", encoding="utf-8") as fw:
        for token, merge in merges_new:
            fw.write(f"{token} {merge}\n")


def merge_annotations(
    annotation_directory: str,
    merge_key: str = "id",
    tag_key: str = "tags",
    text_key: str = "text",
    annotation_tsv: str | None = None,
) -> List[Dict]:
    # go through .jsonl's/.txts in directory
    #

    if isinstance(annotation_tsv, str):
        annotation_df = pd.read_csv(annotation_tsv, sep="\t")
    else:
        annotation_df = None
    annotations = []
    for _file in tqdm(os.listdir(annotation_directory)):
        if _file.endswith(".jsonl"):
            file = os.path.join(annotation_directory, _file)
            with open(file, "r", encoding="utf-8") as fr:
                for line in fr:
                    annotations.append(json.loads(line))
        elif _file.endswith(".txt"):
            if annotation_df is None:
                UserWarning(
                    "The annotation tsv needs to be provided if you parse a list of .txts"
                )
            file = os.path.join(annotation_directory, _file)
            with open(file, "r", encoding="utf-") as fr:
                text = fr.read()
            fn = _file.split(".")[0]
            d = {merge_key: fn, text_key: text}
            if annotation_df is None:
                d[tag_key] = []
            else:
                # name, tag, start_span, end_span
                # filename, label, start_span, end_span
                r = annotation_df.query(f'filename=="{fn.strip()}"')[
                    ["label", "start_span", "end_span"]
                ]
                d[tag_key] = [
                    {"tag": row[0], "start": row[1], "end": row[2]} for row in r
                ]
            annotations.append(d)
    if len(annotations) == 0:
        raise ValueError(
            "No JSONL or TXT file found in the directory, maybe check the directory?"
        )

    NEW_DICT = defaultdict(lambda: defaultdict(list))
    for d in tqdm(annotations):
        # list of tags
        tags = d[tag_key]
        text = d[text_key]
        id = d[merge_key]

        # assert tags is not None, f"Tags should not be none: {id}"

        NEW_DICT[id][tag_key].extend(tags)
        NEW_DICT[id][text_key].extend([text])

    NEW_DICT_LIST = []
    for k, v in tqdm(NEW_DICT.items()):
        # check if text is consistent
        #
        list_of_texts = v[text_key]
        set_of_hashes = set()
        for t in list_of_texts:
            _hash = hashlib.md5(t.encode("utf-8")).hexdigest()
            set_of_hashes.add(_hash)

        if len(set_of_hashes) > 1:
            print(
                f"Skipping {k} because there are {len(set_of_hashes)} different texts"
            )
            continue

        if len(v[tag_key]) == 0:
            vt = None
        else:
            vt = v[tag_key]
        _d = {"id": k, "tags": vt, "text": list_of_texts[0]}
        NEW_DICT_LIST.append(_d)
    return NEW_DICT_LIST


def clean_spans(
    entities,
    original_text,
    lang: str = "nl",
    trim_trailing_cutoff_words_enabled: bool = False,
    numeric_only_allowed_tags: Optional[List[str] | set[str]] = None,
):
    """
    Apply post-hoc cleaning to entity spans based on predictor_manuela.py logic.

    This function performs text cleaning and entity validation on detected spans:
    - Removes trailing space + punctuation if no opening parenthesis
    - Removes trailing closing parenthesis if no opening parenthesis
    - Removes leading whitespace
    - Does not strip language-specific articles
    - Validates entities (non-empty, not numeric except for configured tags like AGE,
      not special chars)

    Args:
        entities (list): List of entity dictionaries with 'start', 'end', 'text', 'tag', 'score' keys
        original_text (str): The original text from which entities were extracted
        lang (str): Language code used for language-specific cleanup rules.
        trim_trailing_cutoff_words_enabled (bool): If True, trim spans at language-specific
            trailing cutoff words (e.g. prepositions/conjunctions). Disabled by default.
        numeric_only_allowed_tags (Optional[List[str] | set[str]]): Tag names for which
            numeric-only spans are allowed (e.g. ["AGE"]). Defaults to {"AGE"}.

    Returns:
        list: Cleaned entities with updated spans
    """
    import re

    def is_special_char(text):
        """Check if text consists only of special characters."""
        return bool(re.fullmatch(r"\W+", text.strip()))

    # Language-aware trailing cutoff words: trim only when cutoff words occur at the very end
    TRAILING_CUTOFF_WORDS_BY_LANG = {
        "nl": {"van", "met", "in", "op", "door", "als"},
        "en": {"of", "with", "in", "on", "by", "as"},
        "sv": {"av", "med", "i", "på", "genom", "som"},
        "it": {"di", "con", "in", "su", "da", "come"},
        "ro": {"de", "cu", "în", "in", "pe", "prin", "ca"},
        "cz": {"z", "s", "v", "na", "přes", "pres", "jako"},
        "es": {"de", "con", "en", "sobre", "por", "como"},
    }

    def trim_trailing_cutoff_words(entity_text, start_span, end_span, lang_code):
        if not entity_text:
            return entity_text, start_span, end_span

        cutoff_words = TRAILING_CUTOFF_WORDS_BY_LANG.get(
            (lang_code or "nl").lower(), TRAILING_CUTOFF_WORDS_BY_LANG["nl"]
        )

        changed = True
        while changed and entity_text:
            changed = False
            lowered = entity_text.lower()

            for word in cutoff_words:
                # Match cutoff word only at the end (optionally preceded by whitespace)
                pattern = r"(?:^|\s)" + re.escape(word) + r"\s*$"
                match = re.search(pattern, lowered)
                if match:
                    entity_text = entity_text[: match.start()].rstrip()
                    end_span = start_span + len(entity_text)
                    changed = True
                    break

        return entity_text, start_span, end_span

    if numeric_only_allowed_tags is None:
        normalized_numeric_only_allowed_tags = {"AGE"}
    else:
        normalized_numeric_only_allowed_tags = {
            str(tag).strip().upper().removeprefix("B-").removeprefix("I-")
            for tag in numeric_only_allowed_tags
            if str(tag).strip() != ""
        }

    cleaned_entities = []

    for entity in entities:
        # Get the entity text and span boundaries
        entity_text = original_text[entity["start"] : entity["end"]]
        start_span = entity["start"]
        end_span = entity["end"]

        # If span contains an unmatched opening parenthesis, extend to the next closing parenthesis
        if entity_text.count("(") > entity_text.count(")"):
            next_closing = original_text.find(")", end_span)
            if next_closing != -1:
                end_span = next_closing + 1
                entity_text = original_text[start_span:end_span]

        # Iterative edge cleanup:
        # keep trimming until no more leading/trailing whitespace or trailing symbols can be removed
        bracket_pairs = {"(": ")", "[": "]", "{": "}"}
        changed = True
        while changed and len(entity_text) > 0:
            changed = False

            # Remove leading whitespace (including spaces, tabs, and line breaks)
            while len(entity_text) > 0 and entity_text[0].isspace():
                entity_text = entity_text[1:]
                start_span = start_span + 1
                changed = True

            # Remove trailing whitespace (including spaces, tabs, and line breaks)
            while len(entity_text) > 0 and entity_text[-1].isspace():
                entity_text = entity_text[:-1]
                end_span = end_span - 1
                changed = True

            if len(entity_text) == 0:
                break

            # Remove trailing space + punctuation if no opening parenthesis
            if (
                len(entity_text) >= 2
                and entity_text[-2] == " "
                and entity_text[-1] in ".,;:!?"
                and "(" not in entity_text[:-2]
            ):
                entity_text = entity_text[:-2]
                end_span = end_span - 2
                changed = True
                continue

            # Remove attached trailing punctuation if no opening parenthesis
            if entity_text[-1] in ".,;:!?" and "(" not in entity_text[:-1]:
                entity_text = entity_text[:-1]
                end_span = end_span - 1
                changed = True
                continue

            # Remove trailing closing parenthesis if no opening parenthesis
            if entity_text[-1] == ")" and "(" not in entity_text:
                entity_text = entity_text[:-1]
                end_span = end_span - 1
                changed = True
                continue

            # Trim mismatched leading '(' when there is no closing ')'
            if entity_text.startswith("(") and ")" not in entity_text:
                entity_text = entity_text[1:]
                start_span = start_span + 1
                changed = True
                continue

            # Symmetric bracket trimming when both sides are present
            if len(entity_text) >= 2:
                opener = entity_text[0]
                closer = entity_text[-1]
                if opener in bracket_pairs and bracket_pairs[opener] == closer:
                    entity_text = entity_text[1:-1]
                    start_span = start_span + 1
                    end_span = end_span - 1
                    changed = True

        if trim_trailing_cutoff_words_enabled:
            # Trim language-aware trailing cutoff words (e.g. nl/en/sv/it/ro/cz/es)
            entity_text, start_span, end_span = trim_trailing_cutoff_words(
                entity_text, start_span, end_span, lang
            )

            # Cleanup possible whitespace left after cutoff-word trimming
            while len(entity_text) > 0 and entity_text[-1].isspace():
                entity_text = entity_text[:-1]
                end_span = end_span - 1

        # Entity validation
        # Check if text is purely numeric (including decimals)
        def is_numeric_only(text):
            try:
                float(text.strip())
                return True
            except ValueError:
                return text.strip().isnumeric()

        def is_all_caps_with_numbers(text):
            stripped = text.strip()
            return (
                bool(re.fullmatch(r"[A-Z0-9]+", stripped))
                and any(ch.isdigit() for ch in stripped)
                and any(ch.isalpha() for ch in stripped)
            )

        raw_tag = str(entity.get("tag", ""))
        normalized_tag = raw_tag.upper().removeprefix("B-").removeprefix("I-")

        tag_blocks_caps_numeric = normalized_tag in {
            "PROCEDURE",
            "SYMPTOM",
            "DISEASE",
        }

        numeric_only_disallowed_for_tag = (
            is_numeric_only(entity_text)
            and normalized_tag not in normalized_numeric_only_allowed_tags
        )

        if (
            entity_text.strip() != ""
            and not numeric_only_disallowed_for_tag
            and not is_special_char(entity_text)
            and not (tag_blocks_caps_numeric and is_all_caps_with_numbers(entity_text))
        ):
            # Create cleaned entity
            cleaned_entity = entity.copy()
            cleaned_entity["text"] = entity_text
            cleaned_entity["start"] = start_span
            cleaned_entity["end"] = end_span
            cleaned_entities.append(cleaned_entity)

    return cleaned_entities


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix tokenizer encoding issues")
    parser.add_argument(
        "--vocab-file",
        type=str,
        required=True,
        help="Path to the vocab.json file to fix",
    )
    parser.add_argument(
        "--initial-encoding",
        type=str,
        default="cp1252",
        choices=["latin-1", "cp1252"],
        help="Initial encoding to fix from (default: cp1252)",
    )

    args = parser.parse_args()

    fix_tokenizer(args.vocab_file, args.initial_encoding)
