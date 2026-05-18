import glob
import os
import shutil
from os import environ
from pathlib import Path

environ["WANDB_MODE"] = "disabled"
environ["WANDB_DISABLED"] = "true"

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import logging

logging.set_verbosity_debug()

from cardioner.utils import pretty_print_classifier

try:
    from cardioner.multilabel.modeling import MultiLabelTokenClassificationModelCustom
except ImportError as e_abs:
    try:
        from .modeling import MultiLabelTokenClassificationModelCustom
    except ImportError as e_rel:
        print(
            "Warning: Could not import MultiLabelTokenClassificationModelCustom. "
            f"Absolute import error: {e_abs}. Relative import error: {e_rel}"
        )
        MultiLabelTokenClassificationModelCustom = None

import evaluate

metric = evaluate.load("seqeval")


class WhoAmI(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        print("[DBG] on_save:", kw["model"].__class__.__name__)

    def on_evaluate(self, args, state, control, **kw):
        print("[DBG] on_eval:", kw["model"].__class__.__name__)

    def on_train_end(self, args, state, control, **kw):
        print("[DBG] end:", kw["model"].__class__.__name__)


@dataclass
class MultiLabelDataCollatorForTokenClassification:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = "max_length"
    max_length: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features, *args, **kwargs):
        labels = [feature.pop("labels") for feature in features]
        # Remove unnecessary keys if needed

        # Pad the inputs using the tokenizer's pad method
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Convert labels to tensors
        labels_tensors = [torch.tensor(label, dtype=torch.float) for label in labels]
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_tensors, batch_first=True, padding_value=self.label_pad_token_id
        )

        # Ensure labels are padded to max_seq_length
        max_seq_length = batch["input_ids"].shape[1]
        if padded_labels.shape[1] < max_seq_length:
            padding_size = max_seq_length - padded_labels.shape[1]
            padding = torch.full(
                (len(labels), padding_size, padded_labels.shape[2]),
                fill_value=self.label_pad_token_id,
                dtype=torch.float,
            )
            padded_labels = torch.cat([padded_labels, padding], dim=1)
        elif padded_labels.shape[1] > max_seq_length:
            padded_labels = padded_labels[:, :max_seq_length, :]

        batch["labels"] = padded_labels

        return batch


@dataclass
class MultiLabelDataCollatorWordLevel:
    """Data collator that also passes word_ids for word-level loss computation."""

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = "max_length"
    max_length: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features, *args, **kwargs):
        labels = [feature.pop("labels") for feature in features]
        word_ids_list = [feature.pop("word_ids", None) for feature in features]

        # Pad the inputs using the tokenizer's pad method
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Convert labels to tensors
        labels_tensors = [torch.tensor(label, dtype=torch.float) for label in labels]
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_tensors, batch_first=True, padding_value=self.label_pad_token_id
        )

        # Ensure labels are padded to max_seq_length
        max_seq_length = batch["input_ids"].shape[1]
        if padded_labels.shape[1] < max_seq_length:
            padding_size = max_seq_length - padded_labels.shape[1]
            padding = torch.full(
                (len(labels), padding_size, padded_labels.shape[2]),
                fill_value=self.label_pad_token_id,
                dtype=torch.float,
            )
            padded_labels = torch.cat([padded_labels, padding], dim=1)
        elif padded_labels.shape[1] > max_seq_length:
            padded_labels = padded_labels[:, :max_seq_length, :]

        batch["labels"] = padded_labels

        # Process word_ids: convert None to -1 for special tokens, pad to max_seq_length
        if word_ids_list[0] is not None:
            padded_word_ids = []
            for wids in word_ids_list:
                # Convert None to -1 for special tokens
                wids_converted = [w if w is not None else -1 for w in wids]
                padded_word_ids.append(torch.tensor(wids_converted, dtype=torch.long))

            padded_word_ids = torch.nn.utils.rnn.pad_sequence(
                padded_word_ids, batch_first=True, padding_value=-1
            )

            if padded_word_ids.shape[1] < max_seq_length:
                padding = torch.full(
                    (len(word_ids_list), max_seq_length - padded_word_ids.shape[1]),
                    fill_value=-1,
                    dtype=torch.long,
                )
                padded_word_ids = torch.cat([padded_word_ids, padding], dim=1)
            elif padded_word_ids.shape[1] > max_seq_length:
                padded_word_ids = padded_word_ids[:, :max_seq_length]

            batch["word_ids"] = padded_word_ids

        return batch


class MultiLabelTokenClassificationModelHF(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Use backbone_model_name if available, fallback to name_or_path
        # This is critical because name_or_path gets overwritten during save/load
        backbone_name = getattr(config, "backbone_model_name", None)
        if backbone_name is None:
            backbone_name = getattr(config, "name_or_path", None)
        if backbone_name is None:
            raise ValueError(
                "config.backbone_model_name (or config.name_or_path) is required to load pretrained backbone"
            )

        # Create a clean config for the backbone
        backbone_config = AutoConfig.from_pretrained(
            backbone_name, trust_remote_code=True, use_safetensors=True
        )
        backbone_config.hidden_dropout_prob = getattr(
            config, "hidden_dropout_prob", 0.1
        )

        self.roberta = AutoModel.from_pretrained(
            backbone_name,
            config=backbone_config,
            trust_remote_code=True,
            use_safetensors=True,
        )

        # Store backbone_model_name in config for future loading
        if (
            not hasattr(config, "backbone_model_name")
            or config.backbone_model_name is None
        ):
            config.backbone_model_name = backbone_name

        freeze_backbone = getattr(config, "freeze_backbone", False)
        if freeze_backbone:
            for param in self.roberta.parameters():
                param.requires_grad = False
            self.roberta.eval()
        else:
            self.roberta.train()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    @classmethod
    def from_config(cls, config, **kwargs):
        """Override from_config to ensure config is properly passed to __init__"""
        return cls(config=config, **kwargs)

    def train(self, mode=True):
        """Override train method to keep frozen backbone in eval mode"""
        super().train(mode)
        if hasattr(self.config, "freeze_backbone") and self.config.freeze_backbone:
            self.roberta.eval()
        return self

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Compute loss here if necessary
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            mask = (labels != -100).float()
            labels = torch.where(labels == -100, torch.zeros_like(labels), labels)
            loss_tensor = loss_fct(logits, labels.float())
            loss = (loss_tensor * mask).sum() / mask.sum()

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )


class MultiLabelTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: Optional[torch.FloatTensor] = None,
        id2label: Dict[int, str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)
            print(
                f"Using multi-label classification with class weights: {class_weights}"
            )
        self.loss_fct = nn.BCEWithLogitsLoss(weight=class_weights, reduction="none")

        # Class counting for debugging
        self.id2label = id2label or {}
        self.class_counts = None
        self.total_tokens = 0
        self.batch_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # === COUNT CLASSES IN THIS BATCH ===
        # Flatten labels: (batch, seq_len, num_labels) -> (batch*seq_len, num_labels)
        labels_flat = labels.view(-1, labels.shape[-1])

        # Mask out padding tokens (where all values are -100)
        valid_mask = ~(labels_flat == -100).all(dim=-1)
        valid_labels = labels_flat[valid_mask]

        # Count positives for each class
        batch_counts = (valid_labels == 1).sum(dim=0).cpu()

        if self.class_counts is None:
            self.class_counts = batch_counts
        else:
            self.class_counts += batch_counts

        self.total_tokens += valid_mask.sum().item()
        self.batch_count += 1

        # Print every 100 batches
        if self.batch_count % 100 == 0:
            print(
                f"\n=== Class counts after {self.batch_count} batches ({self.total_tokens} tokens) ==="
            )
            for i, count in enumerate(self.class_counts.tolist()):
                label_name = self.id2label.get(i, f"class_{i}")
                pct = 100 * count / self.total_tokens if self.total_tokens > 0 else 0
                print(f"  {label_name}: {count} ({pct:.2f}%)")
            print("=" * 50 + "\n")
        # === END COUNTING ===
        #
        # # Compute the loss tensor with reduction='none'
        loss_tensor = self.loss_fct(
            logits.view(-1, model.num_labels), labels.view(-1, model.num_labels)
        )
        # Apply mask to ignore padding (-100 labels)
        mask = (labels.view(-1, model.num_labels) != -100).float()
        loss_tensor = loss_tensor * mask

        # Reduce the loss tensor to a scalar
        loss = loss_tensor.sum() / mask.sum()

        return (loss, outputs) if return_outputs else loss


class MultiLabelTrainerWordLevel(Trainer):
    """
    Trainer that computes loss at the word level instead of token level.

    This ensures training is aligned with word-level evaluation by aggregating
    subword token predictions before computing the loss.
    """

    def __init__(
        self,
        *args,
        class_weights: Optional[torch.FloatTensor] = None,
        word_aggregation: str = "first",  # "first", "last", "mean"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)
            print(
                f"Using multi-label classification with class weights: {class_weights}"
            )
        self.loss_fct = nn.BCEWithLogitsLoss(weight=class_weights, reduction="none")
        self.word_aggregation = word_aggregation
        print(
            f"Word-level training enabled with aggregation strategy: {word_aggregation}"
        )

    def aggregate_to_word_level(self, logits, word_ids, labels):
        """
        Aggregate token-level logits to word-level before computing loss.

        Args:
            logits: (batch_size, seq_len, num_labels)
            word_ids: (batch_size, seq_len) - word index for each token, -1 for special/padding
            labels: (batch_size, seq_len, num_labels) - token-level labels

        Returns:
            word_logits, word_labels: aggregated at word level
        """
        batch_size, seq_len, num_labels = logits.shape
        device = logits.device

        all_word_logits = []
        all_word_labels = []

        for b in range(batch_size):
            # Get unique word indices (excluding -1 for special tokens/padding)
            unique_words = word_ids[b].unique()
            unique_words = unique_words[unique_words >= 0]  # exclude -1

            for word_idx in unique_words:
                # Find all tokens belonging to this word
                token_mask = word_ids[b] == word_idx
                token_indices = token_mask.nonzero(as_tuple=True)[0]

                if len(token_indices) == 0:
                    continue

                # Aggregate logits based on strategy
                word_token_logits = logits[b, token_indices]  # (num_tokens, num_labels)

                if self.word_aggregation == "first":
                    word_logit = word_token_logits[0]
                elif self.word_aggregation == "last":
                    word_logit = word_token_logits[-1]
                elif self.word_aggregation == "mean":
                    word_logit = word_token_logits.mean(dim=0)
                else:
                    raise ValueError(f"Unknown aggregation: {self.word_aggregation}")

                # Get word-level label (use first token's label - they should all be same at word level)
                word_label = labels[b, token_indices[0]]

                # Skip if this is a padding label
                if (word_label == -100).any():
                    continue

                all_word_logits.append(word_logit)
                all_word_labels.append(word_label)

        if len(all_word_logits) == 0:
            return None, None

        word_logits = torch.stack(all_word_logits)  # (total_words, num_labels)
        word_labels = torch.stack(all_word_labels)  # (total_words, num_labels)

        return word_logits, word_labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        word_ids = inputs.pop("word_ids", None)
        outputs = model(**inputs)
        logits = outputs.logits

        if word_ids is not None:
            # Word-level loss
            word_logits, word_labels = self.aggregate_to_word_level(
                logits, word_ids, labels
            )

            if word_logits is None:
                # Fallback to zero loss if aggregation produces nothing
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            else:
                loss_tensor = self.loss_fct(word_logits, word_labels.float())
                loss = loss_tensor.mean()
        else:
            # Fallback: token-level loss (original behavior)
            loss_tensor = self.loss_fct(
                logits.view(-1, model.num_labels), labels.view(-1, model.num_labels)
            )
            mask = (labels.view(-1, model.num_labels) != -100).float()
            loss_tensor = loss_tensor * mask
            loss = loss_tensor.sum() / mask.sum()

        return (loss, outputs) if return_outputs else loss


class ModelTrainer:
    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        tokenizer=None,
        model: str = "CLTL/MedRoBERTa.nl",
        batch_size: int = 48,
        max_length: int = 514,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.001,
        gradient_accumulation_steps: int = 1,
        num_train_epochs: int = 20,
        output_dir: str = "../../output",
        hf_token: str = None,
        freeze_backbone: bool = False,
        classifier_hidden_layers: tuple | None = None,
        classifier_dropout: float = 0.1,
        class_weights: List[float] | None = None,
        word_level: bool = False,
        word_aggregation: str = "mean",  # "first", "last", "mean"
    ):
        self.model_name_or_path = model
        self.hf_token = hf_token
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float)
            if class_weights is not None
            else None
        )
        if self.class_weights is not None:
            # Ensure all weights are positive
            self.class_weights = torch.abs(self.class_weights)
            print("Class weights are positive")
            print(f"Weight: {self.class_weights}")
            print("#" * 100)

        self.label2id = label2id
        self.id2label = id2label
        self.output_dir = output_dir

        # Word-level training configuration
        self.word_level = word_level
        self.word_aggregation = word_aggregation
        if word_level:
            print(
                f"Word-level training enabled with '{word_aggregation}' aggregation strategy"
            )

        self.train_kwargs = {
            "run_name": "CardioNER",
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": 0,
            "num_train_epochs": num_train_epochs,
            "weight_decay": weight_decay,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "report_to": "tensorboard",
            "use_cpu": False,
            "fp16": True,
            "logging_dir": f"{output_dir}/logs",
            "logging_strategy": "steps",
            "logging_steps": 256,
            "load_best_model_at_end": True,
            "greater_is_better": True,
            "metric_for_best_model": "f1_macro",
        }

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model,
                add_prefix_space=True,
                model_max_length=max_length,
                padding=False,
                truncation=False,
                token=hf_token,
            )
        else:
            self.tokenizer = tokenizer

        self.tokenizer.model_max_length = max_length
        # Select data collator based on word_level setting
        if self.word_level:
            self.data_collator = MultiLabelDataCollatorWordLevel(
                tokenizer=self.tokenizer,
                padding="max_length",
                max_length=max_length,
                label_pad_token_id=-100,
            )
        else:
            self.data_collator = MultiLabelDataCollatorForTokenClassification(
                tokenizer=self.tokenizer,
                padding="max_length",
                max_length=max_length,
                label_pad_token_id=-100,
            )
        or_config = AutoConfig.from_pretrained(
            model,
            hf_token=self.hf_token,
            return_unused_kwargs=False,
            trust_remote_code=True,
        )
        or_config.num_labels = len(self.label2id)
        or_config.id2label = self.id2label
        or_config.label2id = self.label2id
        or_config.hidden_dropout_prob = 0.1
        or_config.freeze_backbone = freeze_backbone
        or_config.output_hidden_states = False
        or_config.output_attentions = False
        # Store the original backbone model name for proper loading later
        # This is critical because name_or_path gets overwritten during save/load
        or_config.backbone_model_name = model

        if isinstance(classifier_hidden_layers, tuple):
            if MultiLabelTokenClassificationModelCustom is None:
                raise ImportError(
                    "MultiLabelTokenClassificationModelCustom could not be imported. Please ensure modeling.py is available."
                )

            print(
                "Warning: You are now creating a custom model class which requires trust_remote_code=True to load!"
            )
            base_model = AutoModel.from_pretrained(
                model, token=hf_token, trust_remote_code=True, use_safetensors=True
            )

            # Store custom parameters in config for proper saving/loading
            or_config.classifier_hidden_layers = classifier_hidden_layers
            or_config.classifier_dropout = classifier_dropout
            or_config.auto_map = {
                "AutoModelForTokenClassification": "modeling.MultiLabelTokenClassificationModelCustom"
            }
            self.or_config = or_config

            self.custom_kwargs = {
                "config": or_config,
                "base_model": base_model,
                "freeze_backbone": freeze_backbone,
                "classifier_hidden_layers": classifier_hidden_layers,
                "classifier_dropout": classifier_dropout,
            }

            self.model = MultiLabelTokenClassificationModelCustom(**self.custom_kwargs)
            # Set up auto_map for trust_remote_code loading
            # self.model.config.auto_map = {
            #    "AutoModelForTokenClassification": "modeling.MultiLabelTokenClassificationModelCustom"
            # }
            self.custom_model = True
        else:
            self.model = MultiLabelTokenClassificationModelHF.from_pretrained(
                model, config=or_config
            )
            self.custom_model = False
            self.or_config = or_config

        print("Tokenizer max length:", self.tokenizer.model_max_length)
        print(
            "Model max position embeddings:", self.model.config.max_position_embeddings
        )
        print("Number of labels:", len(self.label2id))
        print("Labels:", self.label2id)
        print("id2label:", self.id2label)
        print("Model config:", self.model.config)
        print("Head only fine-tuning:", freeze_backbone)
        if freeze_backbone:
            print(
                f"Hidden layers: {classifier_hidden_layers},{type(classifier_hidden_layers)}"
            )
        print(
            "Classifier architecture:", pretty_print_classifier(self.model.classifier)
        )
        print("Word-level training:", self.word_level)
        if self.word_level:
            print("Word aggregation strategy:", self.word_aggregation)

        self.args = TrainingArguments(output_dir=output_dir, **self.train_kwargs)

    def compute_seqeval_metrics(self, eval_preds):
        logits, labels = eval_preds
        probs = torch.sigmoid(torch.tensor(logits))
        preds = (probs > 0.5).int().numpy()

        # we only consider the non-ambiguous labels
        idcs = np.argwhere(labels.sum(axis=-1) > 0)
        labels = np.argmax(labels[idcs[:, 0]], axis=-1)
        preds = np.argmax(preds[idcs[:, 0]], axis=-1)

        # Access the id2label mapping
        id2label = self.id2label  # Dictionary mapping IDs to labels

        # Remove ignored index (special tokens) and convert to label names
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]

        try:
            true_predictions = [
                [id2label[p] for (p, l) in zip(preds, label) if l != -100]
                for preds, label in zip(preds, labels)
            ]

            all_metrics = metric.compute(
                predictions=true_predictions, references=true_labels
            )
            return all_metrics
        except Exception as e:
            print(
                f"Seqeval metrics failed: {e}. \n True labels sample: {true_labels[0]} \n Predictions sample: {true_predictions[0]}"
            )
            return {}

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        probs = torch.sigmoid(torch.tensor(logits))
        preds = (probs > 0.5).int().numpy()
        labels = labels.reshape(-1, labels.shape[-1])
        preds = preds.reshape(-1, preds.shape[-1])

        print("Labels shape:", labels.shape)
        print("Preds shape:", preds.shape)

        # Exclude padded tokens
        # mask = (labels.sum(axis=1) != -100 * labels.shape[1])
        mask = ~np.all(labels == -100, axis=1)
        probs_flat = probs.reshape(-1, probs.shape[-1]).numpy()
        probs_masked = probs_flat[mask]

        labels = labels[mask]
        preds = preds[mask]

        # DEBUG OUTPUT
        print("\n=== O-class debugging ===")
        print(f"Mean probability for O (column 0): {probs_masked[:, 0].mean():.4f}")
        print(f"O predictions sum (how many 1s): {preds[:, 0].sum()}")
        print(f"O labels sum (how many 1s expected): {labels[:, 0].sum()}")
        print(f"All-zeros predictions count: {(preds.sum(axis=1) == 0).sum()}")
        print("========================\n")

        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        precision_micro = precision_score(
            labels, preds, average="micro", zero_division=0
        )
        recall_micro = recall_score(labels, preds, average="micro", zero_division=0)
        f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
        roc_auc_micro = roc_auc_score(labels, preds, average="micro", multi_class="ovr")

        precision_macro = precision_score(
            labels, preds, average="macro", zero_division=0
        )
        recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        roc_auc_macro = roc_auc_score(labels, preds, average="macro", multi_class="ovr")

        res_dict = {
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "rauc_micro": roc_auc_micro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "rauc_macro": roc_auc_macro,
        }

        precision_all = precision_score(labels, preds, average=None, zero_division=0)
        recall_all = recall_score(labels, preds, average=None, zero_division=0)
        f1_all = f1_score(labels, preds, average=None, zero_division=0)
        roc_auc_all = roc_auc_score(labels, preds, average=None, multi_class="ovr")

        precision_dict = defaultdict(float)
        recall_dict = defaultdict(float)
        f1_dict = defaultdict(float)
        roc_auc_dict = defaultdict(float)

        for k, v in self.id2label.items():
            precision_dict[f"precision_{v}"] = precision_all[k]
            recall_dict[f"recall_{v}"] = recall_all[k]
            f1_dict[f"f1_{v}"] = f1_all[k]
            roc_auc_dict[f"roc_auc_{v}"] = roc_auc_all[k]

        res_dict.update(precision_dict)
        res_dict.update(recall_dict)
        res_dict.update(f1_dict)
        res_dict.update(roc_auc_dict)

        # ADD metrics from seqeval
        # seq_eval = {f'SEQ_{k}':v for k,v in self.compute_seqeval_metrics(eval_preds).items()}
        # res_dict.update(seq_eval)
        return res_dict

    def train(
        self,
        train_data: List[Dict],
        test_data: List[Dict],
        eval_data: List[Dict],
        profile: bool = False,
    ):
        if self.custom_model:

            def model_init():
                cfg = copy.deepcopy(self.or_config)
                cfg.name_or_path = self.model_name_or_path
                # Ensure backbone_model_name is set (critical for loading saved models)
                cfg.backbone_model_name = self.model_name_or_path
                # base_model = AutoModel.from_pretrained(self.model_name_or_path, token=self.hf_token)
                return MultiLabelTokenClassificationModelCustom(
                    config=cfg,
                    freeze_backbone=self.custom_kwargs.get("freeze_backbone", False),
                    classifier_hidden_layers=self.custom_kwargs.get(
                        "classifier_hidden_layers", None
                    ),
                    classifier_dropout=self.custom_kwargs.get(
                        "classifier_dropout", 0.1
                    ),
                )
        else:

            def model_init():
                cfg = copy.deepcopy(self.or_config)
                cfg.name_or_path = self.model_name_or_path
                cfg.backbone_model_name = self.model_name_or_path
                return MultiLabelTokenClassificationModelHF.from_config(config=cfg)

        if len(test_data) > 0:
            _eval_data = test_data
        else:
            _eval_data = eval_data

        # Select trainer based on word_level setting
        if self.word_level:
            trainer = MultiLabelTrainerWordLevel(
                args=self.args,
                class_weights=self.class_weights,
                word_aggregation=self.word_aggregation,
                train_dataset=train_data,
                eval_dataset=_eval_data,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
                processing_class=self.tokenizer,
                model_init=model_init,
            )
        else:
            trainer = MultiLabelTrainer(
                args=self.args,
                class_weights=self.class_weights,
                train_dataset=train_data,
                eval_dataset=_eval_data,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
                processing_class=self.tokenizer,
                model_init=model_init,
                id2label=self.id2label,
            )
        trainer.add_callback(WhoAmI())

        if profile:
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
                record_shapes=False,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                trainer.train()
        else:
            trainer.train()

        # TODO: if there is a test set and evaluation set, evaluate on the eval set
        metrics = trainer.evaluate(eval_dataset=eval_data)
        # Add training arguments to model config before saving
        training_config = {
            "training_arguments": {**self.train_kwargs, "output_dir": self.output_dir},
            "model_arguments": {
                "max_length": self.tokenizer.model_max_length,
                "freeze_backbone": getattr(self.model.config, "freeze_backbone", False),
                "classifier_dropout": getattr(
                    self.model.config, "classifier_dropout", 0.1
                ),
            },
        }

        # Update model config with training arguments
        for key, value in training_config.items():
            setattr(self.model.config, key, value)

        self.model = self.model.to(dtype=torch.bfloat16)
        try:
            if self.custom_model:
                # Copy the modeling.py file to output_dir for trust_remote_code compatibility

                current_dir = os.path.dirname(os.path.abspath(__file__))
                modeling_src = os.path.join(current_dir, "modeling.py")
                modeling_dst = os.path.join(self.output_dir, "modeling.py")

                if os.path.exists(modeling_src):
                    # Ensure output directory exists
                    os.makedirs(self.output_dir, exist_ok=True)
                    shutil.copyfile(modeling_src, modeling_dst)
                    print(f"Successfully copied modeling.py to {modeling_dst}")

                    # Verify the copy was successful
                    if not os.path.exists(modeling_dst):
                        raise ValueError(
                            f"Failed to copy modeling.py to {modeling_dst}"
                        )
                else:
                    raise ValueError(
                        f"modeling.py not found at {modeling_src}. This file is required for custom models with trust_remote_code=True."
                    )

                # Add additional metadata to config for loading
                self.model.config.custom_model_type = (
                    "MultiLabelTokenClassificationModelCustom"
                )
                self.model.config.requires_trust_remote_code = True

            trainer.save_model(self.output_dir)
            print(f"Computed metrics: {metrics}")
            trainer.save_metrics("eval", metrics=metrics)
            self.tokenizer.save_pretrained(self.output_dir)

            if self.custom_model:
                print(f"Custom model saved successfully! To load this model, use:")
                print(
                    f"AutoModelForTokenClassification.from_pretrained('{self.output_dir}', trust_remote_code=True)"
                )

            checkpoint_dirs = glob.glob(os.path.join(self.output_dir, "checkpoint-*"))
            for checkpoint_dir in checkpoint_dirs:
                trainer_state_src = os.path.join(checkpoint_dir, "trainer_state.json")
                trainer_state_dst = os.path.join(self.output_dir, "trainer_state.json")

                # Move trainer_state.json if it exists
                if os.path.exists(trainer_state_src):
                    shutil.move(trainer_state_src, trainer_state_dst)
                    print(f"Moved trainer_state.json to {trainer_state_dst}")

                # Remove only real checkpoint directories
                checkpoint_dir_resolved = Path(checkpoint_dir).resolve()
                if checkpoint_dir_resolved.name.startswith("checkpoint-"):
                    shutil.rmtree(checkpoint_dir)
                    print(f"Removed checkpoint directory: {checkpoint_dir}")
                else:
                    print(f"Skipped non-checkpoint directory: {checkpoint_dir}")

        except Exception as e:
            raise ValueError(f"Failed to save model and metrics: {str(e)}")
        torch.cuda.empty_cache()
        return True
