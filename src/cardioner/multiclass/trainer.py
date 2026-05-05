import glob
import json
import os
import shutil
from os import environ
from pathlib import Path

import spacy

# Load a spaCy model for tokenization
environ["WANDB_MODE"] = "disabled"
environ["WANDB_DISABLED"] = "true"

from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torchcrf import CRF
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedModel,
    RobertaForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import logging

from cardioner.utils import pretty_print_classifier

# Import custom model classes
try:
    from .modeling import (
        MultiHeadConfig,
        MultiHeadCRFConfig,
        TokenClassificationModel,
        TokenClassificationModelCRF,
        TokenClassificationModelMultiHead,
        TokenClassificationModelMultiHeadCRF,
    )
except ImportError as e_abs:
    try:
        from modeling import (
            MultiHeadConfig,
            MultiHeadCRFConfig,
            TokenClassificationModel,
            TokenClassificationModelCRF,
            TokenClassificationModelMultiHead,
            TokenClassificationModelMultiHeadCRF,
        )
    except ImportError as e_rel:
        print(
            "Warning: Could not import custom model classes from modeling.py. "
            f"Absolute import error: {e_abs}. Relative import error: {e_rel}"
        )
        TokenClassificationModelCRF = None
        TokenClassificationModel = None
        TokenClassificationModelMultiHeadCRF = None
        TokenClassificationModelMultiHead = None
        MultiHeadCRFConfig = None
        MultiHeadConfig = None

logging.set_verbosity_debug()

import evaluate
from torch import nn

metric = evaluate.load("seqeval")


class CustomDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    def __call__(self, features, *args, **kwargs):
        # Call the superclass method to process the batch
        #
        for feature in features:
            if "labels" in feature:
                lbls = feature["labels"]
                # If lbls is one-hot encoded (e.g. [seq_length, num_labels] per example),
                # first convert to a tensor and then argmax:
                if torch.Tensor(lbls).ndim == 2:
                    lbls_tensor = torch.tensor(lbls, dtype=torch.float)
                    lbls = lbls_tensor.argmax(dim=-1).tolist()
                    lbls = [
                        label if label != -100 else self.label_pad_token_id
                        for label in lbls
                    ]
                    if len(lbls) < len(feature["input_ids"]):
                        lbls = lbls + [self.label_pad_token_id] * (
                            len(feature["input_ids"]) - len(lbls)
                        )
                    # Now lbls is a simple list of integers
                else:
                    lbls = [
                        label if label != -100 else self.label_pad_token_id
                        for label in lbls
                    ]
                    lbls = lbls + [self.label_pad_token_id] * (
                        len(feature["input_ids"]) - len(lbls)
                    )

                if (-100 in lbls) & (self.label_pad_token_id != -100):
                    print("Warning: -100 found in labels after replacement!")

                feature["labels"] = lbls

        # Now call the superclass, which will handle converting everything to tensors
        batch = super().__call__(features)
        return batch


class ModelTrainer:
    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        tokenizer=None,
        model: str = "CLTL/MedRoBERTa.nl",
        use_crf: bool = False,
        batch_size: int = 48,
        max_length: int = 514,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.001,
        num_train_epochs: int = 3,
        output_dir: str = "../../output",
        hf_token: str = None,
        freeze_backbone: bool = False,
        gradient_accumulation_steps: int = 1,
        classifier_hidden_layers: tuple | None = None,
        classifier_dropout: float = 0.1,
        class_weights: List[float] | None = None,
    ):
        self.label2id = label2id
        self.id2label = id2label
        self.output_dir = output_dir

        self.train_kwargs = {
            "run_name": "CardioNER",
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "weight_decay": weight_decay,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "report_to": "tensorboard",
            "use_cpu": False,
            "fp16": True,
            "load_best_model_at_end": True,
            "greater_is_better": True,
            "metric_for_best_model": "overall_f1",
            "logging_dir": f"{output_dir}/logs",
            "logging_strategy": "steps",
            "logging_steps": 256,
        }
        self.crf = use_crf

        if use_crf:
            # Use O label (0) for masked positions - no separate PADDING class needed
            # Following ieeta-pt approach: special tokens get O label
            self.pad_token_id = 0  # O label
        else:
            self.pad_token_id = -100
            # num_labels = len(label2id)  # unused variable

        if tokenizer is None:
            print("LOADING TOKENIZER")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model,
                add_prefix_space=True,
                model_max_length=max_length,
                padding=False,
                truncation=True,
                token=hf_token,
            )
        else:
            self.tokenizer = tokenizer

        self.tokenizer.model_max_length = max_length
        self.data_collator = CustomDataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            max_length=max_length,
            padding=True,
            label_pad_token_id=self.pad_token_id,
        )

        or_config = AutoConfig.from_pretrained(
            model, hf_token=hf_token, return_unused_kwargs=False, trust_remote_code=True
        )
        or_config.num_labels = len(self.label2id)
        or_config.id2label = self.id2label
        or_config.label2id = self.label2id
        or_config.hidden_dropout_prob = 0.1
        or_config.classifier_hidden_layers = classifier_hidden_layers
        or_config.classifier_dropout = classifier_dropout
        or_config.class_weights = class_weights
        or_config.output_hidden_states = False
        or_config.output_attentions = False
        # Store the original backbone model name for proper loading later
        # This is critical because name_or_path gets overwritten during save/load
        or_config.backbone_model_name = model

        self.classifier_hidden_layers = classifier_hidden_layers
        self.classifier_dropout = classifier_dropout

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

        if use_crf:
            if TokenClassificationModelCRF is None:
                raise ImportError(
                    "TokenClassificationModelCRF could not be imported. Please ensure modeling.py is available."
                )
            print("USING CRF:", self.crf)
            base_model = RobertaForTokenClassification.from_pretrained(
                model,
                config=or_config,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            )
            self.model = TokenClassificationModelCRF(
                or_config,
                base_model,
                freeze_backbone,
                classifier_hidden_layers,
                classifier_dropout,
            )

            # Set up auto_map for trust_remote_code loading
            self.model.config.auto_map = {
                "AutoModelForTokenClassification": "modeling.TokenClassificationModelCRF"
            }
        else:
            if TokenClassificationModel is None:
                raise ImportError(
                    "TokenClassificationModel could not be imported. Please ensure modeling.py is available."
                )
            try:
                base_model = AutoModel.from_pretrained(
                    model,
                    config=or_config,
                    add_pooling_layer=False,
                    trust_remote_code=True,
                    use_safetensors=True,
                )
            except TypeError:
                base_model = AutoModel.from_pretrained(
                    model,
                    config=or_config,
                    trust_remote_code=True,
                    use_safetensors=True,
                )
            self.model = TokenClassificationModel(or_config, base_model=base_model)

            # Set up auto_map for trust_remote_code loading
            self.model.config.auto_map = {
                "AutoModelForTokenClassification": "modeling.TokenClassificationModel"
            }

        # Mark that this is a custom model requiring trust_remote_code
        self.model.config.custom_model_type = (
            "TokenClassificationModelCRF" if use_crf else "TokenClassificationModel"
        )
        self.model.config.requires_trust_remote_code = True

        # optionally freeze the backbone
        if freeze_backbone:
            for p in self.model.roberta.parameters():
                p.requires_grad = False
            self.model.roberta.eval()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Model:", type(self.model))
        print("Device:", self.device)
        print("Tokenizer max length:", self.tokenizer.model_max_length)
        print(
            "Model max position embeddings:", self.model.config.max_position_embeddings
        )
        print("Number of labels:", len(self.label2id))
        print("Labels:", self.label2id)
        print("id2label:", self.id2label)
        print("Model config:", self.model.config)
        print(
            "Classifier architecture:", pretty_print_classifier(self.model.classifier)
        )

        self.args = TrainingArguments(output_dir=output_dir, **self.train_kwargs)

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        if self.crf:
            crf_device = next(self.model.crf.parameters()).device
            emissions_torch = torch.from_numpy(logits).float().to(crf_device)
            labels_torch = torch.from_numpy(labels).to(crf_device)
            mask = labels_torch != -100
            predictions = self.model.crf.decode(emissions=emissions_torch, mask=mask)
        else:
            predictions = np.argmax(logits, -1)

        # === DEBUG: Check prediction distribution ===
        pred_counts = {label_id: 0 for label_id in self.id2label}
        true_counts = {label_id: 0 for label_id in self.id2label}

        for pred_seq, label_seq in zip(predictions, labels):
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                if int(p) in pred_counts:
                    pred_counts[int(p)] += 1
                if int(l) in true_counts:
                    true_counts[int(l)] += 1

        print("\n=== Evaluation prediction distribution ===")
        for label_id, label_name in self.id2label.items():
            pred_count = pred_counts.get(label_id, 0)
            true_count = true_counts.get(label_id, 0)
            print(f"  {label_name}: pred={pred_count}, true={true_count}")
        print("=" * 50 + "\n")
        # === END DEBUG ===
        #
        # Access the id2label mapping
        id2label = self.id2label  # Dictionary mapping IDs to labels

        # Convert to label names for seqeval
        # Following ieeta-pt approach: all positions have valid O/B/I labels
        true_labels = []
        true_predictions = []

        for prediction, label in zip(predictions, labels):
            seq_labels = []
            seq_preds = []
            for p, l in zip(prediction, label):
                if l == -100:
                    continue
                label_name = id2label.get(int(l), "O")
                pred_name = id2label.get(int(p), "O")

                seq_labels.append(label_name)
                seq_preds.append(pred_name)

            true_labels.append(seq_labels)
            true_predictions.append(seq_preds)

        all_metrics = metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return all_metrics

    def save_model(self, output_dir=None):
        """Custom method to save both the model architecture and state dict properly"""
        save_dir = output_dir or self.output_dir

        # Copy the modeling.py file to output_dir for trust_remote_code compatibility
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        modeling_src = os.path.join(current_dir, "modeling.py")
        modeling_dst = os.path.join(save_dir, "modeling.py")

        # Ensure output directory exists
        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(modeling_src):
            shutil.copyfile(modeling_src, modeling_dst)
            print(f"Successfully copied modeling.py to {modeling_dst}")

            # Verify the copy was successful
            if not os.path.exists(modeling_dst):
                raise ValueError(f"Failed to copy modeling.py to {modeling_dst}")
        else:
            raise ValueError(
                f"modeling.py not found at {modeling_src}. This file is required for custom models with trust_remote_code=True."
            )

        # Set model configuration
        self.model.config.architectures = [self.model.__class__.__name__]
        self.model.config.classifier_hidden_layers = self.classifier_hidden_layers
        self.model.config.classifier_dropout = self.classifier_dropout
        # Convert tensor to list for JSON serialization
        self.model.config.class_weights = (
            self.class_weights.tolist() if self.class_weights is not None else None
        )

        # Save model and tokenizer
        self.model.save_pretrained(save_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

        print(f"Custom model saved successfully! To load this model, use:")
        print(
            f"AutoModelForTokenClassification.from_pretrained('{save_dir}', trust_remote_code=True)"
        )
        print(f"Model saved to {save_dir}")

    def train(
        self,
        train_data: List[Dict],
        test_data: List[Dict],
        eval_data: List[Dict],
        profile: bool = False,
    ):
        if len(test_data) > 0:
            _eval_data = test_data
        else:
            _eval_data = eval_data
        # Print initial class distribution from training data
        print("\n=== Initial class distribution in training data ===")
        initial_counts = torch.zeros(len(self.label2id), dtype=torch.long)
        total_initial_tokens = 0
        for sample in train_data:
            if "labels" in sample:
                labels = sample["labels"]
                if isinstance(labels, list):
                    labels = torch.tensor(labels)
                elif not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                for label_id in range(len(self.label2id)):
                    count = (labels == label_id).sum().item()
                    initial_counts[label_id] += count
                valid_count = (labels != -100).sum().item()
                total_initial_tokens += valid_count

        for i, count in enumerate(initial_counts.tolist()):
            label_name = self.id2label.get(i, f"class_{i}")
            pct = 100 * count / total_initial_tokens if total_initial_tokens > 0 else 0
            print(f"  {label_name}: {count} ({pct:.2f}%)")
        print(f"Total tokens: {total_initial_tokens}")
        print("=" * 50 + "\n")

        # Custom save function to properly handle non-PreTrainedModel models
        class CustomTrainer(Trainer):
            def __init__(
                self, parent_trainer, class_weights=None, max_weight=50.0, **kwargs
            ):
                super().__init__(**kwargs)
                self.parent_trainer = parent_trainer
                self.class_weights = class_weights
                if class_weights is not None:
                    # Cap weights to prevent numerical instability
                    capped_weights = [min(w, max_weight) for w in class_weights]
                    self.loss_fct = nn.CrossEntropyLoss(
                        weight=torch.tensor(capped_weights, dtype=torch.float).to(
                            self.args.device
                        ),
                        ignore_index=-100,
                    )
                else:
                    self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            def compute_loss(
                self,
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=None,
                **kwargs,
            ):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits

                # logits: (batch, seq_len, num_labels)
                # labels: (batch, seq_len)
                loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

                return (loss, outputs) if return_outputs else loss

            def prediction_step(
                self, model, inputs, prediction_loss_only, ignore_keys=None
            ):
                labels = inputs.get("labels")
                attention_mask = inputs.get("attention_mask")

                with torch.no_grad():
                    outputs = model(**inputs)

                if prediction_loss_only:
                    loss = outputs.loss if hasattr(outputs, "loss") else None
                    return (loss, None, None)

                logits = outputs.logits
                if labels is not None and attention_mask is not None:
                    labels = labels.clone()
                    labels[attention_mask == 0] = -100

                return (None, logits, labels)

            def save_model(self, output_dir=None, _internal_call=False):
                self.parent_trainer.save_model(output_dir)

        trainer = CustomTrainer(
            parent_trainer=self,
            class_weights=self.class_weights,  # pass the weights
            model=self.model,
            args=self.args,
            train_dataset=train_data,
            eval_dataset=_eval_data,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.tokenizer,
        )

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
        self.model = self.model.to(dtype=torch.bfloat16)
        save_dir = self.output_dir
        try:
            trainer.save_model(save_dir)
            trainer.save_metrics(save_dir, metrics=metrics)
        except Exception as e:
            try:
                save_dir = "output"
                trainer.save_model(save_dir)
                trainer.save_metrics(save_dir, metrics=metrics)
                print(f"Saved to fallback directory 'output' due to error: {str(e)}")
            except Exception as e2:
                raise ValueError(f"Failed to save model and metrics: {str(e2)}")

        checkpoint_dirs = glob.glob(os.path.join(save_dir, "checkpoint-*"))
        for checkpoint_dir in checkpoint_dirs:
            trainer_state_src = os.path.join(checkpoint_dir, "trainer_state.json")
            trainer_state_dst = os.path.join(save_dir, "trainer_state.json")

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

        torch.cuda.empty_cache()
        return True


class MultiHeadDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    """
    Data collator for Multi-Head CRF that handles labels as dictionaries
    mapping entity types to their respective label sequences.
    """

    def __init__(
        self,
        tokenizer,
        entity_types: List[str],
        max_length: int = 512,
        padding: str = "longest",
        label_pad_token_id: int = 0,  # Use O label for masked positions (CRF can't handle -100)
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            label_pad_token_id=label_pad_token_id,
        )
        self.entity_types = entity_types
        self.ignore_index = -100  # Original ignore index from tokenizer alignment

    def __call__(self, features, *args, **kwargs):
        # Separate out the multi-head labels before standard collation
        # Following ieeta-pt approach: use O label (0) for all special/padding positions
        # No separate mask tracking needed - CRF handles all positions uniformly
        entity_labels = {ent: [] for ent in self.entity_types}

        for feature in features:
            if "labels" in feature and isinstance(feature["labels"], dict):
                for ent in self.entity_types:
                    if ent in feature["labels"]:
                        lbls = feature["labels"][ent]
                        # Handle one-hot encoded labels
                        if torch.Tensor(lbls).ndim == 2:
                            lbls_tensor = torch.tensor(lbls, dtype=torch.float)
                            lbls = lbls_tensor.argmax(dim=-1).tolist()

                        # Replace -100 (ignore index) with O label (0)
                        # This follows ieeta-pt approach: special tokens get O label
                        lbls = [
                            l if l != self.ignore_index else self.label_pad_token_id
                            for l in lbls
                        ]

                        # Pad to input length with O label
                        if len(lbls) < len(feature["input_ids"]):
                            pad_len = len(feature["input_ids"]) - len(lbls)
                            lbls = lbls + [self.label_pad_token_id] * pad_len

                        entity_labels[ent].append(lbls)
                    else:
                        # Default to O label if entity type not present
                        entity_labels[ent].append(
                            [self.label_pad_token_id] * len(feature["input_ids"])
                        )

                # Remove the dict labels so standard collation can proceed
                del feature["labels"]

        # Standard collation for non-label fields
        batch = super().__call__(features)

        # Add back the entity-specific labels as tensors
        batch["labels"] = {
            ent: torch.tensor(entity_labels[ent], dtype=torch.long)
            for ent in self.entity_types
        }

        return batch


class MultiHeadCRFTrainer:
    """
    Trainer for Multi-Head CRF models that handle multiple entity types simultaneously.

    Each entity type gets its own CRF head, allowing for overlapping entities
    and entity-type-specific transition patterns.
    """

    def __init__(
        self,
        entity_types: List[str],
        label2id: Dict[str, int],  # BIO labels: {"O": 0, "B": 1, "I": 2}
        id2label: Dict[int, str],
        tokenizer=None,
        model: str = "CLTL/MedRoBERTa.nl",
        batch_size: int = 48,
        max_length: int = 514,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.001,
        num_train_epochs: int = 3,
        output_dir: str = "../../output",
        hf_token: str = None,
        freeze_backbone: bool = False,
        num_frozen_encoders: int = 0,
        gradient_accumulation_steps: int = 1,
        number_of_layers_per_head: int = 1,
        classifier_dropout: float = 0.1,
        crf_reduction: str = "mean",
    ):
        self.entity_types = sorted(entity_types)  # Sort for consistency
        self.label2id = label2id
        self.id2label = id2label
        self.output_dir = output_dir

        # Use O label (0) for masked positions - no separate PADDING class needed
        # The attention_mask excludes these positions from CRF loss and decoding
        self.label_pad_token_id = 0  # O label
        num_labels = len(label2id)  # Should be 3: O, B, I

        self.train_kwargs = {
            "run_name": "CardioNER-MultiHeadCRF",
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "weight_decay": weight_decay,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "report_to": "tensorboard",
            "use_cpu": False,
            "fp16": True,
            "load_best_model_at_end": True,
            "greater_is_better": True,
            "metric_for_best_model": "macro_f1",
            "logging_dir": f"{output_dir}/logs",
            "logging_strategy": "steps",
            "logging_steps": 256,
        }

        # Load tokenizer
        if tokenizer is None:
            print("LOADING TOKENIZER")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model,
                add_prefix_space=True,
                model_max_length=max_length,
                padding=False,
                truncation=True,
                token=hf_token,
            )
        else:
            self.tokenizer = tokenizer

        self.tokenizer.model_max_length = max_length

        # Data collator for multi-head labels
        self.data_collator = MultiHeadDataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            entity_types=self.entity_types,
            max_length=max_length,
            padding=True,
            label_pad_token_id=self.label_pad_token_id,
        )

        # Create MultiHeadCRF config
        base_config = AutoConfig.from_pretrained(
            model, token=hf_token, trust_remote_code=True
        )
        config = MultiHeadCRFConfig(
            entity_types=self.entity_types,
            number_of_layers_per_head=number_of_layers_per_head,
            crf_reduction=crf_reduction,
            freeze_backbone=freeze_backbone,
            num_frozen_encoders=num_frozen_encoders,
            classifier_dropout=classifier_dropout,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            # Copy base config attributes
            hidden_size=base_config.hidden_size,
            hidden_dropout_prob=getattr(
                base_config,
                "hidden_dropout_prob",
                getattr(base_config, "hidden_dropout", 0.1),
            ),
            vocab_size=base_config.vocab_size,
            max_position_embeddings=base_config.max_position_embeddings,
            type_vocab_size=getattr(base_config, "type_vocab_size", 1),
            num_attention_heads=base_config.num_attention_heads,
            num_hidden_layers=base_config.num_hidden_layers,
            intermediate_size=base_config.intermediate_size,
            # Store the original backbone model name for proper loading later
            backbone_model_name=model,
        )

        # Load base model and create MultiHeadCRF model
        if TokenClassificationModelMultiHeadCRF is None:
            raise ImportError(
                "TokenClassificationModelMultiHeadCRF could not be imported."
            )

        try:
            base_model = AutoModel.from_pretrained(
                model,
                config=base_config,
                token=hf_token,
                add_pooling_layer=False,
                trust_remote_code=True,
                use_safetensors=True,
            )
        except TypeError:
            base_model = AutoModel.from_pretrained(
                model,
                config=base_config,
                token=hf_token,
                trust_remote_code=True,
                use_safetensors=True,
            )
        self.model = TokenClassificationModelMultiHeadCRF(
            config, base_model, freeze_backbone
        )

        # Set up auto_map for trust_remote_code loading
        self.model.config.auto_map = {
            "AutoModel": "modeling.TokenClassificationModelMultiHeadCRF",
            "AutoModelForTokenClassification": "modeling.TokenClassificationModelMultiHeadCRF",
        }
        self.model.config.architectures = ["TokenClassificationModelMultiHeadCRF"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("=" * 50)
        print("Multi-Head CRF Model Configuration:")
        print(f"  Entity types: {self.entity_types}")
        print(f"  Number of labels per head: {num_labels}")
        print(f"  Layers per head: {number_of_layers_per_head}")
        print(f"  CRF reduction: {crf_reduction}")
        print(f"  Freeze backbone: {freeze_backbone}")
        print(f"  Device: {self.device}")
        print("=" * 50)

        self.args = TrainingArguments(output_dir=output_dir, **self.train_kwargs)

    def compute_metrics(self, eval_preds):
        """
        Compute metrics for multi-head predictions.

        For multi-head CRF, predictions are organized by entity type.
        Following ieeta-pt approach: all positions use valid labels (O for special tokens),
        so we evaluate all non-padding positions based on attention_mask.

        Args:
            eval_preds: Tuple of (predictions_dict, labels_dict)
                - predictions_dict: {entity_type: decoded_sequences}
                - labels_dict: {entity_type: label_sequences}
        """
        predictions_dict, labels_dict = eval_preds

        all_metrics = {}

        for entity_type in self.entity_types:
            if entity_type not in predictions_dict or entity_type not in labels_dict:
                continue

            predictions = predictions_dict[entity_type]
            labels = labels_dict[entity_type]

            # Convert to label names for seqeval
            # All positions have valid labels (O for special/padding tokens)
            true_labels = []
            true_predictions = []

            for pred_seq, label_seq in zip(predictions, labels):
                seq_labels = []
                seq_preds = []
                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    label_name = self.id2label.get(int(l), "O")
                    pred_name = self.id2label.get(int(p), "O")

                    seq_labels.append(label_name)
                    seq_preds.append(pred_name)

                if seq_labels:  # Only add non-empty sequences
                    true_labels.append(seq_labels)
                    true_predictions.append(seq_preds)

            # Compute metrics for this entity type
            if true_labels:
                entity_metrics = metric.compute(
                    predictions=true_predictions, references=true_labels
                )

                # Prefix metrics with entity type
                for key, value in entity_metrics.items():
                    all_metrics[f"{entity_type}_{key}"] = value

        return all_metrics

    def save_model(self, output_dir=None):
        """Save the multi-head CRF model with all necessary files."""
        import os

        save_dir = output_dir or self.output_dir

        # Ensure output directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Copy modeling.py for trust_remote_code compatibility
        current_dir = os.path.dirname(os.path.abspath(__file__))
        modeling_src = os.path.join(current_dir, "modeling.py")
        modeling_dst = os.path.join(save_dir, "modeling.py")

        if os.path.exists(modeling_src):
            shutil.copyfile(modeling_src, modeling_dst)
            print(f"Copied modeling.py to {modeling_dst}")
        else:
            raise ValueError(f"modeling.py not found at {modeling_src}")

        # Save model and tokenizer
        self.model.save_pretrained(save_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

        print(f"Multi-Head CRF model saved to {save_dir}")
        print(
            f"To load: TokenClassificationModelMultiHeadCRF.from_pretrained('{save_dir}', trust_remote_code=True)"
        )

    def train(
        self,
        train_data: List[Dict],
        test_data: List[Dict],
        eval_data: List[Dict],
        profile: bool = False,
    ):
        """
        Train the multi-head CRF model.

        Data should have 'labels' as a dict mapping entity types to label sequences.
        """
        _eval_data = test_data if len(test_data) > 0 else eval_data

        # Custom trainer that handles multi-head loss computation
        class MultiHeadCRFHFTrainer(Trainer):
            def __init__(self, parent_trainer, **kwargs):
                super().__init__(**kwargs)
                self.parent_trainer = parent_trainer

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels", None)
                outputs = model(**inputs, labels=labels)

                if labels is not None:
                    loss, logits = outputs
                else:
                    loss = None
                    logits = outputs

                return (loss, {"logits": logits}) if return_outputs else loss

            def prediction_step(
                self, model, inputs, prediction_loss_only, ignore_keys=None
            ):
                """Override to handle multi-head CRF decoding."""
                labels = inputs.pop("labels", None)
                attention_mask = inputs.get("attention_mask")

                with torch.no_grad():
                    outputs = model(**inputs, labels=labels)

                    if labels is not None:
                        loss, logits = outputs
                    else:
                        loss = None
                        logits = outputs

                if prediction_loss_only:
                    return (loss, None, None)

                # Decode using CRF for each entity type
                predictions = {}
                for entity_type in self.parent_trainer.entity_types:
                    crf = getattr(model, f"{entity_type}_crf")
                    if attention_mask is not None:
                        mask = attention_mask.bool()
                        decoded = crf.decode(logits[entity_type], mask=mask)
                    else:
                        decoded = crf.decode(logits[entity_type])
                    # Pad decoded sequences to same length for batching
                    max_len = max(len(seq) for seq in decoded)
                    padded = [seq + [0] * (max_len - len(seq)) for seq in decoded]
                    # Return as tensor to be compatible with HuggingFace Trainer
                    predictions[entity_type] = torch.tensor(padded)

                # Return labels with padding masked out
                labels_out = {}
                if labels is not None:
                    for entity_type in self.parent_trainer.entity_types:
                        labels_masked = labels[entity_type].clone()
                        if attention_mask is not None:
                            labels_masked[attention_mask == 0] = -100
                        labels_out[entity_type] = labels_masked.cpu()

                return (loss, predictions, labels_out)

            def save_model(self, output_dir=None, _internal_call=False):
                self.parent_trainer.save_model(output_dir)

        trainer = MultiHeadCRFHFTrainer(
            parent_trainer=self,
            model=self.model,
            args=self.args,
            train_dataset=train_data,
            eval_dataset=_eval_data,
            data_collator=self.data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

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

        # Evaluate and save metrics
        metrics = trainer.evaluate(eval_dataset=_eval_data)

        # Save metrics to JSON file
        metrics_path = os.path.join(self.output_dir, "eval_results.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Evaluation metrics saved to {metrics_path}")

        # Save model
        self.model = self.model.to(dtype=torch.bfloat16)
        save_dir = self.output_dir
        try:
            trainer.save_model(save_dir)
        except Exception as e:
            print(f"Error saving model: {e}")
            try:
                save_dir = "output"
                trainer.save_model(save_dir)
            except Exception as e2:
                raise ValueError(f"Failed to save model: {e2}")

        checkpoint_dirs = glob.glob(os.path.join(save_dir, "checkpoint-*"))
        for checkpoint_dir in checkpoint_dirs:
            trainer_state_src = os.path.join(checkpoint_dir, "trainer_state.json")
            trainer_state_dst = os.path.join(save_dir, "trainer_state.json")

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

        torch.cuda.empty_cache()
        return metrics


class MultiHeadTrainer:
    """
    Trainer for Multi-Head models (without CRF) that handle multiple entity types simultaneously.

    Each entity type gets its own classification head, allowing for overlapping entities.
    Uses standard CrossEntropyLoss instead of CRF, which is faster but doesn't enforce
    valid BIO sequences.
    """

    def __init__(
        self,
        entity_types: List[str],
        label2id: Dict[str, int],  # BIO labels: {"O": 0, "B": 1, "I": 2}
        id2label: Dict[int, str],
        tokenizer=None,
        model: str = "CLTL/MedRoBERTa.nl",
        batch_size: int = 48,
        max_length: int = 514,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.001,
        num_train_epochs: int = 3,
        output_dir: str = "../../output",
        hf_token: str = None,
        freeze_backbone: bool = False,
        num_frozen_encoders: int = 0,
        gradient_accumulation_steps: int = 1,
        number_of_layers_per_head: int = 1,
        classifier_dropout: float = 0.1,
        use_class_weights: bool = False,
        class_weights: Optional[Dict[str, List[float]]] = None,
    ):
        self.entity_types = sorted(entity_types)  # Sort for consistency
        self.label2id = label2id
        self.id2label = id2label
        self.output_dir = output_dir

        # Use -100 for ignored positions in CrossEntropyLoss
        self.label_pad_token_id = 0  # O label for padding
        num_labels = len(label2id)  # Should be 3: O, B, I

        self.train_kwargs = {
            "run_name": "CardioNER-MultiHead",
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "weight_decay": weight_decay,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "report_to": "tensorboard",
            "use_cpu": False,
            "fp16": True,
            "load_best_model_at_end": True,
            "greater_is_better": True,
            "metric_for_best_model": "macro_f1",
            "logging_dir": f"{output_dir}/logs",
            "logging_strategy": "steps",
            "logging_steps": 256,
        }

        # Load tokenizer
        if tokenizer is None:
            print("LOADING TOKENIZER")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model,
                add_prefix_space=True,
                model_max_length=max_length,
                padding=False,
                truncation=True,
                token=hf_token,
            )
        else:
            self.tokenizer = tokenizer

        self.tokenizer.model_max_length = max_length

        # Data collator for multi-head labels (reuse from CRF trainer)
        self.data_collator = MultiHeadDataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            entity_types=self.entity_types,
            max_length=max_length,
            padding=True,
            label_pad_token_id=self.label_pad_token_id,
        )

        # Create MultiHead config (no CRF)
        base_config = AutoConfig.from_pretrained(
            model, token=hf_token, trust_remote_code=True
        )
        config = MultiHeadConfig(
            entity_types=self.entity_types,
            number_of_layers_per_head=number_of_layers_per_head,
            freeze_backbone=freeze_backbone,
            num_frozen_encoders=num_frozen_encoders,
            classifier_dropout=classifier_dropout,
            use_class_weights=use_class_weights,
            class_weights=class_weights,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            # Copy base config attributes
            hidden_size=base_config.hidden_size,
            hidden_dropout_prob=getattr(
                base_config,
                "hidden_dropout_prob",
                getattr(base_config, "hidden_dropout", 0.1),
            ),
            vocab_size=base_config.vocab_size,
            max_position_embeddings=base_config.max_position_embeddings,
            type_vocab_size=getattr(base_config, "type_vocab_size", 1),
            num_attention_heads=base_config.num_attention_heads,
            num_hidden_layers=base_config.num_hidden_layers,
            intermediate_size=base_config.intermediate_size,
            # Store the original backbone model name for proper loading later
            backbone_model_name=model,
        )

        # Load base model and create MultiHead model
        if TokenClassificationModelMultiHead is None:
            raise ImportError(
                "TokenClassificationModelMultiHead could not be imported."
            )

        try:
            base_model = AutoModel.from_pretrained(
                model,
                config=base_config,
                token=hf_token,
                add_pooling_layer=False,
                trust_remote_code=True,
                use_safetensors=True,
            )
        except TypeError:
            base_model = AutoModel.from_pretrained(
                model,
                config=base_config,
                token=hf_token,
                trust_remote_code=True,
            )
        self.model = TokenClassificationModelMultiHead(
            config, base_model, freeze_backbone
        )

        # Set up auto_map for trust_remote_code loading
        self.model.config.auto_map = {
            "AutoModel": "modeling.TokenClassificationModelMultiHead",
            "AutoModelForTokenClassification": "modeling.TokenClassificationModelMultiHead",
        }
        self.model.config.architectures = ["TokenClassificationModelMultiHead"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("=" * 50)
        print("Multi-Head Model Configuration (no CRF):")
        print(f"  Entity types: {self.entity_types}")
        print(f"  Number of labels per head: {num_labels}")
        print(f"  Layers per head: {number_of_layers_per_head}")
        print(f"  Use class weights: {use_class_weights}")
        print(f"  Freeze backbone: {freeze_backbone}")
        print(f"  Device: {self.device}")
        print("=" * 50)

        self.args = TrainingArguments(output_dir=output_dir, **self.train_kwargs)

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for multi-head model evaluation.

        Returns metrics per entity type and averaged across all entity types.
        """
        predictions, labels = eval_pred

        # Handle both dict and tuple/list formats
        if isinstance(predictions, dict):
            pred_dict = predictions
            label_dict = labels
        else:
            # Assume sorted entity types order
            pred_dict = {ent: predictions[i] for i, ent in enumerate(self.entity_types)}
            label_dict = {ent: labels[i] for i, ent in enumerate(self.entity_types)}

        all_metrics = {}
        per_entity_f1 = []

        for entity_type in self.entity_types:
            preds = pred_dict[entity_type]
            labs = label_dict[entity_type]

            # Convert to lists and filter padding
            true_predictions = []
            true_labels = []

            for pred_seq, label_seq in zip(preds, labs):
                pred_tags = []
                label_tags = []

                for p, l in zip(pred_seq, label_seq):
                    # Skip padding positions (label == -100 or position after sequence ends)
                    if l == -100:
                        continue
                    pred_tags.append(self.id2label[int(p)])
                    label_tags.append(self.id2label[int(l)])

                if pred_tags:
                    true_predictions.append(pred_tags)
                    true_labels.append(label_tags)

            if true_predictions:
                results = metric.compute(
                    predictions=true_predictions, references=true_labels
                )

                all_metrics[f"{entity_type}_precision"] = results.get(
                    "overall_precision", 0.0
                )
                all_metrics[f"{entity_type}_recall"] = results.get(
                    "overall_recall", 0.0
                )
                all_metrics[f"{entity_type}_f1"] = results.get("overall_f1", 0.0)

                per_entity_f1.append(results.get("overall_f1", 0.0))

        # Compute macro average
        if per_entity_f1:
            all_metrics["macro_f1"] = sum(per_entity_f1) / len(per_entity_f1)

        return all_metrics

    def save_model(self, output_dir=None):
        """Save model, tokenizer, and modeling.py for trust_remote_code loading."""
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save model and config
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Copy modeling.py for trust_remote_code
        modeling_src = os.path.join(os.path.dirname(__file__), "modeling.py")
        modeling_dst = os.path.join(output_dir, "modeling.py")
        if os.path.exists(modeling_src):
            shutil.copy(modeling_src, modeling_dst)
            print(f"Copied modeling.py to {modeling_dst}")

        print(f"Model saved to {output_dir}")

    def train(
        self,
        train_data: List[Dict],
        test_data: List[Dict],
        eval_data: List[Dict],
        profile: bool = False,
    ):
        """
        Train the multi-head model.

        Data should have 'labels' as a dict mapping entity types to label sequences.
        """
        _eval_data = test_data if len(test_data) > 0 else eval_data

        # Custom trainer that handles multi-head loss computation
        class MultiHeadHFTrainer(Trainer):
            def __init__(self, parent_trainer, **kwargs):
                super().__init__(**kwargs)
                self.parent_trainer = parent_trainer

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels", None)
                outputs = model(**inputs, labels=labels)

                if labels is not None:
                    loss, logits = outputs
                else:
                    loss = None
                    logits = outputs

                return (loss, {"logits": logits}) if return_outputs else loss

            def prediction_step(
                self, model, inputs, prediction_loss_only, ignore_keys=None
            ):
                """Override to handle multi-head decoding."""
                labels = inputs.pop("labels", None)
                attention_mask = inputs.get("attention_mask")

                with torch.no_grad():
                    outputs = model(**inputs, labels=labels)

                    if labels is not None:
                        loss, logits = outputs
                    else:
                        loss = None
                        logits = outputs

                if prediction_loss_only:
                    return (loss, None, None)

                # Argmax decoding for each entity type
                predictions = {}
                for entity_type in self.parent_trainer.entity_types:
                    preds = torch.argmax(logits[entity_type], dim=-1)
                    predictions[entity_type] = preds.cpu()

                # Return labels with padding masked out
                labels_out = {}
                if labels is not None:
                    for entity_type in self.parent_trainer.entity_types:
                        labels_masked = labels[entity_type].clone()
                        if attention_mask is not None:
                            labels_masked[attention_mask == 0] = -100
                        labels_out[entity_type] = labels_masked.cpu()

                return (loss, predictions, labels_out)

            def save_model(self, output_dir=None, _internal_call=False):
                self.parent_trainer.save_model(output_dir)

        trainer = MultiHeadHFTrainer(
            parent_trainer=self,
            model=self.model,
            args=self.args,
            train_dataset=train_data,
            eval_dataset=_eval_data,
            data_collator=self.data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

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

        # Evaluate and save metrics
        metrics = trainer.evaluate(eval_dataset=_eval_data)

        # Save metrics to JSON file
        metrics_path = os.path.join(self.output_dir, "eval_results.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Evaluation metrics saved to {metrics_path}")

        # Save model
        self.model = self.model.to(dtype=torch.bfloat16)
        save_dir = self.output_dir
        try:
            trainer.save_model(save_dir)
        except Exception as e:
            print(f"Error saving model: {e}")
            try:
                save_dir = "output"
                trainer.save_model(save_dir)
            except Exception as e2:
                raise ValueError(f"Failed to save model: {e2}")

        checkpoint_dirs = glob.glob(os.path.join(save_dir, "checkpoint-*"))
        for checkpoint_dir in checkpoint_dirs:
            trainer_state_src = os.path.join(checkpoint_dir, "trainer_state.json")
            trainer_state_dst = os.path.join(save_dir, "trainer_state.json")

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

        torch.cuda.empty_cache()
        return metrics
