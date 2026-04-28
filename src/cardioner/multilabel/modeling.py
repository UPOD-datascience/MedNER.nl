import inspect
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    DebertaV2Config,
    PreTrainedModel,
    RobertaConfig,
)

try:
    from transformers import EuroBertModel
except ImportError:
    EuroBertModel = None
from transformers.modeling_outputs import TokenClassifierOutput


class MultiLabelTokenClassificationModelCustom(PreTrainedModel):
    """
    Custom multi-label token classification model with configurable classifier head.
    This model can be loaded with trust_remote_code=True for HuggingFace Hub compatibility.
    """

    # Supports BERT, RoBERTa, and DeBERTa(V2) models

    # config_class = RobertaConfig

    def __init__(
        self,
        config,
        base_model=None,
        freeze_backbone=False,
        classifier_hidden_layers=None,
        classifier_dropout=0.1,
    ):
        super().__init__(config)
        self.__class__.config_class = config.__class__
        self.config = config
        self.num_labels = config.num_labels

        # READ FROM CONFIG (critical)
        freeze_backbone = getattr(config, "freeze_backbone", False)
        classifier_hidden_layers = getattr(config, "classifier_hidden_layers", None)
        classifier_dropout = getattr(config, "classifier_dropout", 0.1)

        # If base_model is not provided, load it from config
        # modeling.py
        if base_model is None:
            # IMPORTANT: Use backbone_model_name (the original pretrained model) for loading,
            # NOT name_or_path which gets updated to the checkpoint path during saving/loading.
            backbone_name = getattr(config, "backbone_model_name", None)
            if backbone_name is None:
                # Fallback to name_or_path for backwards compatibility
                backbone_name = getattr(config, "_name_or_path", None)
            if backbone_name is None:
                raise ValueError(
                    "config.backbone_model_name (or config.name_or_path) is required to load pretrained backbone"
                )

            # Create a clean config for the backbone (without custom attributes that might confuse AutoModel)
            backbone_config = AutoConfig.from_pretrained(
                backbone_name, trust_remote_code=True
            )
            # Copy over relevant attributes from our config
            backbone_config.hidden_dropout_prob = getattr(
                config, "hidden_dropout_prob", 0.1
            )
            if "eurobert" in backbone_name.lower() and EuroBertModel is not None:
                self.backbone = EuroBertModel(backbone_config)
            else:
                self.backbone = AutoModel.from_config(
                    backbone_config,
                    trust_remote_code=True,
                )
        else:
            self.backbone = base_model

        # Store the backbone model name for future loading
        if (
            not hasattr(config, "backbone_model_name")
            or config.backbone_model_name is None
        ):
            # If we got here with a base_model, try to get its name
            if base_model is not None and hasattr(base_model.config, "_name_or_path"):
                config.backbone_model_name = base_model.config._name_or_path
            elif hasattr(config, "_name_or_path"):
                config.backbone_model_name = config._name_or_path

        # Access custom attributes correctly
        self.lm_output_size = self.backbone.config.hidden_size
        # Store configuration for saving/loading
        self.config.freeze_backbone = freeze_backbone
        self.config.classifier_hidden_layers = classifier_hidden_layers
        self.config.classifier_dropout = classifier_dropout

        if freeze_backbone:
            # print("+" * 30, "\n\n", "Freezing backbone...", "+" * 30, "\n\n")
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        else:
            self.backbone.train(True)

        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
            if hasattr(config, "hidden_dropout_prob")
            else 0.1
        )
        self._build_classifier_head(classifier_hidden_layers, classifier_dropout)
        self.post_init()

    @classmethod
    def from_config(cls, config):
        return cls(
            config=config,
            freeze_backbone=getattr(config, "freeze_backbone", False),
            classifier_hidden_layers=getattr(config, "classifier_hidden_layers", None),
            classifier_dropout=getattr(config, "classifier_dropout", 0.1),
        )

    def _set_backbone_model_name(self, name: str):
        """Explicitly set the backbone model name in config (call before saving)."""
        self.config.backbone_model_name = name

    def _build_classifier_head(self, hidden_layers, dropout_rate):
        """
        Build a flexible classifier head with configurable hidden layers and dropout.

        Args:
            hidden_layers: Tuple of integers representing the number of neurons in each hidden layer.
                          None or empty tuple means a simple linear layer.
            dropout_rate: Dropout probability between layers
        """
        layers = []
        input_size = self.lm_output_size

        # If hidden_layers is None or empty, just create a simple linear layer
        if not hidden_layers:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(input_size, self.num_labels)
            )
            return

        # Build MLP with specified hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        # Final classification layer
        layers.append(nn.Linear(input_size, self.num_labels))

        # Create sequential model
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        forward_kwargs = {
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "head_mask": head_mask,
        }

        # Remove head_mask for DeBERTa models as they don't support it
        # Filter by the backbone's actual signature and skip Nones
        sig = inspect.signature(
            getattr(self.backbone, "forward", self.backbone.__call__)
        )
        allowed = sig.parameters.keys()
        forward_kwargs = {
            k: v for k, v in forward_kwargs.items() if k in allowed and v is not None
        }

        outputs = self.backbone(input_ids, **forward_kwargs)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Compute multi-label loss
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # Create mask for valid labels (not -100)
            mask = (labels != -100).float()
            # Replace -100 with 0 for loss computation
            labels_masked = torch.where(
                labels == -100, torch.zeros_like(labels), labels
            )
            # Compute loss only on valid positions
            loss_tensor = loss_fct(logits, labels_masked.float())
            loss = (loss_tensor * mask).sum() / mask.sum()

            # loss_fct = nn.BCEWithLogitsLoss(reduction="none")
            # # labels: [B, T, C] with -100 for masked positions
            # valid = (labels != -100).float()
            # labels = torch.where(labels == -100, torch.zeros_like(labels), labels).float()
            # per_elem = loss_fct(logits, labels)          # [B, T, C]
            # loss = (per_elem * valid).sum() / valid.sum().clamp_min(1.0)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    #     config = kwargs.pop('config', None)
    #     if config is None:
    #         config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

    #     # Build instance from config (no weights)
    #     _ = cls(
    #         config=config,
    #         freeze_backbone=getattr(config, 'freeze_backbone', False),
    #         classifier_hidden_layers=getattr(config, 'classifier_hidden_layers', None),
    #         classifier_dropout=getattr(config, 'classifier_dropout', 0.1),
    #     )

    #     # Now let HF load the actual weights (safetensors/shards/Hub/local, etc.)
    #     return super(MultiLabelTokenClassificationModelCustom, cls).from_pretrained(
    #         pretrained_model_name_or_path,
    #         *model_args,
    #         config=config,
    #         **kwargs
    #     )

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    #     """Override from_pretrained to handle custom model loading"""
    #     config = kwargs.pop('config', None)
    #     if config is None:
    #         from transformers import AutoConfig
    #         config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

    #     # Extract custom parameters from config if they exist
    #     freeze_backbone = getattr(config, 'freeze_backbone', False)
    #     classifier_hidden_layers = getattr(config, 'classifier_hidden_layers', None)
    #     classifier_dropout = getattr(config, 'classifier_dropout', 0.1)

    #     model = cls(
    #         config=config,
    #         freeze_backbone=freeze_backbone,
    #         classifier_hidden_layers=classifier_hidden_layers,
    #         classifier_dropout=classifier_dropout
    #     )

    #     # Load state dict if available
    #     try:
    #         state_dict = torch.load(
    #             f"{pretrained_model_name_or_path}/model.safetensors",
    #             map_location="cpu"
    #         )
    #         model.load_state_dict(state_dict)
    #     except Exception as e:
    #         # If loading fails, the model will be initialized with random weights
    #         print(f"Warning: Could not load pre-trained weights. Using randomly initialized model: {e}")

    #     return model


def load_custom_cardioner_model(model_path: str, device: str = "auto"):
    """
    Utility function to easily load a custom CardioNER model.

    Args:
        model_path: Path to the saved model directory
        device: Device to load model on ("auto", "cpu", "cuda", etc.)

    Returns:
        tuple: (model, tokenizer, config)
    """
    # Validate model directory
    import os

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    required_files = ["config.json", "modeling.py", "pytorch_model.bin"]
    missing_files = [
        f for f in required_files if not os.path.exists(os.path.join(model_path, f))
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {model_path}: {missing_files}"
        )

    print(f"Loading custom CardioNER model from: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model with trust_remote_code=True
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, trust_remote_code=True, use_safetensors=True
    )

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    print(f"Model loaded successfully on {device}")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of labels: {model.num_labels}")

    return model, tokenizer, model.config


def validate_custom_model_directory(model_path: str) -> dict:
    """
    Validate that a model directory contains all necessary files for custom model loading.

    Args:
        model_path: Path to the model directory

    Returns:
        dict: Validation results with status and details
    """
    import json
    import os

    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "files_found": [],
        "model_info": {},
    }

    # Required files
    required_files = {
        "config.json": "Model configuration",
        "modeling.py": "Custom model class definition",
        "pytorch_model.bin": "Model weights",
    }

    # Optional files
    optional_files = {
        "tokenizer.json": "Tokenizer vocabulary",
        "tokenizer_config.json": "Tokenizer configuration",
        "training_args.json": "Training arguments",
    }

    # Check required files
    for filename, description in required_files.items():
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            validation_results["files_found"].append(f"{filename} ({description})")
        else:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Missing required file: {filename} - {description}"
            )

    # Check optional files
    for filename, description in optional_files.items():
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            validation_results["files_found"].append(f"{filename} ({description})")
        else:
            validation_results["warnings"].append(
                f"Missing optional file: {filename} - {description}"
            )

    # Parse config if available
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            validation_results["model_info"]["num_labels"] = config.get(
                "num_labels", "Unknown"
            )
            validation_results["model_info"]["model_type"] = config.get(
                "model_type", "Unknown"
            )
            validation_results["model_info"]["has_auto_map"] = "auto_map" in config
            validation_results["model_info"]["classifier_hidden_layers"] = config.get(
                "classifier_hidden_layers", None
            )
            validation_results["model_info"]["freeze_backbone"] = config.get(
                "freeze_backbone", None
            )

            if not config.get("auto_map"):
                validation_results["warnings"].append(
                    "No auto_map found in config - may not load correctly with trust_remote_code=True"
                )

        except json.JSONDecodeError as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Invalid config.json: {str(e)}")

    # Check modeling.py content
    modeling_path = os.path.join(model_path, "modeling.py")
    if os.path.exists(modeling_path):
        try:
            with open(modeling_path, "r") as f:
                content = f.read()

            if "MultiLabelTokenClassificationModelCustom" not in content:
                validation_results["valid"] = False
                validation_results["errors"].append(
                    "modeling.py does not contain MultiLabelTokenClassificationModelCustom class"
                )

        except Exception as e:
            validation_results["warnings"].append(
                f"Could not read modeling.py: {str(e)}"
            )

    return validation_results
