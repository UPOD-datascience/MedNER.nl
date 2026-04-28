from itertools import islice
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

try:
    from transformers.models.eurobert.modeling_eurobert import EuroBertModel
except Exception:
    try:
        from transformers import EuroBertModel
    except Exception:
        EuroBertModel = None
        print("COULD NOT IMPORT EUROBERT MODEL")

# Large negative number for masking impossible transitions
LARGE_NEGATIVE_NUMBER = -1e9
NUM_PER_LAYER = 16


def _build_backbone_from_config(config):
    """
    Build a backbone model structure from config only.
    Never call from_pretrained() here; outer model loading will restore weights.
    """
    from transformers import AutoConfig, AutoModel

    backbone_name = getattr(config, "backbone_model_name", None)
    if backbone_name is None:
        backbone_name = getattr(config, "_name_or_path", None)

    if backbone_name is None:
        raise ValueError(
            "config.backbone_model_name (or config._name_or_path) is required to load backbone"
        )

    backbone_config = AutoConfig.from_pretrained(
        backbone_name,
        trust_remote_code=True,
    )

    if hasattr(config, "hidden_dropout_prob"):
        backbone_config.hidden_dropout_prob = getattr(
            config, "hidden_dropout_prob", 0.1
        )

    if hasattr(config, "num_labels"):
        backbone_config.num_labels = getattr(config, "num_labels")

    if "eurobert" in backbone_name.lower() and EuroBertModel is not None:
        backbone = EuroBertModel(backbone_config)
    else:
        backbone = AutoModel.from_config(
            backbone_config,
            trust_remote_code=True,
        )

    if getattr(config, "backbone_model_name", None) is None:
        config.backbone_model_name = backbone_name

    return backbone, backbone_name


class MultiHeadCRFConfig(PretrainedConfig):
    """
    Configuration class for Multi-Head CRF models.
    """

    model_type = "multihead-crf-tagger"

    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        number_of_layers_per_head: int = 1,
        crf_reduction: str = "mean",
        freeze_backbone: bool = False,
        num_frozen_encoders: int = 0,
        classifier_dropout: float = 0.1,
        classifier_hidden_layers: Optional[Tuple] = None,
        class_weights: Optional[List[float]] = None,
        backbone_model_name: Optional[str] = None,
        **kwargs,
    ):
        self.entity_types = entity_types or []
        self.number_of_layers_per_head = number_of_layers_per_head
        self.crf_reduction = crf_reduction
        self.freeze_backbone = freeze_backbone
        self.num_frozen_encoders = num_frozen_encoders
        self.classifier_dropout = classifier_dropout
        self.classifier_hidden_layers = classifier_hidden_layers
        self.class_weights = class_weights
        self.backbone_model_name = backbone_model_name
        super().__init__(**kwargs)


class MultiHeadCRF(nn.Module):
    """
    Custom CRF implementation with BIO transition masking.
    """

    def __init__(self, num_tags: int, batch_first: bool = True) -> None:
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()
        self.mask_impossible_transitions()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def mask_impossible_transitions(self) -> None:
        with torch.no_grad():
            if self.num_tags > 2:
                self.start_transitions[2] = LARGE_NEGATIVE_NUMBER
                self.transitions[0][2] = LARGE_NEGATIVE_NUMBER

            if self.num_tags > 3:
                self.start_transitions[3] = LARGE_NEGATIVE_NUMBER
                for i in range(3):
                    self.transitions[i][3] = LARGE_NEGATIVE_NUMBER
                for i in range(3):
                    self.transitions[3][i] = LARGE_NEGATIVE_NUMBER

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        device = emissions.device
        tags = tags.to(device)
        mask = mask.to(device)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator
        nllh = -llh

        if reduction == "none":
            return nllh
        if reduction == "sum":
            return nllh.sum()
        if reduction == "mean":
            return nllh.mean()
        return nllh.sum() / mask.type_as(emissions).sum()

    def decode(
        self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must have dimension of 3, got {emissions.dim()}"
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, "
                f"got {emissions.size(2)}"
            )

        if tags is not None and emissions.shape[:2] != tags.shape:
            raise ValueError(
                "the first two dimensions of emissions and tags must match, "
                f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
            )

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        device = emissions.device
        tags = tags.to(device)
        mask = mask.to(device)

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        batch_indices = torch.arange(batch_size, device=device)
        score = self.start_transitions[tags[0]]
        score += emissions[0, batch_indices, tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, batch_indices, tags[i]] * mask[i]

        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, batch_indices]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> List[List[int]]:
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        score += self.end_transitions

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class TokenClassificationModelCRF(PreTrainedModel):
    """
    Custom token classification model with CRF layer and configurable classifier head.
    """

    def __init__(
        self,
        config,
        base_model=None,
        freeze_backbone=False,
        classifier_hidden_layers=None,
        classifier_dropout=0.1,
    ):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        if base_model is None:
            self.roberta, backbone_name = _build_backbone_from_config(config)
        else:
            if hasattr(base_model, "roberta"):
                self.roberta = base_model.roberta
            else:
                self.roberta = base_model
            backbone_name = (
                getattr(getattr(self.roberta, "config", None), "_name_or_path", None)
                or getattr(config, "backbone_model_name", None)
                or getattr(config, "_name_or_path", None)
            )
            if getattr(config, "backbone_model_name", None) is None:
                config.backbone_model_name = backbone_name

        self.lm_output_size = self.roberta.config.hidden_size

        self.config.freeze_backbone = freeze_backbone
        self.config.classifier_hidden_layers = classifier_hidden_layers
        self.config.classifier_dropout = classifier_dropout

        if freeze_backbone:
            print("+" * 30, "\n\n", "Freezing backbone...", "+" * 30, "\n\n")
            for param in self.roberta.parameters():
                param.requires_grad = False
            self.roberta.eval()
        else:
            print("+" * 30, "\n\n", "NOT Freezing backbone...", "+" * 30, "\n\n")
            self.roberta.train(True)

        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.crf = CRF(self.num_labels, batch_first=True)

        self._build_classifier_head(classifier_hidden_layers, classifier_dropout)
        self.post_init()

    def _build_classifier_head(self, hidden_layers, dropout_rate):
        layers = []
        input_size = self.lm_output_size

        if not hidden_layers:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(input_size, self.num_labels)
            )
            return

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, self.num_labels))
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

        try:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        except TypeError:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels_long = labels.long()
            if attention_mask is not None:
                mask = attention_mask.bool()
                loss = -self.crf(logits, labels_long, mask=mask, reduction="mean")
            else:
                if not getattr(self, "_warned_no_attention_mask", False):
                    print(
                        "WARNING: attention_mask is None; CRF loss will include padding tokens."
                    )
                    self._warned_no_attention_mask = True
                loss = -self.crf(logits, labels_long, reduction="mean")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def device_info(self):
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.roberta.set_input_embeddings(value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        freeze_backbone = getattr(config, "freeze_backbone", False)
        classifier_hidden_layers = getattr(config, "classifier_hidden_layers", None)
        classifier_dropout = getattr(config, "classifier_dropout", 0.1)

        model = cls(
            config=config,
            freeze_backbone=freeze_backbone,
            classifier_hidden_layers=classifier_hidden_layers,
            classifier_dropout=classifier_dropout,
        )

        try:
            state_dict = torch.load(
                f"{pretrained_model_name_or_path}/pytorch_model.bin", map_location="cpu"
            )
            model.load_state_dict(state_dict)
        except Exception:
            print(
                "Warning: Could not load pre-trained weights. Using randomly initialized model."
            )

        return model


class TokenClassificationModelMultiHeadCRF(PreTrainedModel):
    """
    Multi-Head CRF model for token classification with multiple entity types.
    """

    config_class = MultiHeadCRFConfig
    base_model_prefix = "roberta"
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, base_model=None, freeze_backbone=None):
        super().__init__(config)
        self.config = config

        self.entity_types = getattr(config, "entity_types", [])
        if not self.entity_types:
            raise ValueError("entity_types must be provided in config")

        self.num_labels = config.num_labels
        self.number_of_layers_per_head = getattr(config, "number_of_layers_per_head", 1)
        self.crf_reduction = getattr(config, "crf_reduction", "mean")
        freeze_backbone = (
            freeze_backbone
            if freeze_backbone is not None
            else getattr(config, "freeze_backbone", False)
        )
        self.num_frozen_encoders = getattr(config, "num_frozen_encoders", 0)
        classifier_dropout = getattr(config, "classifier_dropout", 0.1)

        if base_model is None:
            self.roberta, backbone_name = _build_backbone_from_config(config)
        else:
            if hasattr(base_model, "roberta"):
                self.roberta = base_model.roberta
            else:
                self.roberta = base_model
            backbone_name = (
                getattr(getattr(self.roberta, "config", None), "_name_or_path", None)
                or getattr(config, "backbone_model_name", None)
                or getattr(config, "_name_or_path", None)
            )
            if getattr(config, "backbone_model_name", None) is None:
                config.backbone_model_name = backbone_name

        self.hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        print(f"Creating Multi-Head CRF with entity types: {sorted(self.entity_types)}")

        for entity_type in self.entity_types:
            for i in range(self.number_of_layers_per_head):
                setattr(
                    self,
                    f"{entity_type}_dense_{i}",
                    nn.Linear(self.hidden_size, self.hidden_size),
                )
                setattr(
                    self,
                    f"{entity_type}_dense_activation_{i}",
                    nn.GELU(approximate="none"),
                )
                setattr(
                    self, f"{entity_type}_dropout_{i}", nn.Dropout(classifier_dropout)
                )

            setattr(
                self,
                f"{entity_type}_classifier",
                nn.Linear(self.hidden_size, self.num_labels),
            )
            setattr(
                self,
                f"{entity_type}_crf",
                MultiHeadCRF(num_tags=self.num_labels, batch_first=True),
            )

        if freeze_backbone:
            self._freeze_backbone()

        self.post_init()

    def _freeze_backbone(self):
        print("+" * 30, "\n\n", "Freezing backbone...", "+" * 30, "\n\n")

        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False

        if self.num_frozen_encoders > 0:
            for _, param in islice(
                self.roberta.encoder.named_parameters(),
                self.num_frozen_encoders * NUM_PER_LAYER,
            ):
                param.requires_grad = False

    def reset_head_parameters(self):
        for entity_type in self.entity_types:
            for i in range(self.number_of_layers_per_head):
                getattr(self, f"{entity_type}_dense_{i}").reset_parameters()
            getattr(self, f"{entity_type}_classifier").reset_parameters()
            getattr(self, f"{entity_type}_crf").reset_parameters()
            getattr(self, f"{entity_type}_crf").mask_impossible_transitions()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[Dict[str, torch.LongTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        try:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        except TypeError:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = {}
        for entity_type in self.entity_types:
            head_output = sequence_output
            for i in range(self.number_of_layers_per_head):
                head_output = getattr(self, f"{entity_type}_dense_{i}")(head_output)
                head_output = getattr(self, f"{entity_type}_dense_activation_{i}")(
                    head_output
                )
                head_output = getattr(self, f"{entity_type}_dropout_{i}")(head_output)
            logits[entity_type] = getattr(self, f"{entity_type}_classifier")(
                head_output
            )

        if labels is not None:
            losses = {}
            mask = attention_mask.bool() if attention_mask is not None else None

            for entity_type in self.entity_types:
                if entity_type in labels:
                    entity_labels = (
                        labels[entity_type].long().to(logits[entity_type].device)
                    )
                    crf = getattr(self, f"{entity_type}_crf")
                    if mask is not None:
                        losses[entity_type] = crf(
                            logits[entity_type],
                            entity_labels,
                            mask=mask,
                            reduction=self.crf_reduction,
                        )
                    else:
                        if not getattr(self, "_warned_no_attention_mask", False):
                            print(
                                "WARNING: attention_mask is None; CRF loss will include padding tokens."
                            )
                            self._warned_no_attention_mask = True
                        losses[entity_type] = crf(
                            logits[entity_type],
                            entity_labels,
                            reduction=self.crf_reduction,
                        )

            total_loss = sum(losses.values())
            return total_loss, logits

        predictions = {}
        mask = attention_mask.bool() if attention_mask is not None else None

        for entity_type in self.entity_types:
            crf = getattr(self, f"{entity_type}_crf")
            if mask is not None:
                decoded = crf.decode(logits[entity_type], mask=mask)
            else:
                decoded = crf.decode(logits[entity_type])
            predictions[entity_type] = torch.tensor(decoded)

        return [predictions[ent] for ent in sorted(self.entity_types)]

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.roberta.set_input_embeddings(value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        import json
        import os

        config = kwargs.pop("config", None)

        if config is None:
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config_dict = json.load(f)
                config = MultiHeadCRFConfig(**config_dict)
            else:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=kwargs.get("trust_remote_code", True),
                )

        roberta_defaults = {
            "layer_norm_eps": 1e-5,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 514,
            "type_vocab_size": 1,
            "initializer_range": 0.02,
            "vocab_size": 52000,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "position_embedding_type": "absolute",
            "use_cache": True,
            "is_decoder": False,
            "add_cross_attention": False,
            "chunk_size_feed_forward": 0,
            "output_hidden_states": False,
            "output_attentions": False,
            "torchscript": False,
            "tie_word_embeddings": True,
            "return_dict": True,
            "gradient_checkpointing": False,
            "pruned_heads": {},
            "problem_type": None,
            "embedding_size": None,
        }

        for key, default_value in roberta_defaults.items():
            if not hasattr(config, key) or getattr(config, key) is None:
                setattr(config, key, default_value)

        freeze_backbone = getattr(config, "freeze_backbone", False)
        model = cls(config=config, freeze_backbone=freeze_backbone)

        weight_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        safetensors_file = os.path.join(
            pretrained_model_name_or_path, "model.safetensors"
        )

        try:
            if os.path.exists(safetensors_file):
                from safetensors.torch import load_file

                state_dict = load_file(safetensors_file)
                model.load_state_dict(state_dict)
            elif os.path.exists(weight_file):
                state_dict = torch.load(weight_file, map_location="cpu")
                model.load_state_dict(state_dict)
            else:
                print(
                    "Warning: No pre-trained weights found. Using randomly initialized model."
                )
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights: {e}")

        return model


class MultiHeadConfig(PretrainedConfig):
    """
    Configuration class for Multi-Head models (without CRF).
    """

    model_type = "multihead-tagger"

    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        number_of_layers_per_head: int = 1,
        freeze_backbone: bool = False,
        num_frozen_encoders: int = 0,
        classifier_dropout: float = 0.1,
        use_class_weights: bool = False,
        class_weights: Optional[Dict[str, List[float]]] = None,
        backbone_model_name: Optional[str] = None,
        **kwargs,
    ):
        self.entity_types = entity_types or []
        self.number_of_layers_per_head = number_of_layers_per_head
        self.freeze_backbone = freeze_backbone
        self.num_frozen_encoders = num_frozen_encoders
        self.classifier_dropout = classifier_dropout
        self.use_class_weights = use_class_weights
        self.class_weights = class_weights
        self.backbone_model_name = backbone_model_name
        super().__init__(**kwargs)


class TokenClassificationModelMultiHead(PreTrainedModel):
    """
    Multi-Head model for token classification with multiple entity types (no CRF).
    """

    config_class = MultiHeadConfig
    base_model_prefix = "roberta"
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, base_model=None, freeze_backbone=None):
        super().__init__(config)
        self.config = config

        self.entity_types = getattr(config, "entity_types", [])
        if not self.entity_types:
            raise ValueError("entity_types must be provided in config")

        self.num_labels = config.num_labels
        self.number_of_layers_per_head = getattr(config, "number_of_layers_per_head", 1)
        freeze_backbone = (
            freeze_backbone
            if freeze_backbone is not None
            else getattr(config, "freeze_backbone", False)
        )
        self.num_frozen_encoders = getattr(config, "num_frozen_encoders", 0)
        classifier_dropout = getattr(config, "classifier_dropout", 0.1)

        self.use_class_weights = getattr(config, "use_class_weights", False)
        self.class_weights = getattr(config, "class_weights", None)

        if base_model is None:
            self.roberta, backbone_name = _build_backbone_from_config(config)
        else:
            if hasattr(base_model, "roberta"):
                self.roberta = base_model.roberta
            else:
                self.roberta = base_model
            backbone_name = (
                getattr(getattr(self.roberta, "config", None), "_name_or_path", None)
                or getattr(config, "backbone_model_name", None)
                or getattr(config, "_name_or_path", None)
            )
            if getattr(config, "backbone_model_name", None) is None:
                config.backbone_model_name = backbone_name

        self.hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        print(
            f"Creating Multi-Head model with entity types: {sorted(self.entity_types)}"
        )

        for entity_type in self.entity_types:
            for i in range(self.number_of_layers_per_head):
                setattr(
                    self,
                    f"{entity_type}_dense_{i}",
                    nn.Linear(self.hidden_size, self.hidden_size),
                )
                setattr(
                    self,
                    f"{entity_type}_dense_activation_{i}",
                    nn.GELU(approximate="none"),
                )
                setattr(
                    self, f"{entity_type}_dropout_{i}", nn.Dropout(classifier_dropout)
                )

            setattr(
                self,
                f"{entity_type}_classifier",
                nn.Linear(self.hidden_size, self.num_labels),
            )

        self.loss_fns = nn.ModuleDict()
        for entity_type in self.entity_types:
            if (
                self.use_class_weights
                and self.class_weights
                and entity_type in self.class_weights
            ):
                weight = torch.tensor(
                    self.class_weights[entity_type], dtype=torch.float
                )
                self.loss_fns[entity_type] = nn.CrossEntropyLoss(
                    weight=weight, ignore_index=-100
                )
            else:
                self.loss_fns[entity_type] = nn.CrossEntropyLoss(ignore_index=-100)

        if freeze_backbone:
            self._freeze_backbone()

        self.post_init()

    def _freeze_backbone(self):
        print("+" * 30, "\n\n", "Freezing backbone...", "+" * 30, "\n\n")

        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False

        if self.num_frozen_encoders > 0:
            for _, param in islice(
                self.roberta.encoder.named_parameters(),
                self.num_frozen_encoders * NUM_PER_LAYER,
            ):
                param.requires_grad = False

    def reset_head_parameters(self):
        for entity_type in self.entity_types:
            for i in range(self.number_of_layers_per_head):
                getattr(self, f"{entity_type}_dense_{i}").reset_parameters()
            getattr(self, f"{entity_type}_classifier").reset_parameters()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[Dict[str, torch.LongTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        try:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        except TypeError:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = {}
        for entity_type in self.entity_types:
            head_output = sequence_output
            for i in range(self.number_of_layers_per_head):
                head_output = getattr(self, f"{entity_type}_dense_{i}")(head_output)
                head_output = getattr(self, f"{entity_type}_dense_activation_{i}")(
                    head_output
                )
                head_output = getattr(self, f"{entity_type}_dropout_{i}")(head_output)
            logits[entity_type] = getattr(self, f"{entity_type}_classifier")(
                head_output
            )

        if labels is not None:
            losses = {}

            for entity_type in self.entity_types:
                if entity_type in labels:
                    entity_labels = (
                        labels[entity_type].long().to(logits[entity_type].device)
                    )
                    entity_logits = logits[entity_type]
                    loss_fct = self.loss_fns[entity_type]

                    if hasattr(loss_fct, "weight") and loss_fct.weight is not None:
                        loss_fct.weight = loss_fct.weight.to(entity_logits.device)

                    losses[entity_type] = loss_fct(
                        entity_logits.view(-1, self.num_labels),
                        entity_labels.view(-1),
                    )

            total_loss = sum(losses.values())
            return total_loss, logits

        predictions = {}
        for entity_type in self.entity_types:
            preds = torch.argmax(logits[entity_type], dim=-1)
            predictions[entity_type] = preds

        return [predictions[ent] for ent in sorted(self.entity_types)]

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.roberta.set_input_embeddings(value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        import json
        import os

        config = kwargs.pop("config", None)

        if config is None:
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config_dict = json.load(f)
                config = MultiHeadConfig(**config_dict)
            else:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=kwargs.get("trust_remote_code", True),
                )

        roberta_defaults = {
            "layer_norm_eps": 1e-5,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 514,
            "type_vocab_size": 1,
            "initializer_range": 0.02,
            "vocab_size": 52000,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "position_embedding_type": "absolute",
            "use_cache": True,
            "is_decoder": False,
            "add_cross_attention": False,
            "chunk_size_feed_forward": 0,
            "output_hidden_states": False,
            "output_attentions": False,
            "torchscript": False,
            "tie_word_embeddings": True,
            "return_dict": True,
            "gradient_checkpointing": False,
            "pruned_heads": {},
            "problem_type": None,
            "embedding_size": None,
        }

        for key, default_value in roberta_defaults.items():
            if not hasattr(config, key) or getattr(config, key) is None:
                setattr(config, key, default_value)

        freeze_backbone = getattr(config, "freeze_backbone", False)
        model = cls(config=config, freeze_backbone=freeze_backbone)

        weight_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        safetensors_file = os.path.join(
            pretrained_model_name_or_path, "model.safetensors"
        )

        try:
            if os.path.exists(safetensors_file):
                from safetensors.torch import load_file

                state_dict = load_file(safetensors_file)
                model.load_state_dict(state_dict)
            elif os.path.exists(weight_file):
                state_dict = torch.load(weight_file, map_location="cpu")
                model.load_state_dict(state_dict)
            else:
                print(
                    "Warning: No pre-trained weights found. Using randomly initialized model."
                )
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights: {e}")

        return model


class TokenClassificationModel(PreTrainedModel):
    """
    Custom token classification model with configurable classifier head (no CRF).
    """

    def __init__(self, config, base_model=None):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        if base_model is None:
            self.roberta, backbone_name = _build_backbone_from_config(config)
        else:
            if hasattr(base_model, "roberta"):
                self.roberta = base_model.roberta
            else:
                self.roberta = base_model
            backbone_name = (
                getattr(getattr(self.roberta, "config", None), "_name_or_path", None)
                or getattr(config, "backbone_model_name", None)
                or getattr(config, "_name_or_path", None)
            )

        if getattr(config, "backbone_model_name", None) is None:
            config.backbone_model_name = backbone_name

        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        classifier_hidden_layers = getattr(config, "classifier_hidden_layers", None)
        classifier_dropout = getattr(config, "classifier_dropout", 0.1)

        if classifier_hidden_layers is not None:
            in_size = self.roberta.config.hidden_size
            layers = []
            if classifier_hidden_layers:
                for h in classifier_hidden_layers:
                    layers += [
                        nn.Linear(in_size, h),
                        nn.ReLU(),
                        nn.Dropout(classifier_dropout),
                    ]
                    in_size = h
            layers.append(nn.Linear(in_size, config.num_labels))
            self.classifier = nn.Sequential(*layers)
        else:
            self.classifier = nn.Linear(
                self.roberta.config.hidden_size, config.num_labels
            )

        if isinstance(self.classifier, nn.Sequential):
            for module in self.classifier:
                if isinstance(module, nn.Linear):
                    self._init_weights(module)
        elif isinstance(self.classifier, nn.Linear):
            self._init_weights(self.classifier)

        self.post_init()

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

        try:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        except TypeError:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.roberta.set_input_embeddings(value)


def load_custom_cardioner_multiclass_model(model_path: str, device: str = "auto"):
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

    print(f"Loading custom CardioNER multiclass model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForTokenClassification.from_pretrained(
        model_path, trust_remote_code=True, use_safetensors=True, use_safetensors=True
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    print(f"Model loaded successfully on {device}")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of labels: {model.num_labels}")

    return model, tokenizer, model.config


def load_custom_multihead_crf_model(model_path: str, device: str = "auto"):
    import json
    import os

    from transformers import AutoTokenizer

    required_files = ["config.json", "modeling.py"]
    missing_files = [
        f for f in required_files if not os.path.exists(os.path.join(model_path, f))
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {model_path}: {missing_files}"
        )

    print(f"Loading Multi-Head CRF model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(os.path.join(model_path, "config.json"), "r") as f:
        config_dict = json.load(f)

    config = MultiHeadCRFConfig(**config_dict)

    model = TokenClassificationModelMultiHeadCRF.from_pretrained(
        model_path, config=config
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    print(f"Model loaded successfully on {device}")
    print(f"Model type: {type(model).__name__}")
    print(f"Entity types: {model.entity_types}")
    print(f"Number of labels per head: {model.num_labels}")

    return model, tokenizer, model.config


def validate_custom_multiclass_model_directory(model_path: str) -> dict:
    import json
    import os

    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "files_found": [],
        "model_info": {},
    }

    required_files = {
        "config.json": "Model configuration",
        "modeling.py": "Custom model class definition",
        "pytorch_model.bin": "Model weights",
    }

    optional_files = {
        "tokenizer.json": "Tokenizer vocabulary",
        "tokenizer_config.json": "Tokenizer configuration",
        "training_args.json": "Training arguments",
    }

    for filename, description in required_files.items():
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            validation_results["files_found"].append(f"{filename} ({description})")
        else:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Missing required file: {filename} - {description}"
            )

    for filename, description in optional_files.items():
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            validation_results["files_found"].append(f"{filename} ({description})")
        else:
            validation_results["warnings"].append(
                f"Missing optional file: {filename} - {description}"
            )

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
            validation_results["model_info"]["use_crf"] = (
                "TokenClassificationModelCRF" in str(config.get("architectures", []))
            )

            if not config.get("auto_map"):
                validation_results["warnings"].append(
                    "No auto_map found in config - may not load correctly with trust_remote_code=True"
                )

        except json.JSONDecodeError as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Invalid config.json: {str(e)}")

    modeling_path = os.path.join(model_path, "modeling.py")
    if os.path.exists(modeling_path):
        try:
            with open(modeling_path, "r") as f:
                content = f.read()

            required_classes = [
                "TokenClassificationModel",
                "TokenClassificationModelCRF",
            ]
            missing_classes = [cls for cls in required_classes if cls not in content]

            if missing_classes:
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"modeling.py missing required classes: {missing_classes}"
                )

        except Exception as e:
            validation_results["warnings"].append(
                f"Could not read modeling.py: {str(e)}"
            )

    return validation_results


try:
    from transformers import AutoConfig

    AutoConfig.register("multihead-crf-tagger", MultiHeadCRFConfig)
except Exception:
    pass


def patch_legacy_model(
    model_path: str, backbone_model_name: str, dry_run: bool = True
) -> bool:
    import json
    import os
    import shutil

    config_path = os.path.join(model_path, "config.json")

    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found at {config_path}")
        return False

    with open(config_path, "r") as f:
        config = json.load(f)

    if "backbone_model_name" in config:
        print(f"Model already has backbone_model_name: {config['backbone_model_name']}")
        if config["backbone_model_name"] == backbone_model_name:
            print("No changes needed.")
            return True
        else:
            print(f"WARNING: Existing backbone_model_name differs from provided value!")
            print(f"  Existing: {config['backbone_model_name']}")
            print(f"  Provided: {backbone_model_name}")
            if dry_run:
                print("Would update to new value (dry_run=True)")
            else:
                print("Updating to new value...")

    config["backbone_model_name"] = backbone_model_name

    if dry_run:
        print(f"\n[DRY RUN] Would patch {config_path}:")
        print(f'  Adding: backbone_model_name = "{backbone_model_name}"')
        print("\nTo apply this patch, run with dry_run=False")
        return True

    backup_path = config_path + ".backup"
    shutil.copy2(config_path, backup_path)
    print(f"Created backup at {backup_path}")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Successfully patched {config_path}")
    print(f'  Added: backbone_model_name = "{backbone_model_name}"')

    return True


def patch_multiple_models(
    model_paths: list, backbone_model_name: str, dry_run: bool = True
) -> dict:
    results = {}
    for path in model_paths:
        print(f"\n{'=' * 60}")
        print(f"Processing: {path}")
        print("=" * 60)
        results[path] = patch_legacy_model(path, backbone_model_name, dry_run)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    success = sum(1 for v in results.values() if v)
    print(
        f"Successfully {'would patch' if dry_run else 'patched'}: {success}/{len(model_paths)}"
    )

    return results
