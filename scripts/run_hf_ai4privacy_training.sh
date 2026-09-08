#!/usr/bin/env bash
set -euo pipefail

# Ready-to-run training script for:
#   ai4privacy/pii-masking-openpii-1.5m
#
# Usage:
#   bash scripts/run_hf_ai4privacy_training.sh multilabel
#   bash scripts/run_hf_ai4privacy_training.sh multiclass
#
# Optional:
#   export HF_TOKEN=hf_xxx   # only needed for gated/private datasets

MODE="${1:-multiclass}"

COMMON_ARGS=(
  --lang multi
  --hf_dataset ai4privacy/pii-masking-openpii-1.5m
  --text_column source_text
  --tags_column privacy_mask
  --selector_column language
  --selection nl en fr de es it da sv pt
  --parse_annotations
  --train_model
  --force_splitter
  --model FacebookAI/xlm-roberta-base
  --chunk_size=32
  --max_token_length=32
  --batch_size 2
  --accumulation_steps 8
  --learning_rate=2e-5
  --weight_decay=1e-5
  --num_epochs=5
  --num_splits=5
  --use_class_weights
  --classifier_hidden_layers 768 768 768
  --chunk_type paragraph
  --entity_types AGE DATE BUILDINGNUM CITY ZIPCODE STREET GIVENNAME SURNAME SOCIALNUM PASSPORTNUM SEX GENDER TELEPHONENUM IDCARDNUM EMAIL

)

if [ -n "${HF_TOKEN:-}" ]; then
  COMMON_ARGS+=(--hf_token "${HF_TOKEN}")
fi

if [ "${MODE}" = "multiclass" ]; then
  poetry run python -m cardioner.main \
    "${COMMON_ARGS[@]}" \
    --multiclass \
    --output_dir /media/bramiozo/Storage2/DATA/NER/PII/hf_ai4privacy_multiclass
elif [ "${MODE}" = "multilabel" ]; then
  poetry run python -m cardioner.main \
    "${COMMON_ARGS[@]}" \
    --output_dir /media/bramiozo/Storage2/DATA/NER/PII/hf_ai4privacy_multilabel
else
  echo "Unknown mode: ${MODE}. Use 'multilabel' or 'multiclass'." >&2
  exit 1
fi
