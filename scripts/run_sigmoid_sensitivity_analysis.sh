#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=/media/bramiozo/Storage1/bramiozo/DEV/GIT/UMCU/CardioNER.nl/assets/CardioCCC
MODEL_DIR=/media/bramiozo/T7/DATA/NER/CardioCCC
LANGUAGE=NL
MODELS=(CardioBERTa.nl RobBERT2023_base)
SPEC=multilabel_maxTL256_batch16_chunk256_epochs10_paper_paragraph

# language mapper if NL then Dutch if SV then Swedish
if [[ "${LANGUAGE}" == "NL" ]]; then LANGUAGE_FULL=Dutch; fi
if [[ "${LANGUAGE}" == "SV" ]]; then LANGUAGE_FULL=Swedish; fi

# JSONL corpus file expected by cardioner.run_inference_on_folds
BULK_FILE=${BASE_DIR}/${LANGUAGE_FULL}/cardioccc_${LANGUAGE,,}.json
SPLIT_FILE=${BASE_DIR}/paper_splits/splits_cv10_holdout_test.json

for MODEL in "${MODELS[@]}"; do
    poetry run python scripts/run_sigmoid_sensitivity_analysis.py \
        --activation sigmoid \
        --bulk_file "${BULK_FILE}" \
        --split_file "${SPLIT_FILE}" \
        --model_folder "${MODEL_DIR}/${LANGUAGE}/${MODEL}/multilabel/${SPEC}" \
        --output_dir "output/${LANGUAGE}_SIGMOID_SENSITIVITY/${MODEL}" \
        --lang "${LANGUAGE,,}" \
        --pipe dt4h
done


```bash
poetry run python scripts/aggregate_sigmoid_sensitivity_results.py \
  --input-root output/${LANGUAGE}_SIGMOID_SENSITIVITY
