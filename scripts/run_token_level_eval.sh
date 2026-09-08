#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=/media/bramiozo/Storage1/bramiozo/DEV/GIT/UMCU/CardioNER.nl/assets/CardioCCC
MODEL_DIR=/media/bramiozo/T7/DATA/NER/CardioCCC
LANGUAGE=SV
MODELS=(CardioBERTa.sv SV-BERT-base)
SPEC=multilabel_maxTL256_batch16_chunk256_epochs10_paper_paragraph

# language mapper (extend here for additional languages)
case "${LANGUAGE}" in
    NL) LANGUAGE_FULL=Dutch ;;
    SV) LANGUAGE_FULL=Swedish ;;
    *)
        echo "Unsupported LANGUAGE='${LANGUAGE}'. Add mapping in run_token_level_eval.sh." >&2
        exit 1
        ;;
esac

# JSONL corpus file expected by scripts/run_token_level_eval.py
BULK_FILE="${BASE_DIR}/${LANGUAGE_FULL}/cardioccc_${LANGUAGE,,}.json"
SPLIT_FILE="${BASE_DIR}/paper_splits/splits_cv10_holdout_test.json"
OUTPUT_ROOT="output/${LANGUAGE}_TOKEN_LEVEL_EVAL"

if [[ ! -f "${BULK_FILE}" ]]; then
    echo "Bulk file not found: ${BULK_FILE}" >&2
    exit 1
fi
if [[ ! -f "${SPLIT_FILE}" ]]; then
    echo "Split file not found: ${SPLIT_FILE}" >&2
    exit 1
fi

for MODEL in "${MODELS[@]}"; do
    MODEL_FOLDER="${MODEL_DIR}/${LANGUAGE}/${MODEL}/multilabel/${SPEC}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL}"

    if [[ ! -d "${MODEL_FOLDER}" ]]; then
        echo "Skipping ${MODEL}: model folder not found: ${MODEL_FOLDER}" >&2
        continue
    fi

    poetry run python scripts/run_token_level_eval.py \
        --bulk_file "${BULK_FILE}" \
        --split_file "${SPLIT_FILE}" \
        --model_folder "${MODEL_FOLDER}" \
        --output_dir "${OUTPUT_DIR}" \
        --fp16

done
