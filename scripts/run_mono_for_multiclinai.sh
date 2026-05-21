#!/usr/bin/env bash

# Do not exit on first error; continue with next language/job.
set -u

LOG_FILE="real_inference_log.txt"
BASE_CORPUS_PATH="/media/bramiozo/Storage1/bramiozo/DEV/GIT/UMCU/CardioNER.nl/assets/MultiClinNER"
LANGUAGES=(nl)

log_msg() {
    local msg="$1"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${ts}] ${msg}" | tee -a "${LOG_FILE}"
}

run_job() {
    local model_path="$1"
    local entity="$2"          # disease | symptom | procedure
    local output_prefix_base="$3"

    local split_file="${model_path}/split.json"

    for lang in "${LANGUAGES[@]}"; do
        local corpus_path="${BASE_CORPUS_PATH}/MultiClinNER-${lang}/${entity}.json"
        local output_prefix="${output_prefix_base}${lang}_test"

        log_msg "Starting language: ${lang} from ${corpus_path}"

        poetry run python -m cardioner.main \
            --inference_only \
            --model_path="${model_path}" \
            --inference_pipe=dt4h \
            --corpus_inference="${corpus_path}" \
            --split_file="${split_file}" \
            --inference_batch_size=4 \
            --lang=multi \
            --trust_remote_code \
            --inference_stride=128 \
            --output_file_prefix="${output_prefix}"

        exit_code=$?
        if [[ ${exit_code} -ne 0 ]]; then
            log_msg "ERROR for language: ${lang} (exit code ${exit_code})"
            continue
        fi

        log_msg "Finished language: ${lang}"
    done
}

# RobBERT2023
##################
# run_job \
#     "/media/bramiozo/Storage2/DATA/NER/MultiClinAI/NL/RobBERT2023-large/DISEASE_multiclass_maxTL128_batch16_chunk128_epochs10_centered_3l_dnn/fold_0" \
#     "disease" \
#     "DISEASE_RobBERT2023_multiclass_"

# run_job \
#     "/media/bramiozo/Storage2/DATA/NER/MultiClinAI/NL/RobBERT2023-large/SYMPTOM_multiclass_maxTL128_batch16_chunk128_epochs10_centered_3l_dnn/fold_0" \
#     "symptom" \
#     "SYMPTOM_RobBERT2023_multiclass_"

# run_job \
#     "/media/bramiozo/Storage2/DATA/NER/MultiClinAI/NL/RobBERT2023-large/PROCEDURE_multiclass_maxTL128_batch16_chunk128_epochs10_centered_3l_dnn/fold_0" \
#     "procedure" \
#     "PROCEDURE_RobBERT2023_multiclass_"

# MedRoBERTa.nl
###################
run_job \
    "/media/bramiozo/Storage2/DATA/NER/MultiClinAI/NL/MedRoBERTa.nl/DISEASE_multiclass_maxTL128_batch16_chunk128_epochs20_centered_3l_dnn/fold_0" \
    "disease" \
    "DISEASE_MedRoBERTa_multiclass_"

run_job \
    "/media/bramiozo/Storage2/DATA/NER/MultiClinAI/NL/MedRoBERTa.nl/SYMPTOM_multiclass_maxTL128_batch16_chunk128_epochs20_centered_3l_dnn/fold_0" \
    "symptom" \
    "SYMPTOM_MedRoBERTa_multiclass_"

run_job \
    "/media/bramiozo/Storage2/DATA/NER/MultiClinAI/NL/MedRoBERTa.nl/PROCEDURE_multiclass_maxTL128_batch16_chunk128_epochs20_centered_3l_dnn/fold_0" \
    "procedure" \
    "PROCEDURE_MedRoBERTa_multiclass_"

echo "All jobs completed."
if [[ -t 0 ]]; then
    read -r -n 1 -s -p "Press any key to exit..."
    echo
fi
