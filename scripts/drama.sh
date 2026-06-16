#!/usr/bin/env bash
set -euo pipefail

categories=("PROCEDURE" "DISEASE" "MEDICATION" "SYMPTOM")
folds=(0 1 2 3 4 5 6 7 8 9)
stride=256

run_inference() {
  local lang="$1"
  local model="$2"
  local model_has_fold_subdir="$3" # "yes" or "no"

  local base_folder="/media/bramiozo/Storage2/DATA/NER/DT4H_results/paper/${lang^^}/${model}/multiclass"

  for category in "${categories[@]}"; do
    for fold in "${folds[@]}"; do
      local model_path
      local corpus_inference

      if [[ "${model_has_fold_subdir}" == "yes" ]]; then
        model_path="${base_folder}/multiclass_maxTL256_batch16_chunk256_epochs10_paper_paragraph_${category}/fold_${fold}"
        corpus_inference="/media/bramiozo/Storage1/bramiozo/DEV/GIT/UMCU/CardioNER.nl/assets/CardioCCC/Eval/${lang}"
      else
        model_path="${base_folder}/multiclass_maxTL256_batch16_chunk256_epochs10_paper_paragraph_${category}"
        corpus_inference="/media/bramiozo/Storage1/bramiozo/DEV/GIT/UMCU/CardioNER.nl/assets/CardioCCC/Eval/${lang}/fold_${fold}"
      fi

      poetry run python -m cardioner.main --inference_only \
        --output_dir="output/${model}/${category}" \
        --output_file_prefix="${lang}_${stride}_${fold}_eval" \
        --model_path="${model_path}" \
        --inference_pipe="dt4h" \
        --corpus_inference="${corpus_inference}" \
        --inference_batch_size=8 \
        --lang="${lang}" \
        --trust_remote_code \
        --inference_stride="${stride}"
    done

    poetry run python scripts/combine_inference.py --results_folder output/${model}/${category} --output_file output/${model}/Union/${category}.tsv
  done
  poetry run python scripts/combine_inference.py --results_folder output/${model}/Union --output_file output/${model}/Union/ALL.tsv
}

# Swedish
#run_inference "sv" "SV-BERT-base" "yes"
#run_inference "sv" "CardioBERTa.sv" "yes"

# # Dutch
#run_inference "nl" "RobBERT2023_base" "yes"
run_inference "nl" "CardioBERTa.nl" "yes"
