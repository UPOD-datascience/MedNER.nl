#!/bin/bash

# Define variables for the model configuration
ORGANIZATION="DT4H"  # Organization name
MODEL_NAME="cardio-ner-sv-symptom-bert-base-swedish-cased-multiclass"  # Model repository name (will use hyphens, not underscores)
MODEL_PATH="/media/bramiozo/T7/DATA/NER/CardioCCC/SV/SV-BERT-base/multiclass/multiclass_maxTL256_batch16_chunk256_epochs10_paper_paragraph_SYMPTOM/merged"  # Path to the directory containing model files (config.json, pytorch_model.bin, tokenizer.json, etc.)
DESCRIPTION="Swedish cardiology NER model for symptom recognition"  # Model description
LANGUAGE="sv"  # Language code (https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes)
TASK="token-classification"  # Task t ype: token-classification, text-classification, sequence-labeling, other
FRAMEWORK="pytorch"  # Framework: pytorch, tensorflow, jax, etc.
TOKEN="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"  # Replace 'YOUR_TOKEN' with your actual Hugging Face token
PRIVATE=false  # Set to true to make the repository private (remove --private flag below if false)

# Collection parameters (optional - must exist on HF Hub)
COLLECTION_NAME="CardioNER"  # Optional: Add model to existing collection

# Optional: Add tags for better discoverability
TAGS=("clinical" "medical" "ner" "healthcare" "cardiology" "swedish")

# Run the script to upload the model to Hugging Face
python src/hf_model.py \
  --organization "$ORGANIZATION" \
  --name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --description "$DESCRIPTION" \
  --language "$LANGUAGE" \
  --task "$TASK" \
  --framework "$FRAMEWORK" \
  --token "$TOKEN" \
  --tags "${TAGS[@]}" \
  --collection_name "$COLLECTION_NAME" \
  $([ "$PRIVATE" = true ] && echo "--private")
