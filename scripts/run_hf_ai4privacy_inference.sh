poetry run python -m cardioner.main --inference_only --model_path=output/hf_ai4privacy_multiclass/merged_chordal --corpus_inference=ai4privacy/pii-masking-openpii-1.5m --text_column=source_text --tags_column=privacy_mask  --lang=nl --inference_stride=32 --selector_column=language --selection=nl --inference_pipe=dt4h --trust_remote_code --output_file_prefix="chordal_"

poetry run python -m cardioner.main --inference_only --model_path=output/hf_ai4privacy_multiclass/merged_harmonic --corpus_inference=ai4privacy/pii-masking-openpii-1.5m --text_column=source_text --tags_column=privacy_mask  --lang=nl --inference_stride=32 --selector_column=language --selection=nl --inference_pipe=dt4h --trust_remote_code --output_file_prefix="harmonic_"

poetry run python -m cardioner.main --inference_only --model_path=output/hf_ai4privacy_multiclass/merged_chain --corpus_inference=ai4privacy/pii-masking-openpii-1.5m --text_column=source_text --tags_column=privacy_mask  --lang=nl --inference_stride=32 --selector_column=language --selection=nl --inference_pipe=dt4h --trust_remote_code --output_file_prefix="chain_"

poetry run python -m cardioner.main --inference_only --model_path=output/hf_ai4privacy_multiclass/fold_0 --corpus_inference=ai4privacy/pii-masking-openpii-1.5m --text_column=source_text --tags_column=privacy_mask  --lang=nl --inference_stride=32 --selector_column=language --selection=nl --inference_pipe=dt4h --trust_remote_code --output_file_prefix="fold0_"

poetry run python -m cardioner.main --inference_only --model_path=output/hf_ai4privacy_multiclass/fold_1 --corpus_inference=ai4privacy/pii-masking-openpii-1.5m --text_column=source_text --tags_column=privacy_mask  --lang=nl --inference_stride=32 --selector_column=language --selection=nl --inference_pipe=dt4h --trust_remote_code --output_file_prefix="fold1_"

poetry run python -m cardioner.main --inference_only --model_path=output/hf_ai4privacy_multiclass/fold_2 --corpus_inference=ai4privacy/pii-masking-openpii-1.5m --text_column=source_text --tags_column=privacy_mask  --lang=nl --inference_stride=32 --selector_column=language --selection=nl --inference_pipe=dt4h --trust_remote_code --output_file_prefix="fold2_"

poetry run python -m cardioner.main --inference_only --model_path=output/hf_ai4privacy_multiclass/fold_3 --corpus_inference=ai4privacy/pii-masking-openpii-1.5m --text_column=source_text --tags_column=privacy_mask  --lang=nl --inference_stride=32 --selector_column=language --selection=nl --inference_pipe=dt4h --trust_remote_code --output_file_prefix="fold3_"

poetry run python -m cardioner.main --inference_only --model_path=output/hf_ai4privacy_multiclass/fold_4 --corpus_inference=ai4privacy/pii-masking-openpii-1.5m --text_column=source_text --tags_column=privacy_mask  --lang=nl --inference_stride=32 --selector_column=language --selection=nl --inference_pipe=dt4h --trust_remote_code --output_file_prefix="fold4_"
