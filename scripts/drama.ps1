$languages = @("sv", "ro", "it", "es", "en", "cz", "nl")

foreach ($lang in $languages) {

poetry run python -m cardioner.main --inference_only `
                                    --output_file_prefix=${lang}_test `
                                    --model_path=T:\laupodteam\AIOS\Bram\language_modeling\Models\language_models\CardioCCC\EuroBERT\multilabel_3ldense_20epochs_40splits\fold_0 `
                                    --inference_pipe=dt4h `
                                    --corpus_inference=T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\CardioCCC_EVAL\data_selection\${lang} `
                                    --inference_batch_size=8 `
                                    --lang=multi `
                                    --trust_remote_code `
                                    --inference_stride=128
}
