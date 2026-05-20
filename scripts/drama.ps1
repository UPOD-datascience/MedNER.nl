$categories = @("PROCEDURE", "DISEASE", "MEDICATION", "SYMPTOM")
$folds = @(0,1,2,3,4,5,6,7,8,9)

$lang="sv"
$stride= 256

$model="SV-BERT-base"
$base_folder="/media/bramiozo/Storage2/DATA/NER/DT4H_results/paper/SV/${model}/multiclass"
foreach ($category in $categories) {
    foreach ($fold in $folds){
        poetry run python -m cardioner.main --inference_only `
                                    --output_file_prefix=${lang}_${category}_${stride}_${fold}_eval `
                                    --model_path=${base_folder} / multiclass_maxTL256_batch16_chunk256_epochs10_paper_paragraph_${category}/fold_${fold}`
                                    --inference_pipe=dt4h `
                                    --corpus_inference=/media/bramiozo/Storage1/bramiozo/DEV/GIT/UMCU/CardioNER.nl/assets/CardioCCC/Eval/${lang} `
                                    --inference_batch_size=8 `
                                    --lang=$lang `
                                    --trust_remote_code `
                                    --inference_stride=$stride
    }
}

$model="CardioBERTa.sv"
$base_folder="/media/bramiozo/Storage2/DATA/NER/DT4H_results/paper/SV/${model}/multiclass"
foreach ($category in $categories) {
    foreach ($fold in $folds){
        poetry run python -m cardioner.main --inference_only `
                                    --output_file_prefix=${lang}_${category}_eval `
                                    --model_path=${base_folder} / multiclass_maxTL256_batch16_chunk256_epochs10_paper_paragraph_${category}`
                                    --inference_pipe=dt4h `
                                    --corpus_inference=/media/bramiozo/Storage1/bramiozo/DEV/GIT/UMCU/CardioNER.nl/assets/CardioCCC/Eval/${lang}/fold_${fold} `
                                    --inference_batch_size=8 `
                                    --lang=$lang `
                                    --trust_remote_code `
                                    --inference_stride=$stride
    }
}
$lang="nl"
$stride= 256

$model="RobBERT2023_base"
$base_folder="/media/bramiozo/Storage2/DATA/NER/DT4H_results/paper/SV/${model}/multiclass"
foreach ($category in $categories) {
    foreach ($fold in $folds){
        poetry run python -m cardioner.main --inference_only `
                                    --output_file_prefix=${lang}_${category}_eval `
                                    --model_path=${base_folder} / multiclass_maxTL256_batch16_chunk256_epochs10_paper_paragraph_${category}`
                                    --inference_pipe=dt4h `
                                    --corpus_inference=/media/bramiozo/Storage1/bramiozo/DEV/GIT/UMCU/CardioNER.nl/assets/CardioCCC/Eval/${lang}/fold_${fold} `
                                    --inference_batch_size=8 `
                                    --lang=$lang `
                                    --trust_remote_code `
                                    --inference_stride=$stride
    }
}
$model="CardioBERTa.nl"
$base_folder="/media/bramiozo/Storage2/DATA/NER/DT4H_results/paper/SV/${model}/multiclass"
foreach ($category in $categories) {
    foreach ($fold in $folds){
        poetry run python -m cardioner.main --inference_only `
                                    --output_file_prefix=${lang}_${category}_eval `
                                    --model_path=${base_folder} / multiclass_maxTL256_batch16_chunk256_epochs10_paper_paragraph_${category}`
                                    --inference_pipe=dt4h `
                                    --corpus_inference=/media/bramiozo/Storage1/bramiozo/DEV/GIT/UMCU/CardioNER.nl/assets/CardioCCC/Eval/${lang}/fold_${fold} `
                                    --inference_batch_size=8 `
                                    --lang=$lang `
                                    --trust_remote_code `
                                    --inference_stride=$stride
}
}
