# Prevent script from stopping on non-terminating errors
$ErrorActionPreference = "Continue"

# Common arguments
$model_path = "T:\laupodteam\AIOS\Bram\language_modeling\Models\language_models\MultiClinAI\DeBERTa\multiclass_3ldense_20epochs_procedure\fold_0"
$base_corpus_path = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined"
$filter_file = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined\batch_1_ids.txt"

# Languages
$languages = @( "nl", "sv", "ro", "it", "es", "en", "cz")

# Log file
$logFile = "real_inference_log.txt"

foreach ($lang in $languages) {
    try {
        $corpus_path = Join-Path $base_corpus_path "MultiClinNER-$lang\test_silver\procedure\txt"
        $output_prefix = "PROCEDURE_DeBERTa_multiclass_$lang"

        $msg = "[$(Get-Date)] Starting language: $lang from $corpus_path"
        Write-Host $msg
        Add-Content $logFile $msg

        poetry run python -m cardioner.main `
            --inference_only `
            --model_path=$model_path `
            --inference_pipe=dt4h `
            --corpus_inference=$corpus_path `
            --inference_batch_size=4 `
            --lang=multi `
            --trust_remote_code `
            --inference_stride=128 `
            --output_file_prefix=$output_prefix

        if ($LASTEXITCODE -ne 0) {
            $err = "[$(Get-Date)] ERROR for language: $lang (exit code $LASTEXITCODE)"
            Write-Warning $err
            Add-Content $logFile $err
            continue   # move to next language instead of stopping
        }

        $done = "[$(Get-Date)] Finished language: $lang"
        Write-Host $done
        Add-Content $logFile $done
    }
    catch {
        $err = "[$(Get-Date)] EXCEPTION for language: $lang - $_"
        Write-Error $err
        Add-Content $logFile $err
        continue
    }
}

# Common arguments
$model_path = "T:\laupodteam\AIOS\Bram\language_modeling\Models\language_models\MultiClinAI\DeBERTa\multiclass_3ldense_20epochs_symptom\fold_0"
$base_corpus_path = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined"
$filter_file = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined\batch_1_ids.txt"

# Languages
$languages = @("sv", "ro", "nl", "it", "es", "en", "cz")

# Log file
$logFile = "real_inference_log.txt"

foreach ($lang in $languages) {
    try {
        $corpus_path = Join-Path $base_corpus_path "MultiClinNER-$lang\test_silver\symptom\txt"
        $output_prefix = "SYMPTOM_DeBERTa_multiclass_$lang"

        $msg = "[$(Get-Date)] Starting language: $lang from $corpus_path"
        Write-Host $msg
        Add-Content $logFile $msg

        poetry run python -m cardioner.main `
            --inference_only `
            --model_path=$model_path `
            --inference_pipe=dt4h `
            --corpus_inference=$corpus_path `
            --inference_batch_size=4 `
            --lang=multi `
            --trust_remote_code `
            --inference_stride=128 `
            --output_file_prefix=$output_prefix

        if ($LASTEXITCODE -ne 0) {
            $err = "[$(Get-Date)] ERROR for language: $lang (exit code $LASTEXITCODE)"
            Write-Warning $err
            Add-Content $logFile $err
            continue   # move to next language instead of stopping
        }

        $done = "[$(Get-Date)] Finished language: $lang"
        Write-Host $done
        Add-Content $logFile $done
    }
    catch {
        $err = "[$(Get-Date)] EXCEPTION for language: $lang - $_"
        Write-Error $err
        Add-Content $logFile $err
        continue
    }
}

# Common arguments
$model_path = "T:\laupodteam\AIOS\Bram\language_modeling\Models\language_models\MultiClinAI\DeBERTa\multiclass_3ldense_20epochs_disease\fold_0"
$base_corpus_path = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined"
$filter_file = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined\batch_1_ids.txt"

# Languages
$languages = @( "nl", "sv", "ro", "it", "es", "en", "cz")

# Log file
$logFile = "real_inference_log.txt"

foreach ($lang in $languages) {
    try {
        $corpus_path = Join-Path $base_corpus_path "MultiClinNER-$lang\test_silver\disease\txt"
        $output_prefix = "DISEASE_DeBERTa_multiclass_$lang"

        $msg = "[$(Get-Date)] Starting language: $lang from $corpus_path"
        Write-Host $msg
        Add-Content $logFile $msg

        poetry run python -m cardioner.main `
            --inference_only `
            --model_path=$model_path `
            --inference_pipe=dt4h `
            --corpus_inference=$corpus_path `
            --inference_batch_size=4 `
            --lang=multi `
            --trust_remote_code `
            --inference_stride=128 `
            --output_file_prefix=$output_prefix

        if ($LASTEXITCODE -ne 0) {
            $err = "[$(Get-Date)] ERROR for language: $lang (exit code $LASTEXITCODE)"
            Write-Warning $err
            Add-Content $logFile $err
            continue   # move to next language instead of stopping
        }

        $done = "[$(Get-Date)] Finished language: $lang"
        Write-Host $done
        Add-Content $logFile $done
    }
    catch {
        $err = "[$(Get-Date)] EXCEPTION for language: $lang - $_"
        Write-Error $err
        Add-Content $logFile $err
        continue
    }
}

Write-Host "All jobs completed. Press any key to exit..."
Pause