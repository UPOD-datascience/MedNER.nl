$ErrorActionPreference = "Continue"

$output_path = "T:\laupodteam\AIOS\Bram\language_modeling\Models\language_models\MultiClinAI\DeBERTa\multiclass_3ldense_20epochs_"
$base_corpus_path = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined\MultiClinNER-multi"

$categories = @("procedure", "symptom") #"procedure", "symptom")
$logFile = "inference_log.txt"

foreach ($cat in $categories) {
    try {
        $corpus_path = Join-Path $base_corpus_path "collected_$cat.json"
        $output_dir = "${output_path}${cat}"
        $entity_type = $cat.ToUpperInvariant()

        $msg = "[$(Get-Date)] Starting category: $entity_type, at $corpus_path"
        Write-Host $msg
        Add-Content -Path $logFile -Value $msg

        poetry run python -m cardioner.main `
            --train_model `
            --model="UMCU/CardioDeBERTa.nl_clinical" `
            --corpus_train=$corpus_path `
			--parse_annotations `
            --lang=nl `
            --trust_remote_code `
            --max_token_length=128 `
            --chunk_size=128 `
            --batch_size=8 `
            --chunk_type=centered `
            --output_dir=$output_dir `
            --num_epochs=10 `
            --learning_rate=2e-5 `
            --weight_decay=1e-4 `
            --accumulation_steps=2 `
            --force_splitter `
            --num_splits=20 `
            --use_class_weights `
			--multiclass `
            --classifier_hidden_layers 768 768 768 `
            --only_first_split `
            --entity_types $entity_type

        if ($LASTEXITCODE -ne 0) {
            $err = "[$(Get-Date)] ERROR for category: $cat (exit code $LASTEXITCODE)"
            Write-Warning $err
            Add-Content -Path $logFile -Value $err
            continue
        }

        $done = "[$(Get-Date)] Finished category: $cat"
        Write-Host $done
        Add-Content -Path $logFile -Value $done
    }
    catch {
        $err = "[$(Get-Date)] EXCEPTION for category: $cat - $_"
        Write-Error $err
        Add-Content -Path $logFile -Value $err
        continue
    }
}

Write-Host "All jobs completed. Press any key to exit..."
Pause