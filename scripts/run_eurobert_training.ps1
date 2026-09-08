$ErrorActionPreference = "Continue"

$output_path = "T:\laupodteam\AIOS\Bram\language_modeling\Models\language_models\MutliClinAI\XLMR\multilabel_3ldense_10epochs_40splits"
$base_corpus_path = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined\MultiClinNER-multi"

$logFile = "training_log.txt"

try {
	$corpus_path = Join-Path $base_corpus_path "merged_all.json"
	$output_dir = "${output_path}"

	$msg = "[$(Get-Date)] Starting at $corpus_path"
	Write-Host $msg
	Add-Content -Path $logFile -Value $msg

	poetry run python -m cardioner.main `
		--train_model `
		--model="FacebookAI/xlm-roberta-large" `
		--corpus_train=$corpus_path `
		--parse_annotations `
		--lang=multi `
		--trust_remote_code `
		--max_token_length=128 `
		--chunk_size=128 `
		--batch_size=8 `
		--chunk_type=centered `
		--output_dir=$output_dir `
		--num_epochs=10 `
		--learning_rate=1e-5 `
		--weight_decay=1e-4 `
		--accumulation_steps=2 `
		--force_splitter `
		--use_class_weights `
		--num_splits=40 `
		--only_first_split `
		--classifier_hidden_layers 768 768 768 `
		--entity_types PROCEDURE DISEASE SYMPTOM MEDICATION

	if ($LASTEXITCODE -ne 0) {
		$err = "[$(Get-Date)] ERROR: (exit code $LASTEXITCODE)"
		Write-Warning $err
		Add-Content -Path $logFile -Value $err
		continue
	}

	$done = "[$(Get-Date)] Finished"
	Write-Host $done
	Add-Content -Path $logFile -Value $done
}
catch {
	$err = "[$(Get-Date)] EXCEPTION"
	Write-Error $err
	Add-Content -Path $logFile -Value $err
	continue
}


Write-Host "All jobs completed. Press any key to exit..."
Pause
