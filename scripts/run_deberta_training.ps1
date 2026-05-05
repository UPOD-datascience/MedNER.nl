$ErrorActionPreference = "Continue"

$output_path = "T:\laupodteam\AIOS\Bram\language_modeling\Models\language_models\MultiClinAI\DeBERTa\multilabel_3ldense_20epochs_NL"
$base_corpus_path = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined\MultiClinNER-nl"

$logFile = "training_log.txt"

try {
	$corpus_path = Join-Path $base_corpus_path "all/all_medmention_strict.json"
	$output_dir = $output_path

	$msg = "[$(Get-Date)] Starting multilabel Dutch training, at $corpus_path"
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
		--num_epochs=20 `
		--learning_rate=2e-5 `
		--weight_decay=1e-4 `
		--accumulation_steps=2 `
		--force_splitter `
		--num_splits=20 `
		--use_class_weights `
		--classifier_hidden_layers 768 768 768 `
		--only_first_split `
		--entity_types PROCEDURE SYMPTOM DISEASE

	if ($LASTEXITCODE -ne 0) {
		$err = "[$(Get-Date)] ERROR (exit code $LASTEXITCODE)"
		Write-Warning $err
		Add-Content -Path $logFile -Value $err
		continue
	}

	$done = "[$(Get-Date)] Finished"
	Write-Host $done
	Add-Content -Path $logFile -Value $done
}
catch {
	$err = "[$(Get-Date)] EXCEPTION - $_"
	Write-Error $err
	Add-Content -Path $logFile -Value $err
	continue
}

$output_path = "T:\laupodteam\AIOS\Bram\language_modeling\Models\language_models\MultiClinAI\DeBERTa\multilabel_3ldense_20epochs_MULTI"
$base_corpus_path = "T:\laupodteam\AIOS\Bram\notebooks\code_dev\CardioNER.nl\assets\MultiClinNER_combined\MultiClinNER-multi"

$logFile = "training_log.txt"

try {
	$corpus_path = Join-Path $base_corpus_path "collected_all_strict_medmention.json"
	$output_dir = $output_path

	$msg = "[$(Get-Date)] Starting multilabel multilingual training, at $corpus_path"
	Write-Host $msg
	Add-Content -Path $logFile -Value $msg

	poetry run python -m cardioner.main `
		--train_model `
		--model="UMCU/CardioDeBERTa.nl_clinical" `
		--corpus_train=$corpus_path `
		--parse_annotations `
		--lang=multi `
		--trust_remote_code `
		--max_token_length=128 `
		--chunk_size=128 `
		--batch_size=8 `
		--chunk_type=centered `
		--output_dir=$output_dir `
		--num_epochs=20 `
		--learning_rate=2e-5 `
		--weight_decay=1e-4 `
		--accumulation_steps=2 `
		--force_splitter `
		--num_splits=20 `
		--use_class_weights `
		--classifier_hidden_layers 768 768 768 `
		--only_first_split `
		--entity_types PROCEDURE SYMPTOM DISEASE

	if ($LASTEXITCODE -ne 0) {
		$err = "[$(Get-Date)] ERROR (exit code $LASTEXITCODE)"
		Write-Warning $err
		Add-Content -Path $logFile -Value $err
		continue
	}

	$done = "[$(Get-Date)] Finished"
	Write-Host $done
	Add-Content -Path $logFile -Value $done
}
catch {
	$err = "[$(Get-Date)] EXCEPTION - $_"
	Write-Error $err
	Add-Content -Path $logFile -Value $err
	continue
}


Write-Host "All jobs completed. Press any key to exit..."
Pause