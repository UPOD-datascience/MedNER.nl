$batches = @("b1", "b2")
$langs   = @("cz", "sv", "en", "nl", "ro", "it", "es")

$base = "D:\Temp\dt4h_cardioccc_ann-transfer_b1+b2_bundle_v1.1\dt4h_cardioccc_ann-transfer_b1+b2_bundle_v1.1"
$errors = @()

foreach ($batch in $batches) {
    foreach ($lang in $langs) {
            $annDir  = "$base\$batch\1_validated_without_sugs\$lang"
            $outPath = "$base\$batch\1_validated_without_sugs\$lang\collection_$lang.json"

            Write-Host "Running for batch=$batch lang=$lang cat=$cat"

            poetry run python ../src/pubscience/ner_caster.py `
                --json_dir="$annDir" `
                --collect_json `
                --out_path="$outPath"

            if ($LASTEXITCODE -ne 0) {
                $msg = "Failed at batch=$batch lang=$lang cat=$cat (exit code $LASTEXITCODE)"
                Write-Warning $msg
                $errors += $msg
                continue
            }
    }
}

Write-Host ""
Write-Host "Done."

if ($errors.Count -gt 0) {
    Write-Host "There were $($errors.Count) failed run(s):" -ForegroundColor Yellow
    $errors | ForEach-Object { Write-Host " - $_" -ForegroundColor Yellow }
} else {
    Write-Host "All runs succeeded." -ForegroundColor Green
}

Read-Host "Press Enter to close"
