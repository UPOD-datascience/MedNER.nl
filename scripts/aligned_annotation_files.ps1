$langs   = @("cz", "sv", "en", "nl", "ro", "it", "es")

$errors = @()

foreach ($lang in $langs) {
            $annDir  = "..\assets\MultiClinNER_combined\MultiClinNER-$lang"
            $outPath = "..\assets\MultiClinNER_combined\MultiClinNER-$lang\merged_$lang.json"
			$reportPath = "..\assets\MultiClinNER_combined\MultiClinNER-$lang\merged_report_$lang.json"

            Write-Host "Running for lang=$lang"

            poetry run python merge_by_canonical.py `
                --json_dir="$annDir" `
                --out_path="$outPath" `
                --report_path="$reportPath" `
				--similarity_threshold=0.75

            if ($LASTEXITCODE -ne 0) {
                $msg = "Failed at batch=$batch lang=$lang (exit code $LASTEXITCODE)"
                Write-Warning $msg
                $errors += $msg
                continue
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
