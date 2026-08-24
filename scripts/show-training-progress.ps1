param(
    [string]$Model = "custom-faster-rcnn-detector"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProgressPath = Join-Path $ProjectRoot "ml\models\$Model\training_progress.json"
$EventsPath = Join-Path $ProjectRoot "ml\models\$Model\training_events.jsonl"

if (-not (Test-Path -LiteralPath $ProgressPath)) {
    Write-Host "No hay progreso registrado para $Model."
    exit 0
}

$Progress = Get-Content -LiteralPath $ProgressPath -Raw | ConvertFrom-Json
$State = $Progress.state
$CompletedEpochs = [int]$State.current_epoch
$TargetEpochs = [int]$State.target_epochs
$DisplayEpoch = if ($CompletedEpochs -ge $TargetEpochs) { $TargetEpochs } else { $CompletedEpochs + 1 }
$NextSample = [int]$State.next_sample_index
$TotalSamples = [int]$State.total_samples
$EpochPercent = 0
if ($TotalSamples -gt 0) {
    $EpochPercent = [math]::Round(($NextSample / $TotalSamples) * 100, 2)
}

Write-Host "Modelo: $($State.model_id)"
Write-Host "Estado: $($State.status)"
Write-Host "Epoca: $DisplayEpoch / $TargetEpochs"
Write-Host "Epocas completadas: $CompletedEpochs / $TargetEpochs"
Write-Host "Avance de epoca: $NextSample / $TotalSamples ($EpochPercent%)"
Write-Host "Ultima perdida: $($State.last_loss)"
Write-Host "Ultimo lote: $($State.last_batch)"
Write-Host "Ultima actualizacion: $($State.last_update)"

if ($Progress.last_event) {
    Write-Host ""
    Write-Host "Ultimo evento:"
    $Progress.last_event | Format-List
}

if (Test-Path -LiteralPath $EventsPath) {
    Write-Host ""
    Write-Host "Eventos recientes:"
    Get-Content -LiteralPath $EventsPath -Tail 5
}
