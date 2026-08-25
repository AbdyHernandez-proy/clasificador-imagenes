param(
    [string]$Model = "custom-faster-rcnn-detector"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProgressPath = Join-Path $ProjectRoot "ml\models\$Model\training_progress.json"
$EventsPath = Join-Path $ProjectRoot "ml\models\$Model\training_events.jsonl"
$ValidationPath = Join-Path $ProjectRoot "ml\models\$Model\validation_history.json"

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

if ($State.best_metric) {
    Write-Host "Mejor metrica: $($State.best_metric)"
    Write-Host "Mejor valor: $($State.best_metric_value)"
    Write-Host "Mejor epoca: $($State.best_epoch)"
}

if ($State.epochs_without_improvement -ne $null) {
    Write-Host "Epocas sin mejora: $($State.epochs_without_improvement)"
}

if ($State.early_stopped) {
    Write-Host "Early stopping: activo en epoca $($State.early_stopped_epoch)"
}

if ($State.best_checkpoint_path) {
    Write-Host "Best checkpoint: $($State.best_checkpoint_path)"
}

if ($Progress.last_event) {
    Write-Host ""
    Write-Host "Ultimo evento:"
    $Progress.last_event | Format-List
}

if (Test-Path -LiteralPath $ValidationPath) {
    $ValidationHistory = Get-Content -LiteralPath $ValidationPath -Raw | ConvertFrom-Json
    $LastValidation = @($ValidationHistory)[-1]
    if ($LastValidation) {
        $Summary = $LastValidation.summary
        $Map50 = if ($Summary) { $Summary.map50 } else { $LastValidation.map50 }
        $Map5095 = if ($Summary) { $Summary.map50_95 } else { $LastValidation.map50_95 }
        $Precision = if ($Summary) { $Summary.precision_at_50 } else { $LastValidation.precision_at_50 }
        $Recall = if ($Summary) { $Summary.recall_at_50 } else { $LastValidation.recall_at_50 }
        $TP = if ($Summary) { $Summary.true_positives_at_50 } else { $LastValidation.true_positives_at_50 }
        $FP = if ($Summary) { $Summary.false_positives_at_50 } else { $LastValidation.false_positives_at_50 }
        $FN = if ($Summary) { $Summary.false_negatives_at_50 } else { $LastValidation.false_negatives_at_50 }

        Write-Host ""
        Write-Host "Ultima validacion:"
        Write-Host "Epoca: $($LastValidation.epoch)"
        Write-Host "mAP@50: $Map50"
        Write-Host "mAP@50:95: $Map5095"
        Write-Host "Precision@50: $Precision"
        Write-Host "Recall@50: $Recall"
        Write-Host "TP/FP/FN: $TP / $FP / $FN"
    }
}

if (Test-Path -LiteralPath $EventsPath) {
    Write-Host ""
    Write-Host "Eventos recientes:"
    Get-Content -LiteralPath $EventsPath -Tail 5
}

