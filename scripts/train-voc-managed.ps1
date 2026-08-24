param(
    [string]$Models = "faster-rcnn,retinanet",
    [int]$Epochs = 3,
    [int]$Batch = 1,
    [int]$ImageSize = 512,
    [int]$SessionSamples = 512,
    [int]$Sessions = 1,
    [int]$LogEvery = 10,
    [int]$CheckpointEvery = 0,
    [int]$Workers = 0,
    [string]$Device = "auto",
    [string]$VenvPath = "",
    [switch]$PublishPartial
)

$ErrorActionPreference = "Stop"

$TrainVocScript = Join-Path $PSScriptRoot "train-voc-detectors.ps1"

$CallParams = @{
    Models = $Models
    Epochs = $Epochs
    Batch = $Batch
    ImageSize = $ImageSize
    SessionSamples = $SessionSamples
    SessionCount = $Sessions
    LogEvery = $LogEvery
    CheckpointEvery = $CheckpointEvery
    Workers = $Workers
    Device = $Device
    Resume = $true
}

if ($VenvPath) {
    $CallParams["VenvPath"] = $VenvPath
}

if ($PublishPartial) {
    $CallParams["PublishPartial"] = $true
}

Write-Host ""
Write-Host "=== Ejecutando $Sessions sesiones dentro de un unico proceso Python ==="
& $TrainVocScript @CallParams

Write-Host ""
Write-Host "Sesiones finalizadas. Revisa el progreso con:"
Write-Host ".\scripts\show-training-progress.ps1 -Model custom-faster-rcnn-detector"
Write-Host ".\scripts\show-training-progress.ps1 -Model custom-retinanet-detector"
