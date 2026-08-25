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
    [double]$LearningRate = 0.001,
    [double]$Momentum = 0.9,
    [double]$WeightDecay = 0.0005,
    [int]$LrStepSize = 2,
    [double]$LrGamma = 0.5,
    [int]$FreezeBackboneEpochs = 1,
    [int]$ValidationLimit = 250,
    [double]$ValidationConfidence = -1,
    [int]$ValidationMaxDetections = 0,
    [string]$BestMetric = "map50_95",
    [double]$MinDelta = 0.0001,
    [int]$EarlyStoppingPatience = 2,
    [string]$VenvPath = "",
    [switch]$PublishPartial,
    [switch]$ValidateEveryEpoch
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
    LearningRate = $LearningRate
    Momentum = $Momentum
    WeightDecay = $WeightDecay
    LrStepSize = $LrStepSize
    LrGamma = $LrGamma
    FreezeBackboneEpochs = $FreezeBackboneEpochs
    ValidationLimit = $ValidationLimit
    ValidationConfidence = $ValidationConfidence
    ValidationMaxDetections = $ValidationMaxDetections
    BestMetric = $BestMetric
    MinDelta = $MinDelta
    EarlyStoppingPatience = $EarlyStoppingPatience
}

if ($VenvPath) {
    $CallParams["VenvPath"] = $VenvPath
}

if ($PublishPartial) {
    $CallParams["PublishPartial"] = $true
}

if ($ValidateEveryEpoch) {
    $CallParams["ValidateEveryEpoch"] = $true
}

Write-Host ""
Write-Host "=== Ejecutando $Sessions sesiones dentro de un unico proceso Python ==="
& $TrainVocScript @CallParams

Write-Host ""
Write-Host "Sesiones finalizadas. Revisa el progreso con:"
Write-Host ".\scripts\show-training-progress.ps1 -Model custom-faster-rcnn-detector"
Write-Host ".\scripts\show-training-progress.ps1 -Model custom-retinanet-detector"
