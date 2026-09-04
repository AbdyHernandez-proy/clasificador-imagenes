param(
    [string]$Models = "faster-rcnn,retinanet",
    [int]$Epochs = 10,
    [int]$Batch = 1,
    [int]$ImageSize = 640,
    [int]$Workers = 0,
    [string]$Device = "auto",
    [int]$MaxSamples = 0,
    [int]$SessionSamples = 512,
    [int]$SessionCount = 1,
    [int]$LogEvery = 10,
    [int]$CheckpointEvery = 0,
    [double]$LearningRate = 0.001,
    [double]$Momentum = 0.9,
    [double]$WeightDecay = 0.0005,
    [int]$LrStepSize = 2,
    [double]$LrGamma = 0.5,
    [int]$FreezeBackboneEpochs = 1,
    [int]$ValidationLimit = 250,
    [double]$ValidationConfidence = -1,
    [int]$ValidationMaxDetections = 0,
    [int]$ValidationVisualLimit = 12,
    [string]$BestMetric = "map50_95",
    [double]$MinDelta = 0.0001,
    [int]$EarlyStoppingPatience = 2,
    [string]$VenvPath = "",
    [switch]$Resume,
    [switch]$PublishPartial,
    [switch]$ValidateEveryEpoch,
    [switch]$ValidateBeforeTraining,
    [switch]$Install
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$TrainScript = Join-Path $ProjectRoot "scripts\training\train-local-models.ps1"

$CallParams = @{
    Models = $Models
    Dataset = "voc-detect"
    Epochs = $Epochs
    Batch = $Batch
    ImageSize = $ImageSize
    Workers = $Workers
    Device = $Device
    MaxSamples = $MaxSamples
    SessionSamples = $SessionSamples
    SessionCount = $SessionCount
    LogEvery = $LogEvery
    CheckpointEvery = $CheckpointEvery
    LearningRate = $LearningRate
    Momentum = $Momentum
    WeightDecay = $WeightDecay
    LrStepSize = $LrStepSize
    LrGamma = $LrGamma
    FreezeBackboneEpochs = $FreezeBackboneEpochs
    ValidationLimit = $ValidationLimit
    ValidationConfidence = $ValidationConfidence
    ValidationMaxDetections = $ValidationMaxDetections
    ValidationVisualLimit = $ValidationVisualLimit
    BestMetric = $BestMetric
    MinDelta = $MinDelta
    EarlyStoppingPatience = $EarlyStoppingPatience
}

if ($VenvPath) {
    $CallParams["VenvPath"] = $VenvPath
}

if ($Resume) {
    $CallParams["Resume"] = $true
}

if ($PublishPartial) {
    $CallParams["PublishPartial"] = $true
}

if ($ValidateEveryEpoch) {
    $CallParams["ValidateEveryEpoch"] = $true
}

if ($ValidateBeforeTraining) {
    $CallParams["ValidateBeforeTraining"] = $true
}

if ($Install) {
    $CallParams["Install"] = $true
}

& $TrainScript @CallParams
