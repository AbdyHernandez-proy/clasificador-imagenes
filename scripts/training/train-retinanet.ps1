param(
    [int]$Epochs = 6,
    [switch]$Resume,
    [switch]$Install
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$TrainScript = Join-Path $ProjectRoot "scripts\training\train-voc-detectors.ps1"

$ArgsList = @{
    Models = "retinanet"
    Epochs = $Epochs
    Batch = 1
    ImageSize = 512
    SessionSamples = 1024
    SessionCount = 64
    LogEvery = 10
    CheckpointEvery = 25
    Workers = 0
    Device = "auto"
    LearningRate = 0.0005
    Momentum = 0.9
    WeightDecay = 0.0005
    LrStepSize = 2
    LrGamma = 0.5
    FreezeBackboneEpochs = 1
    ValidationLimit = 500
    ValidationConfidence = 0.60
    ValidationMaxDetections = 8
    ValidationVisualLimit = 16
    BestMetric = "precision_at_50"
    MinDelta = 0.0005
    EarlyStoppingPatience = 3
    PublishPartial = $true
    ValidateEveryEpoch = $true
}

if ($Resume) {
    $ArgsList["Resume"] = $true
}

if ($Install) {
    $ArgsList["Install"] = $true
}

& $TrainScript @ArgsList
