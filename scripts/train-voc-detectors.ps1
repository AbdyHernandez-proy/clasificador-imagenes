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
    [string]$VenvPath = "",
    [switch]$Resume,
    [switch]$PublishPartial
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TrainScript = Join-Path $ProjectRoot "scripts\train-local-models.ps1"

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

& $TrainScript @CallParams
