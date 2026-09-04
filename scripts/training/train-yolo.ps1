param(
    [int]$Epochs = 25,
    [int]$Batch = 1,
    [int]$ImageSize = 640,
    [int]$Workers = 0,
    [string]$Device = "auto",
    [string]$Dataset = "coco128-detect",
    [string]$VenvPath = "",
    [switch]$Install
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$TrainScript = Join-Path $ProjectRoot "scripts\training\train-local-models.ps1"

$CallParams = @{
    Models = "yolo"
    Dataset = $Dataset
    Epochs = $Epochs
    Batch = $Batch
    ImageSize = $ImageSize
    Workers = $Workers
    Device = $Device
}

if ($VenvPath) {
    $CallParams["VenvPath"] = $VenvPath
}

if ($Install) {
    $CallParams["Install"] = $true
}

& $TrainScript @CallParams
