param(
    [int]$Epochs = 10,
    [int]$Batch = 1,
    [int]$ImageSize = 512,
    [int]$Workers = 0,
    [string]$Device = "auto",
    [int]$SessionSamples = 0,
    [int]$SessionCount = 1,
    [int]$LogEvery = 10,
    [int]$CheckpointEvery = 0,
    [string]$VenvPath = "",
    [switch]$Resume,
    [switch]$PublishPartial,
    [switch]$Install,
    [switch]$TrainAfterInstall
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$TrainScript = Join-Path $ProjectRoot "scripts\training\train-local-models.ps1"

$CallParams = @{
    Models = "rtdetr"
    Dataset = "voc-detect"
    Epochs = $Epochs
    Batch = $Batch
    ImageSize = $ImageSize
    Workers = $Workers
    Device = $Device
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

if ($Install) {
    $CallParams["Install"] = $true
}

if ($Install -and -not $TrainAfterInstall) {
    & $TrainScript @CallParams -InstallOnly
    Write-Host "Instalacion RT-DETR verificada. No se inicio entrenamiento."
    return
}

& $TrainScript @CallParams
