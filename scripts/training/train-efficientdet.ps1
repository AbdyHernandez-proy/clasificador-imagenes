param(
    [int]$Epochs = 6,
    [int]$Batch = 1,
    [int]$ImageSize = 512,
    [int]$Workers = 0,
    [string]$Device = "auto",
    [int]$MaxSamples = 0,
    [int]$SessionSamples = 1024,
    [int]$SessionCount = 64,
    [int]$LogEvery = 10,
    [int]$CheckpointEvery = 25,
    [double]$LearningRate = 0.0005,
    [double]$WeightDecay = 0.0005,
    [int]$LrStepSize = 2,
    [double]$LrGamma = 0.5,
    [int]$ValidationLimit = 500,
    [double]$ValidationConfidence = 0.5,
    [int]$ValidationMaxDetections = 8,
    [int]$ValidationVisualLimit = 16,
    [string]$BestMetric = "precision_at_50",
    [double]$MinDelta = 0.0005,
    [int]$EarlyStoppingPatience = 3,
    [string]$BaseModel = "tf_efficientdet_d0",
    [string]$VenvPath = "",
    [switch]$Resume,
    [switch]$PublishPartial,
    [switch]$ValidateBeforeTraining,
    [switch]$NoPretrainedBackbone,
    [switch]$PreflightOnly,
    [switch]$Install,
    [switch]$TrainAfterInstall
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$TrainScript = Join-Path $ProjectRoot "scripts\training\train-local-models.ps1"
if (-not $VenvPath) {
    $VenvPath = Join-Path $env:USERPROFILE ".ciml\venv"
}
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$CacheRoot = Join-Path $VenvPath "cache"
$env:HF_HOME = Join-Path $CacheRoot "huggingface"
$env:TORCH_HOME = Join-Path $CacheRoot "torch"
$env:ULTRALYTICS_SETTINGS = Join-Path $CacheRoot "ultralytics"
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:TORCH_HOME, $env:ULTRALYTICS_SETTINGS | Out-Null

$CallParams = @{
    Models = "efficientdet"
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
    WeightDecay = $WeightDecay
    LrStepSize = $LrStepSize
    LrGamma = $LrGamma
    ValidationLimit = $ValidationLimit
    ValidationConfidence = $ValidationConfidence
    ValidationMaxDetections = $ValidationMaxDetections
    ValidationVisualLimit = $ValidationVisualLimit
    BestMetric = $BestMetric
    MinDelta = $MinDelta
    EarlyStoppingPatience = $EarlyStoppingPatience
    EfficientDetBaseModel = $BaseModel
    PublishPartial = $true
    ValidateEveryEpoch = $true
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

if ($ValidateBeforeTraining) {
    $CallParams["ValidateBeforeTraining"] = $true
}

if (-not $NoPretrainedBackbone) {
    $CallParams["EfficientDetPretrainedBackbone"] = $true
}

if ($Install) {
    $CallParams["Install"] = $true
}

if ($Install -and -not $TrainAfterInstall) {
    & $TrainScript @CallParams -InstallOnly
    Write-Host "Instalacion EfficientDet verificada. No se inicio entrenamiento."
    return
}

if ($PreflightOnly) {
    Set-Location $ProjectRoot
    Write-Host "Validando preparacion EfficientDet..."
    & $PythonPath -m ml.training.train_efficientdet_preflight --dataset-id "voc-detect" --imgsz "$ImageSize" --base-model $BaseModel
    return
}

& $TrainScript @CallParams
