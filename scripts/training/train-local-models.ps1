param(
    [string]$Models = "yolo,faster-rcnn,retinanet,rtdetr",
    [string]$Dataset = "coco128-detect",
    [int]$Epochs = 3,
    [int]$Batch = 1,
    [int]$ImageSize = 512,
    [string]$Device = "auto",
    [int]$MaxSamples = 0,
    [int]$SessionSamples = 0,
    [int]$SessionCount = 1,
    [int]$LogEvery = 25,
    [int]$CheckpointEvery = 0,
    [int]$Workers = 0,
    [double]$LearningRate = 0.0025,
    [double]$Momentum = 0.9,
    [double]$WeightDecay = 0.0005,
    [int]$LrStepSize = 0,
    [double]$LrGamma = 0.1,
    [int]$FreezeBackboneEpochs = 0,
    [int]$ValidationLimit = 250,
    [double]$ValidationConfidence = -1,
    [int]$ValidationMaxDetections = 0,
    [int]$ValidationVisualLimit = 12,
    [string]$BestMetric = "map50_95",
    [double]$MinDelta = 0.0001,
    [int]$EarlyStoppingPatience = 0,
    [string]$VenvPath = "",
    [switch]$Resume,
    [switch]$PublishPartial,
    [switch]$ValidateEveryEpoch,
    [switch]$ValidateBeforeTraining,
    [switch]$AllowEfficientDet,
    [string]$EfficientDetBaseModel = "tf_efficientdet_d0",
    [switch]$EfficientDetPretrainedBackbone,
    [switch]$Install,
    [switch]$InstallOnly
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $VenvPath) {
    $VenvPath = Join-Path $env:USERPROFILE ".ciml\venv"
}
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$CacheRoot = Join-Path $VenvPath "cache"
$env:HF_HOME = Join-Path $CacheRoot "huggingface"
$env:TORCH_HOME = Join-Path $CacheRoot "torch"
$env:ULTRALYTICS_SETTINGS = Join-Path $CacheRoot "ultralytics"
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:TORCH_HOME, $env:ULTRALYTICS_SETTINGS | Out-Null

Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-Host "Creando entorno local de entrenamiento en $VenvPath..."
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        py -3 -m venv $VenvPath
    } else {
        python -m venv $VenvPath
    }
}

if ($Install) {
    Write-Host "Instalando dependencias base de entrenamiento..."
    & $PythonPath -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "No se pudo actualizar pip." }

    & $PythonPath -m pip install pillow PyYAML numpy
    if ($LASTEXITCODE -ne 0) { throw "No se pudieron instalar dependencias base." }

    & $PythonPath -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { throw "No se pudo instalar PyTorch/torchvision. Revisa si Windows Long Paths esta habilitado o usa -VenvPath con una ruta mas corta." }

    & $PythonPath -m pip install ultralytics
    if ($LASTEXITCODE -ne 0) { throw "No se pudo instalar Ultralytics." }

    if ($Models -match "(^|,)\s*(efficientdet)" -or $Models -eq "all") {
        & $PythonPath -m pip install effdet timm pycocotools
        if ($LASTEXITCODE -ne 0) { throw "No se pudieron instalar dependencias EfficientDet." }
    }

    Write-Host "Dependencias instaladas y verificadas."
    if ($InstallOnly) {
        return
    }
}

if ($Models -match "(^|,)\s*(yolo|rtdetr|detr)" ) {
    & $PythonPath -c "import ultralytics" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Falta Ultralytics en el entorno ML. Ejecuta: .\scripts\training\train-local-models.ps1 -Install -Models yolo"
    }
}

if ($Models -match "(^|,)\s*(efficientdet)" -or $Models -eq "all") {
    & $PythonPath -c "import effdet, timm, pycocotools" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Faltan dependencias EfficientDet. Ejecuta: .\scripts\training\train-local-models.ps1 -Install -Models efficientdet"
    }
}

$ArgsList = @(
    "-m", "ml.training.local_train",
    "--models", $Models,
    "--dataset-id", $Dataset,
    "--epochs", "$Epochs",
    "--batch", "$Batch",
    "--imgsz", "$ImageSize",
    "--device", $Device,
    "--max-samples", "$MaxSamples",
    "--session-samples", "$SessionSamples",
    "--session-count", "$SessionCount",
    "--log-every", "$LogEvery",
    "--checkpoint-every", "$CheckpointEvery",
    "--workers", "$Workers",
    "--lr", "$LearningRate",
    "--momentum", "$Momentum",
    "--weight-decay", "$WeightDecay",
    "--lr-step-size", "$LrStepSize",
    "--lr-gamma", "$LrGamma",
    "--freeze-backbone-epochs", "$FreezeBackboneEpochs",
    "--validation-limit", "$ValidationLimit",
    "--validation-confidence", "$ValidationConfidence",
    "--validation-max-detections", "$ValidationMaxDetections",
    "--validation-visual-limit", "$ValidationVisualLimit",
    "--best-metric", $BestMetric,
    "--min-delta", "$MinDelta",
    "--early-stopping-patience", "$EarlyStoppingPatience",
    "--efficientdet-base-model", $EfficientDetBaseModel
)

if ($Resume) {
    $ArgsList += "--resume"
}

if ($PublishPartial) {
    $ArgsList += "--publish-partial"
}

if ($ValidateEveryEpoch) {
    $ArgsList += "--validate-every-epoch"
}

if ($ValidateBeforeTraining) {
    $ArgsList += "--validate-before-training"
}

if ($AllowEfficientDet) {
    $ArgsList += "--allow-efficientdet"
}

if ($EfficientDetPretrainedBackbone) {
    $ArgsList += "--efficientdet-pretrained-backbone"
}

$env:PYTHONUNBUFFERED = "1"
Write-Host "Ejecutando entrenamiento local..."
& $PythonPath @ArgsList
