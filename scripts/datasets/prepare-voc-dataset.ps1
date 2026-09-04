param(
    [string]$VenvPath = "",
    [double]$TrainRatio = 0.6,
    [int]$SplitSeed = 20260827,
    [switch]$Force,
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $VenvPath) {
    $VenvPath = Join-Path $env:USERPROFILE ".ciml\venv"
}

$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "No existe el entorno ML en $VenvPath. Ejecuta primero: .\scripts\training\train-local-models.ps1 -Install -Models yolo"
}

Set-Location $ProjectRoot

$ArgsList = @(
    "-m", "ml.training.prepare_voc",
    "--train-ratio", $TrainRatio,
    "--split-seed", $SplitSeed
)
if ($Force) {
    $ArgsList += "--force"
}
if ($SkipDownload) {
    $ArgsList += "--skip-download"
}

& $PythonPath @ArgsList
