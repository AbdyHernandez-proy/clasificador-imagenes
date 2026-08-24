param(
    [string]$VenvPath = "",
    [switch]$Force,
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $VenvPath) {
    $VenvPath = Join-Path $env:USERPROFILE ".ciml\venv"
}

$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "No existe el entorno ML en $VenvPath. Ejecuta primero: .\scripts\train-local-models.ps1 -Install -Models yolo"
}

Set-Location $ProjectRoot

$ArgsList = @("-m", "ml.training.prepare_voc")
if ($Force) {
    $ArgsList += "--force"
}
if ($SkipDownload) {
    $ArgsList += "--skip-download"
}

& $PythonPath @ArgsList
