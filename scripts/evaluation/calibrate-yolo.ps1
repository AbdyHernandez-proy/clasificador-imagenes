param(
    [int]$Limit = 120,
    [int]$VisualLimit = 20,
    [int]$ImageSize = 640,
    [string]$Device = "auto",
    [string]$OutputDir = "",
    [string]$Combos = "",
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $VenvPath) {
    $VenvPath = Join-Path $env:USERPROFILE ".ciml\venv"
}

$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonPath)) {
    $PythonPath = "python"
}

$ArgsList = @(
    "-m", "ml.evaluation.calibrate_ultralytics_direct",
    "--model", "custom-yolo-v8-v11-detector",
    "--dataset", "coco128-detect",
    "--limit", "$Limit",
    "--visual-limit", "$VisualLimit",
    "--imgsz", "$ImageSize",
    "--device", $Device
)

if ($OutputDir) {
    $ArgsList += @("--output-dir", $OutputDir)
}

if ($Combos) {
    $ArgsList += @("--combos", $Combos)
}

Set-Location $ProjectRoot
& $PythonPath @ArgsList
