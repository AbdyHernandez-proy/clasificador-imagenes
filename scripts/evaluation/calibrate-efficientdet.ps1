param(
    [int]$Limit = 120,
    [int]$VisualLimit = 20,
    [int]$ImageSize = 512,
    [string]$Device = "auto",
    [string]$OutputDir = "",
    [string]$Combos = "",
    [string]$Artifact = "",
    [string]$BaseModel = "",
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
    "-m", "ml.evaluation.calibrate_efficientdet_direct",
    "--model", "custom-efficientdet-detector",
    "--dataset", "voc-detect",
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

if ($Artifact) {
    $ArgsList += @("--artifact", $Artifact)
}

if ($BaseModel) {
    $ArgsList += @("--base-model", $BaseModel)
}

Set-Location $ProjectRoot
& $PythonPath @ArgsList
