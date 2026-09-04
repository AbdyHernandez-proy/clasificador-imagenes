param(
    [Parameter(Mandatory = $true)]
    [string]$Image,

    [string]$OutputDir = "",
    [double]$Conf = -1,
    [double]$Iou = -1,
    [int]$MaxDetections = 0,
    [int]$ImageSize = 0,
    [string]$Device = "auto",
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
    "-m", "ml.inference.predict_ultralytics_image",
    "--model", "custom-detr-rtdetr-detector",
    "--image", $Image,
    "--device", $Device
)

if ($OutputDir) {
    $ArgsList += @("--output-dir", $OutputDir)
}

if ($Conf -ge 0) {
    $ArgsList += @("--conf", "$Conf")
}

if ($Iou -ge 0) {
    $ArgsList += @("--iou", "$Iou")
}

if ($MaxDetections -gt 0) {
    $ArgsList += @("--max-det", "$MaxDetections")
}

if ($ImageSize -gt 0) {
    $ArgsList += @("--imgsz", "$ImageSize")
}

Set-Location $ProjectRoot
& $PythonPath @ArgsList
