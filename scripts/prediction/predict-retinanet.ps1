param(
    [Parameter(Mandatory = $true)]
    [string]$Image,

    [string]$OutputDir = "",
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
    "-m", "ml.inference.predict_torchvision_image",
    "--image", $Image,
    "--model", "custom-retinanet-detector"
)

if ($OutputDir) {
    $ArgsList += @("--output-dir", $OutputDir)
}

Set-Location $ProjectRoot
& $PythonPath @ArgsList
