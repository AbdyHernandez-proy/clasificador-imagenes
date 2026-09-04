param(
    [string]$DatasetId = "voc-detect",
    [string]$VenvPath = "",
    [double]$TrainRatio = 0.6,
    [int]$SplitSeed = 20260827,
    [switch]$Force,
    [switch]$SkipExisting,
    [switch]$KeepArchives
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
    "-m", "ml.training.download_datasets",
    "--dataset-id", $DatasetId,
    "--train-ratio", "$TrainRatio",
    "--split-seed", "$SplitSeed"
)

if ($Force) {
    $ArgsList += "--force"
}

if ($SkipExisting) {
    $ArgsList += "--skip-existing"
}

if ($KeepArchives) {
    $ArgsList += "--keep-archives"
}

Set-Location $ProjectRoot
& $PythonPath @ArgsList
