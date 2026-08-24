param(
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputPath) {
    $OutputPath = Join-Path $ProjectRoot "exports\colab\clasificador-imagenes-colab.zip"
}

$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$OutputDir = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$StagingRoot = Join-Path $env:TEMP "clasificador-imagenes-colab-source"
if (Test-Path -LiteralPath $StagingRoot) {
    Remove-Item -LiteralPath $StagingRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $StagingRoot | Out-Null

$PathsToCopy = @(
    ".gitignore",
    "README.md",
    "SETUP.md",
    "AUDITORIA.md",
    "backend\app",
    "backend\manage_datasets.py",
    "backend\requirements.txt",
    "backend\requirements-ml.txt",
    "docs",
    "frontend",
    "ml\__init__.py",
    "ml\registry.json",
    "ml\datasets\registry.json",
    "ml\evaluation\validate_torchvision_detectors.py",
    "ml\inference",
    "ml\training",
    "scripts"
)

foreach ($relativePath in $PathsToCopy) {
    $source = Join-Path $ProjectRoot $relativePath
    if (-not (Test-Path -LiteralPath $source)) {
        continue
    }

    $destination = Join-Path $StagingRoot $relativePath
    $destinationParent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
}

$NoisePatterns = @("node_modules", ".venv", ".venv-ml", "__pycache__", ".pytest_cache", ".vite", "dist")
foreach ($pattern in $NoisePatterns) {
    Get-ChildItem -LiteralPath $StagingRoot -Recurse -Force -Directory -Filter $pattern -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force
    }
}

Get-ChildItem -LiteralPath $StagingRoot -Recurse -Force -File -Include "*.pyc", "yolo*.pt" -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force
}

if (Test-Path -LiteralPath $OutputPath) {
    Remove-Item -LiteralPath $OutputPath -Force
}
$itemsToZip = Get-ChildItem -LiteralPath $StagingRoot -Force
if (-not $itemsToZip) {
    throw "No hay archivos para empaquetar en $StagingRoot"
}
Compress-Archive -Path $itemsToZip.FullName -DestinationPath $OutputPath -Force

Write-Host "ZIP Colab creado: $OutputPath"
Write-Host "Este ZIP excluye modelos, datasets crudos, runs, dependencias y caches."


