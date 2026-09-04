param(
    [string]$OwnerRepo = "",
    [string]$ReleaseTag = "",
    [string]$ReleaseTitle = "",
    [string]$ManifestPath = "ml/model_assets.json",
    [switch]$Draft,
    [switch]$Prerelease,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))

function Resolve-ProjectPath {
    param([string]$RelativePath)

    $fullPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $RelativePath))
    $projectPrefix = $ProjectRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar

    if (-not $fullPath.StartsWith($projectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Ruta fuera del proyecto detectada en manifiesto: $RelativePath"
    }

    return $fullPath
}

function Get-OwnerRepo {
    param([string]$RequestedOwnerRepo)

    if (-not [string]::IsNullOrWhiteSpace($RequestedOwnerRepo)) {
        return $RequestedOwnerRepo.Trim().TrimEnd("/")
    }

    $remote = (& git -C $ProjectRoot remote get-url origin 2>$null)
    if ([string]::IsNullOrWhiteSpace($remote)) {
        throw "No se pudo detectar git remote origin. Usa -OwnerRepo `"usuario/repositorio`"."
    }

    if ($remote -match "github\.com[:/](?<repo>[^/]+/.+?)(?:\.git)?/?$") {
        return $Matches.repo.TrimEnd("/")
    }

    throw "No se pudo extraer OwnerRepo desde origin: $remote. Usa -OwnerRepo `"usuario/repositorio`"."
}

function Get-Sha256 {
    param([string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$manifestFullPath = Resolve-ProjectPath $ManifestPath
if (-not (Test-Path -LiteralPath $manifestFullPath)) {
    throw "No existe el manifiesto de modelos: $ManifestPath"
}

$manifest = Get-Content -LiteralPath $manifestFullPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($ReleaseTag)) {
    $ReleaseTag = $manifest.release_tag
}
if ([string]::IsNullOrWhiteSpace($ReleaseTitle)) {
    $ReleaseTitle = $manifest.release_name
}
if ([string]::IsNullOrWhiteSpace($ReleaseTag)) {
    throw "No hay ReleaseTag definido. Usa -ReleaseTag o completa release_tag en $ManifestPath."
}
if ([string]::IsNullOrWhiteSpace($ReleaseTitle)) {
    $ReleaseTitle = $ReleaseTag
}

$resolvedOwnerRepo = Get-OwnerRepo $OwnerRepo
$assetsToUpload = @()

foreach ($asset in $manifest.assets) {
    $sourcePath = Resolve-ProjectPath $asset.source_path
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "No existe el artefacto local requerido: $($asset.source_path)"
    }

    $actualHash = Get-Sha256 $sourcePath
    if ($actualHash -ne ([string]$asset.sha256).ToLowerInvariant()) {
        throw "Hash local distinto al manifiesto para $($asset.asset_name). Actualiza ml/model_assets.json antes de publicar."
    }

    $assetsToUpload += [PSCustomObject]@{
        Model = $asset.model_id
        Asset = $asset.asset_name
        Path = $sourcePath
        SizeBytes = (Get-Item -LiteralPath $sourcePath).Length
        Sha256 = $actualHash
    }
}

if ($DryRun) {
    Write-Host "DRY RUN: se publicaria release '$ReleaseTag' en $resolvedOwnerRepo"
    $assetsToUpload | Format-List
    return
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI no esta instalado o no esta disponible en PATH. Instala gh y ejecuta gh auth login."
}

& gh auth status --hostname github.com
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI no esta autenticado. Ejecuta gh auth login."
}

$viewArgs = @("release", "view", $ReleaseTag, "--repo", $resolvedOwnerRepo)
$previousErrorActionPreference = $ErrorActionPreference
$previousNativeCommandPreference = $null
$hasNativeCommandPreference = Test-Path Variable:\PSNativeCommandUseErrorActionPreference

if ($hasNativeCommandPreference) {
    $previousNativeCommandPreference = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false
}

$ErrorActionPreference = "Continue"
& gh @viewArgs 1>$null 2>$null
$viewExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference

if ($hasNativeCommandPreference) {
    $PSNativeCommandUseErrorActionPreference = $previousNativeCommandPreference
}

$releaseExists = ($viewExitCode -eq 0)

if (-not $releaseExists) {
    $notes = "Artefactos finales de modelos backend. El codigo versiona ml/model_assets.json y estos pesos se descargan con scripts/models/download-models.ps1."
    $createArgs = @("release", "create", $ReleaseTag, "--repo", $resolvedOwnerRepo, "--title", $ReleaseTitle, "--notes", $notes)
    if ($Draft) {
        $createArgs += "--draft"
    }
    if ($Prerelease) {
        $createArgs += "--prerelease"
    }

    & gh @createArgs
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo crear la release $ReleaseTag."
    }
} else {
    Write-Host "Release existente: $ReleaseTag"
}

foreach ($asset in $assetsToUpload) {
    $uploadArgs = @("release", "upload", $ReleaseTag, $asset.Path, "--repo", $resolvedOwnerRepo)
    if ($Force) {
        $uploadArgs += "--clobber"
    }

    Write-Host "Subiendo $($asset.Asset)"
    & gh @uploadArgs
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo subir $($asset.Asset). Si ya existe, usa -Force para reemplazarlo."
    }
}

Write-Host "Release de modelos lista: https://github.com/$resolvedOwnerRepo/releases/tag/$ReleaseTag"
