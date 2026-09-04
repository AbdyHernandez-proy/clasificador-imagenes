param(
    [string]$OwnerRepo = "",
    [string]$ReleaseTag = "",
    [string]$ManifestPath = "ml/model_assets.json",
    [switch]$Force,
    [switch]$SkipHash,
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
if ([string]::IsNullOrWhiteSpace($ReleaseTag)) {
    throw "No hay ReleaseTag definido. Usa -ReleaseTag o completa release_tag en $ManifestPath."
}

$resolvedOwnerRepo = Get-OwnerRepo $OwnerRepo
$summary = @()

foreach ($asset in $manifest.assets) {
    $targetPath = Resolve-ProjectPath $asset.target_path
    $targetDir = Split-Path -Parent $targetPath
    $assetName = [string]$asset.asset_name
    $escapedAssetName = [System.Uri]::EscapeDataString($assetName)
    $url = "https://github.com/$resolvedOwnerRepo/releases/download/$ReleaseTag/$escapedAssetName"
    $status = "pending"

    if (Test-Path -LiteralPath $targetPath) {
        if (-not $SkipHash) {
            $existingHash = Get-Sha256 $targetPath
            if ($existingHash -eq ([string]$asset.sha256).ToLowerInvariant()) {
                $status = "ok-local"
            } elseif (-not $Force) {
                throw "El archivo local existe pero su hash no coincide: $($asset.target_path). Usa -Force para descargarlo otra vez."
            }
        } elseif (-not $Force) {
            $status = "exists-local"
        }
    }

    if ($DryRun) {
        $summary += [PSCustomObject]@{
            Model = $asset.model_id
            Asset = $assetName
            Target = $asset.target_path
            SourceUrl = $url
            Status = if ($status -eq "pending") { "would-download" } else { $status }
        }
        continue
    }

    if ($status -eq "ok-local" -or $status -eq "exists-local") {
        Write-Host "OK $assetName ya existe localmente."
        $summary += [PSCustomObject]@{ Model = $asset.model_id; Asset = $assetName; Target = $asset.target_path; Status = $status }
        continue
    }

    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Write-Host "Descargando $assetName desde $url"
    Invoke-WebRequest -Uri $url -OutFile $targetPath

    if (-not $SkipHash) {
        $downloadedHash = Get-Sha256 $targetPath
        if ($downloadedHash -ne ([string]$asset.sha256).ToLowerInvariant()) {
            Remove-Item -LiteralPath $targetPath -Force
            throw "Hash invalido para $assetName. Se elimino la descarga corrupta."
        }
    }

    $summary += [PSCustomObject]@{ Model = $asset.model_id; Asset = $assetName; Target = $asset.target_path; Status = "downloaded" }
}

if ($DryRun) {
    $summary | Format-List
} else {
    $summary | Format-Table -AutoSize
}
