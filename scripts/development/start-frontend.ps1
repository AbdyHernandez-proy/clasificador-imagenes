param(
    [int]$Port = 3000,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$FrontendRoot = Join-Path $ProjectRoot "frontend"

Set-Location $FrontendRoot
& npm.cmd run dev -- --host $HostAddress --port $Port
