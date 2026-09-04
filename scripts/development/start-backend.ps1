param(
    [int]$Port = 8000,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$BackendRoot = Join-Path $ProjectRoot "backend"
$PythonPath = Join-Path $BackendRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "No existe backend\.venv. Ejecuta: python -m venv backend\.venv; backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt"
}

Set-Location $BackendRoot
& $PythonPath -m uvicorn app.main:app --host $HostAddress --port $Port
