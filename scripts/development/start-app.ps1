param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$BackendRoot = Join-Path $ProjectRoot "backend"
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$PythonPath = Join-Path $BackendRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "No existe backend\.venv. Ejecuta: python -m venv backend\.venv; backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt"
}

$BackendProcess = Start-Process -FilePath $PythonPath `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", $HostAddress, "--port", "$BackendPort") `
    -WorkingDirectory $BackendRoot `
    -WindowStyle Hidden `
    -PassThru

$FrontendProcess = Start-Process -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--", "--host", $HostAddress, "--port", "$FrontendPort") `
    -WorkingDirectory $FrontendRoot `
    -WindowStyle Hidden `
    -PassThru

[PSCustomObject]@{
    backend_pid = $BackendProcess.Id
    frontend_pid = $FrontendProcess.Id
    backend_url = "http://$HostAddress`:$BackendPort"
    frontend_url = "http://$HostAddress`:$FrontendPort"
}
