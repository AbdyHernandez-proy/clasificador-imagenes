@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0train-local-models.ps1" %*
endlocal
