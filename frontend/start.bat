@echo off
REM Script de inicio para desarrollo en Windows

echo 🖼️  Iniciando Clasificador de Imagenes...
echo.

REM Verificar si node_modules existe
if not exist "node_modules" (
    echo 📦 Instalando dependencias...
    call npm install
)

echo.
echo 🚀 Iniciando servidor de desarrollo...
echo El navegador se abrira automaticamente en http://localhost:3000
echo.
call npm run dev

pause
