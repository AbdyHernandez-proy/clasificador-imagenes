#!/bin/bash
# Script de inicio para desarrollo.

echo "Iniciando Clasificador de Imagenes..."
echo ""

if [ ! -d "node_modules" ]; then
    echo "Instalando dependencias..."
    npm install
fi

echo ""
echo "Iniciando servidor de desarrollo..."
echo "El navegador se abrira automaticamente en http://localhost:3000"
echo ""
npm run dev
