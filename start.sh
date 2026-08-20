#!/bin/bash
# Script de inicio para desarrollo

echo "🖼️  Iniciando Clasificador de Imágenes..."
echo ""

# Verificar si node_modules existe
if [ ! -d "node_modules" ]; then
    echo "📦 Instalando dependencias..."
    npm install
fi

echo ""
echo "🚀 Iniciando servidor de desarrollo..."
echo "El navegador se abrirá automáticamente en http://localhost:3000"
echo ""
npm run dev
