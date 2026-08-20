# ⚡ Guía Rápida de Inicio

## Windows 🪟

### Opción 1: Doble clic (Más fácil)
1. Abre la carpeta `Clasificador_Imagenes`
2. Haz doble clic en `start.bat`
3. Se abrirá el navegador automáticamente

### Opción 2: Terminal
```powershell
cd "C:\Users\abdyh\OneDrive\Documents\Proyectos_VSC\Clasificador_Imagenes"
npm run dev
```

## macOS / Linux 🐧

### Terminal
```bash
cd Clasificador_Imagenes
npm run dev
```

O ejecuta el script:
```bash
chmod +x start.sh
./start.sh
```

## Primeros Pasos

Una vez que el servidor esté corriendo (http://localhost:3000):

1. **Clasificar una imagen**
   - Arrastra una imagen al área de carga
   - O haz clic para seleccionar un archivo

2. **Usar la webcam**
   - Haz clic en el botón "📹 Usar Webcam"
   - Autoriza el acceso a tu cámara
   - La clasificación se actualiza en tiempo real

3. **Cambiar modelo**
   - Selecciona entre MobileNet o COCO-SSD
   - MobileNet es más rápido
   - COCO-SSD es más preciso

## Comandos Útiles

```bash
# Desarrollo (con hot reload)
npm run dev

# Compilar para producción
npm run build

# Previsualizar build
npm run preview

# Instalar dependencias
npm install
```

## Requisitos

- Node.js 16+
- npm (viene con Node.js)
- Navegador moderno (Chrome, Firefox, Safari, Edge)

## Solución de Problemas

### Error de scripts deshabilitados en PowerShell
```powershell
powershell -ExecutionPolicy Bypass
npm install
npm run dev
```

### Puerto 3000 en uso
Edita `vite.config.js` y cambia el puerto:
```javascript
server: {
    port: 3001,  // Cambiar a otro puerto
    open: true
}
```

### La webcam no aparece
- Verifica que diste permiso al navegador
- Recarga la página
- Intenta con otro navegador

## 📖 Documentación Completa

Ver `README.md` para documentación completa del proyecto.
