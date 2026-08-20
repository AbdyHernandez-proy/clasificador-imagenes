# Clasificador de Imagenes con IA

Aplicacion web que clasifica imagenes y detecta objetos usando TensorFlow.js directamente en el navegador. No requiere backend: la carga de modelos, el procesamiento de imagenes y la inferencia se ejecutan del lado del cliente.

## Caracteristicas principales

- Clasificacion de imagenes con MobileNet.
- Deteccion de objetos con COCO-SSD.
- Dibujo de cajas de deteccion sobre imagenes o video cuando se usa COCO-SSD.
- Carga de imagenes desde archivo local.
- Validacion de archivos de imagen antes de procesarlos.
- Limite de tamano de imagen de 10 MB.
- Soporte para arrastrar y soltar imagenes.
- Clasificacion en vivo desde webcam.
- Validacion previa de compatibilidad de webcam con `navigator.mediaDevices?.getUserMedia`.
- Control de inferencias de webcam para procesar un frame a la vez.
- Proteccion contra resultados obsoletos mediante contador de ejecucion.
- Vista previa de imagen o video.
- Panel de resultados con confianza y tiempo de procesamiento.
- Interfaz responsiva para escritorio, tablet y movil.
- Renderizado de resultados mediante nodos DOM y `textContent`, sin `innerHTML`.
- Carga dinamica de modelos para reducir el bundle inicial.

## Requisitos

- Node.js 16 o superior.
- npm.
- Navegador moderno compatible con ES Modules, Canvas, FileReader y MediaDevices.
- Conexion a internet para descargar modelos, salvo que se configuren URLs locales en `src/modelManager.js`.
- Para webcam: permisos de camara y ejecucion en `localhost` o HTTPS.

## Instalacion

```bash
npm install
```

## Ejecucion en desarrollo

```bash
npm run dev
```

Vite esta configurado para usar el puerto `3000` y abrir el navegador automaticamente.

Tambien se incluyen scripts de arranque:

```bash
# Windows
start.bat

# macOS / Linux
chmod +x start.sh
./start.sh
```

## Build y preview

```bash
npm run build
npm run preview
```

El build usa `manualChunks` para separar TensorFlow y los modelos en archivos independientes, reduciendo el bundle inicial de la aplicacion.

## Estructura del proyecto

```text
Clasificador_Imagenes/
|-- index.html
|-- package.json
|-- package-lock.json
|-- vite.config.js
|-- SETUP.md
|-- start.bat
|-- start.sh
|-- README.md
|-- AUDITORIA.md
`-- src/
    |-- main.js
    |-- modelManager.js
    |-- imageProcessor.js
    |-- ui.js
    |-- dom.js
    `-- style.css
```

## Flujo general

1. El navegador carga `index.html`.
2. `src/main.js` inicializa la clase principal `ImageClassifier`.
3. `ModelManager` carga MobileNet como modelo inicial bajo demanda.
4. El usuario carga una imagen, arrastra un archivo o activa la webcam.
5. `ImageProcessor` valida tipo/tamano, lee y redimensiona imagenes antes de la inferencia.
6. `ImageClassifier` envia la imagen, canvas o video al modelo activo.
7. `ClassifierUI` muestra resultados, confianza, tiempo de procesamiento y vista previa.
8. Si el modelo activo es COCO-SSD, `ClassifierUI` dibuja cajas de deteccion sobre la imagen o video.
9. Si se cambia de modelo o se inicia una nueva accion, el contador de ejecucion evita que resultados antiguos sobrescriban la UI.
10. Si se cambia de modelo, la app detiene la webcam, carga/reutiliza el modelo seleccionado y limpia resultados.

## Componentes tecnicos

### `index.html`

Define la estructura de la aplicacion:

- Header con titulo y descripcion breve.
- Selector de modelo.
- Area de carga de imagenes.
- Boton de webcam.
- Contenedor de imagen/video.
- Canvas para cajas de deteccion.
- Panel de resultados.
- Panel de estadisticas.
- Footer con modelo activo y estado.

### `src/main.js`

Orquesta la aplicacion mediante la clase `ImageClassifier`.

Responsabilidades principales:

- Inicializar UI, procesador de imagenes y gestor de modelos.
- Cargar el modelo inicial.
- Registrar eventos de archivo, selector de modelo, webcam y drag and drop.
- Validar archivos seleccionados o arrastrados antes de procesarlos.
- Clasificar imagenes cargadas por el usuario.
- Iniciar y detener la webcam.
- Validar disponibilidad de webcam antes de llamar `getUserMedia`.
- Procesar webcam frame por frame sin inferencias concurrentes.
- Ignorar resultados obsoletos con `runId`.
- Enviar imagenes/video al modelo activo.
- Medir tiempo de procesamiento.
- Actualizar resultados, estadisticas y cajas de deteccion en pantalla.

Funciones relevantes:

- `initialize()`
- `setupEventListeners()`
- `handleImageUpload(event)`
- `classifyImage(imageFile)`
- `getCurrentPredictions(imageElement)`
- `switchModel(modelName)`
- `startWebcam()`
- `processWebcamFrames(video, runId)`
- `stopWebcam(options)`
- `handleDrop(event)`
- `validateSelectedImage(file)`
- `nextRunId()`
- `isCurrentRun(runId)`

### `src/modelManager.js`

Gestiona los modelos de IA.

Responsabilidades principales:

- Cargar MobileNet o COCO-SSD bajo demanda mediante `import()` dinamico.
- Reutilizar modelos ya cargados para evitar descargas repetidas.
- Mantener el modelo activo.
- Exponer nombre, tipo e instancia del modelo actual.
- Soportar `modelUrl` local o personalizado mediante `MODEL_URLS`.
- Liberar modelos si se llama a `disposeModel()` o `disposeAll()`.

Modelos soportados:

| Modelo | Uso | Metodo |
| --- | --- | --- |
| MobileNet | Clasificacion general de imagenes | `classify(imageElement, 5)` |
| COCO-SSD | Deteccion de objetos con cajas | `detect(imageElement)` |

### `src/imageProcessor.js`

Prepara imagenes para inferencia.

Responsabilidades principales:

- Validar que el archivo exista.
- Validar que el archivo sea una imagen.
- Rechazar imagenes mayores de 10 MB.
- Leer archivos locales con `FileReader`.
- Crear elementos `Image` desde el archivo cargado.
- Redimensionar imagenes a un maximo de 640 x 480 px.
- Mantener proporcion al redimensionar.
- Dibujar la imagen final en un canvas.

### `src/ui.js`

Centraliza la manipulacion de la interfaz.

Responsabilidades principales:

- Mostrar y ocultar loading.
- Renderizar resultados de MobileNet y COCO-SSD usando nodos DOM.
- Mostrar porcentaje de confianza.
- Mostrar tiempo de procesamiento.
- Mostrar imagen de vista previa.
- Dibujar y limpiar cajas de deteccion de COCO-SSD en `detection-canvas`.
- Actualizar estado del modelo.
- Limpiar resultados.
- Mostrar errores sin insertar HTML dinámico.

### `src/dom.js`

Contiene helpers para elementos requeridos del DOM:

- `requireElementById(id)`
- `requireSelector(selector)`

Estos helpers lanzan errores claros si falta un elemento esperado del HTML.

### `src/style.css`

Define el aspecto visual y layout:

- Tema oscuro con variables CSS.
- Header principal.
- Panel de controles.
- Area de carga con drag and drop.
- Contenedor de imagen/video.
- Overlay absoluto para canvas de deteccion.
- Panel lateral de resultados.
- Estadisticas de confianza y tiempo.
- Reglas responsivas para pantallas pequenas.

### `vite.config.js`

Configuracion de Vite:

```js
export default defineConfig({
  server: {
    port: 3000,
    open: true
  },
  build: {
    target: 'esnext',
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('@tensorflow-models/mobilenet')) return 'model-mobilenet'
          if (id.includes('@tensorflow-models/coco-ssd')) return 'model-coco-ssd'
          if (id.includes('@tensorflow')) return 'tensorflow'
        }
      }
    }
  }
})
```

## Dependencias

| Paquete | Funcion |
| --- | --- |
| `@tensorflow/tfjs` | Runtime de machine learning en navegador. |
| `@tensorflow-models/mobilenet` | Modelo de clasificacion de imagenes. |
| `@tensorflow-models/coco-ssd` | Modelo de deteccion de objetos. |
| `vite` | Servidor de desarrollo y empaquetador frontend. |

## Uso de la aplicacion

### Clasificar una imagen

1. Abrir la aplicacion en el navegador.
2. Seleccionar MobileNet o COCO-SSD.
3. Hacer clic en el area de carga o arrastrar una imagen valida menor o igual a 10 MB.
4. Revisar resultados, confianza y tiempo de procesamiento.
5. Si se usa COCO-SSD, revisar las cajas de deteccion sobre la imagen.

### Usar webcam

1. Hacer clic en `Usar Webcam`.
2. Autorizar permisos de camara en el navegador.
3. Ver resultados actualizados en tiempo real.
4. Si se usa COCO-SSD, revisar las cajas de deteccion sobre el video.
5. Hacer clic en `Detener Webcam` para finalizar.

### Cambiar modelo

- MobileNet: opcion mas rapida para clasificacion general.
- COCO-SSD: opcion para detectar/listar multiples objetos y dibujar cajas de deteccion.

## Auditoria

El archivo `AUDITORIA.md` contiene el estado actual de incidentes activos del proyecto.

## Licencia

No hay archivo `LICENSE` en el proyecto. Si se va a publicar o distribuir, definir la licencia real y agregar el archivo correspondiente.
