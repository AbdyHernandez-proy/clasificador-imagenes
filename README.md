# Clasificador de Imagenes

Proyecto interno para clasificar imagenes con una arquitectura separada en frontend, backend y area de machine learning. La web funciona como interfaz de uso; los modelos, datasets y entrenamiento viven fuera del frontend.

## Objetivo

Construir una plataforma interna donde se puedan crear, almacenar, entrenar y usar modelos propios de clasificacion de imagenes.

El usuario de la web solo debe:

- Escoger un modelo disponible.
- Subir una imagen o usar webcam.
- Ver predicciones y resultados.

La administracion de datasets, entrenamiento y modelos se maneja desde carpetas y scripts internos, no desde cuentas de usuario ni autenticacion.

## Arquitectura

```text
Clasificador_Imagenes/
|-- frontend/              # Interfaz web Vite/TensorFlow.js actual
|-- backend/               # API interna FastAPI
|-- ml/                    # Registro, datasets y modelos propios
|-- docs/                  # Documentacion tecnica por modelo
|-- .github/workflows/     # Deploy del frontend a GitHub Pages
`-- README.md
```

## Ejecucion completa local

Desde la raiz del proyecto:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\development\start-app.ps1
```

Esto inicia:

- Backend FastAPI en `http://127.0.0.1:8000`.
- Frontend Vite en `http://127.0.0.1:3000`.

Los modelos propios necesitan el backend activo. La primera prediccion de cada familia de modelo puede tardar mas porque se cargan arquitectura y pesos en memoria; las siguientes predicciones reutilizan workers persistentes.

## Frontend

Ubicacion: `frontend/`

Responsabilidades actuales:

- Mostrar la interfaz de clasificacion.
- Cargar el listado de modelos desde el backend cuando este disponible.
- Permitir seleccionar modelos del navegador o del backend.
- Permitir subir imagenes o usar webcam.
- Validar imagenes antes de procesarlas.
- Mostrar predicciones, confianza y tiempo de procesamiento.
- Dibujar cajas de deteccion para modelos del navegador y modelos propios del backend.
- Mantener GitHub Pages funcionando para pruebas remotas con modelos del navegador.
- Usar fallback automatico a modelos del navegador si el backend no esta disponible.

Formatos de imagen permitidos:

- JPG / JPEG
- PNG
- WEBP
- GIF
- BMP

Tamano maximo por imagen: 10 MB.

Comandos:

```bash
cd frontend
npm install
npm run dev
npm run build
npm run preview
```

## Backend

Ubicacion: `backend/`

Responsabilidades actuales:

- Exponer API interna para la web.
- Listar modelos registrados en `ml/registry.json`.
- Recibir imagenes para inferencia.
- Ejecutar el modelo seleccionado desde backend.
- Devolver predicciones normalizadas al frontend.
- Validar tipo y tamano de imagen antes de procesar.

Endpoints:

- `GET /health`
- `GET /models`
- `GET /models/registry`
- `GET /datasets`
- `GET /datasets/{dataset_id}`
- `POST /predict`

Ejecucion local:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## ML

Ubicacion: `ml/`

Responsabilidades:

- Registrar datasets disponibles para descarga/preparacion local.
- Guardar modelos entrenados.
- Registrar modelos disponibles en `ml/registry.json`.

Estructura:

```text
ml/
|-- datasets/
|-- models/
|-- model_assets.json
`-- registry.json
```

Los scripts de entrenamiento, evaluacion e inferencia propia viven en `scripts/`, `ml/training/`, `ml/evaluation/` y `ml/inference/`.

Los datasets descargados no se versionan en Git para evitar subir archivos pesados. Se reconstruyen localmente desde el catalogo con:

```powershell
.\scripts\datasets\download-datasets.ps1
```

Los pesos finales de modelos tampoco se versionan en Git. Se mantienen fuera del repositorio en `ml/models/` y se publican como assets de GitHub Releases. El archivo `ml/model_assets.json` registra cada artefacto, su destino local, tamano y SHA256.

Para restaurar los modelos despues de clonar el repositorio:

```powershell
.\scripts\models\download-models.ps1
```

Si PowerShell bloquea scripts en Windows:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\models\download-models.ps1
```

Para publicar los pesos locales en GitHub Releases desde una maquina autenticada con GitHub CLI:

```powershell
gh auth login
.\scripts\models\publish-models-release.ps1
```

Alternativa con bypass temporal de politica:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\models\publish-models-release.ps1
```

## Registro de modelos

No se usara base de datos inicialmente. El backend leera `ml/registry.json`.

El registro actual contiene el modelo backend de perfil visual y cinco detectores propios listos para inferencia:

- `custom-yolo-v8-v11-detector`
- `custom-faster-rcnn-detector`
- `custom-retinanet-detector`
- `custom-detr-rtdetr-detector`
- `custom-efficientdet-detector`

## Modelos actuales

### MobileNet (Navegador)

Modelo preentrenado de TensorFlow.js para clasificacion general de imagenes. Corre directamente en el navegador, soporta imagenes cargadas y webcam, y devuelve etiquetas con probabilidad.

### COCO-SSD (Navegador)

Modelo preentrenado de TensorFlow.js para deteccion de objetos. Corre directamente en el navegador, soporta imagenes cargadas y webcam, y permite dibujar cajas de deteccion sobre la imagen.

### BlazeFace (Navegador)

Modelo ligero para deteccion de rostros. Devuelve rostros detectados con confianza y caja aproximada.

### HandPose (Navegador)

Modelo para deteccion de manos. Devuelve una caja aproximada por mano y el conteo de puntos clave detectados.

### BodyPix (Navegador)

Modelo de segmentacion de persona. Indica si encuentra una persona y calcula la proporcion aproximada de pixeles segmentados.

### Perfil visual backend interno

Modelo interno de referencia registrado como `visual-profile-v1`. Corre en FastAPI con Pillow y clasifica propiedades visuales basicas: brillo, color dominante y orientacion. Se mantiene como utilidad interna del backend; no forma parte del selector principal de modelos visibles.

### YOLOv8 / YOLOv11 Detector (Backend)

Detector general entrenado con COCO128. Localiza objetos de las clases COCO mediante cajas y confianza. Runtime: Ultralytics.

### Faster R-CNN Detector (Backend)

Detector entrenado con PASCAL VOC 60/40. Prioriza precision mediante propuestas de region antes de clasificar objetos. Runtime: TorchVision.

### RetinaNet Detector (Backend)

Detector entrenado con PASCAL VOC 60/40. Usa focal loss y FPN para manejar mejor clases dificiles o desbalanceadas. Runtime: TorchVision.

### DETR / RT-DETR Detector (Backend)

Detector entrenado con PASCAL VOC 60/40. Usa arquitectura tipo transformer para comparar contra detectores CNN clasicos. Runtime: Ultralytics RT-DETR.

### EfficientDet Detector (Backend)

Detector entrenado con PASCAL VOC 60/40 y consolidado desde la epoca 8. Usa EfficientNet y BiFPN para equilibrar consumo, velocidad y precision. Runtime: EfficientDet en backend mediante worker persistente, configurado para inferencia CPU por estabilidad local.

## GitHub Pages

El workflow publica solo el frontend desde `frontend/dist`. En GitHub Pages funcionan los modelos del navegador; los modelos propios requieren un backend activo accesible desde la web, por ejemplo mediante tunnel o despliegue de API.

## GitHub Releases

El codigo fuente, registros, scripts y documentacion se suben a Git. Los artefactos `.pt` de los modelos backend se suben a una Release para evitar que el repositorio crezca innecesariamente.

Flujo recomendado:

1. Subir el codigo a Git.
2. Publicar los modelos con `.\scripts\models\publish-models-release.ps1`.
3. En una instalacion nueva, ejecutar `.\scripts\models\download-models.ps1`.
4. Iniciar backend y frontend con `.\scripts\development\start-app.ps1`.

URL actual:

```text
https://abdyhernandez-proy.github.io/clasificador-imagenes/
```

## Sin base de datos, usuarios ni autenticacion

Por decision de arquitectura, el proyecto no incluye inicialmente:

- Usuarios.
- Login.
- Autenticacion.
- Cola de entrenamiento.
- Base de datos.

La persistencia inicial se resuelve con carpetas y archivos versionables.
