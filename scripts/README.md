# Scripts del proyecto

Esta carpeta separa los scripts por tipo de uso. No se mantienen duplicados en la raiz de `scripts/`; cada script vive en la subcarpeta que corresponde a su funcion.

## Entrenamiento

Ubicacion: `scripts/training/`

- `train-local-models.ps1`: orquestador central para entrenar modelos locales.
- `train-voc-detectors.ps1`: wrapper especializado para detectores basados en PASCAL VOC.
- `train-yolo.ps1`: entrenamiento local de YOLO.
- `train-faster-rcnn.ps1`: entrenamiento local de Faster R-CNN.
- `train-retinanet.ps1`: entrenamiento local de RetinaNet.
- `train-rtdetr.ps1`: entrenamiento local de RT-DETR.
- `train-efficientdet.ps1`: preflight, instalacion y entrenamiento de EfficientDet.

## Prediccion visual

Ubicacion: `scripts/prediction/`

- `predict-faster-rcnn.ps1`: genera salida visual y JSON para Faster R-CNN.
- `predict-yolo.ps1`: genera salida visual y JSON para YOLO.
- `predict-retinanet.ps1`: genera salida visual y JSON para RetinaNet.
- `predict-rtdetr.ps1`: genera salida visual y JSON para RT-DETR.
- `predict-efficientdet.ps1`: genera salida visual y JSON para EfficientDet usando el artefacto final del modelo.

## Datasets

Ubicacion: `scripts/datasets/`

- `prepare-voc-dataset.ps1`: prepara el dataset VOC en el split activo 60/40.
- `download-datasets.ps1`: descarga datasets registrados. Por defecto prepara `voc-detect`; acepta `-DatasetId all`.

## Desarrollo local

Ubicacion: `scripts/development/`

- `start-backend.ps1`: inicia la API FastAPI en `http://127.0.0.1:8000`.
- `start-frontend.ps1`: inicia Vite en `http://127.0.0.1:3000`.
- `start-app.ps1`: inicia backend y frontend en segundo plano.

## Evaluacion y calibracion

Ubicacion: `scripts/evaluation/`

- `calibrate-yolo.ps1`: calibra umbrales de YOLO sobre COCO128.
- `calibrate-rtdetr.ps1`: calibra umbrales de RT-DETR sobre VOC.
- `calibrate-faster-rcnn.ps1`: calibra umbrales de Faster R-CNN sobre VOC.
- `calibrate-retinanet.ps1`: calibra umbrales de RetinaNet sobre VOC.
- `calibrate-efficientdet.ps1`: calibra umbrales de EfficientDet sobre VOC.

## Monitoreo

Ubicacion: `scripts/monitoring/`

- `show-training-progress.ps1`: consulta progreso, eventos y validaciones de entrenamientos por lotes.

## Modelos y GitHub Releases

Ubicacion: `scripts/models/`

- `download-models.ps1`: descarga los artefactos `.pt` publicados en GitHub Releases y verifica SHA256 contra `ml/model_assets.json`.
- `publish-models-release.ps1`: publica los `.pt` locales como assets de una Release usando GitHub CLI.

Los pesos viven en `ml/models/`, pero esa carpeta esta ignorada por Git. El repositorio versiona el manifiesto `ml/model_assets.json` para saber que descargar y donde colocarlo.

## Comandos principales

Estos comandos se ejecutan desde la raiz del proyecto:

```powershell
.\scripts\training\train-rtdetr.ps1
.\scripts\training\train-yolo.ps1
.\scripts\training\train-faster-rcnn.ps1
.\scripts\training\train-retinanet.ps1
.\scripts\training\train-efficientdet.ps1
.\scripts\prediction\predict-yolo.ps1 -Image "C:\ruta\a\imagen.jpg"
.\scripts\prediction\predict-rtdetr.ps1 -Image "C:\ruta\a\imagen.jpg"
.\scripts\prediction\predict-faster-rcnn.ps1 -Image "C:\ruta\a\imagen.jpg"
.\scripts\prediction\predict-retinanet.ps1 -Image "C:\ruta\a\imagen.jpg"
.\scripts\prediction\predict-efficientdet.ps1 -Image "C:\ruta\a\imagen.jpg"
.\scripts\evaluation\calibrate-rtdetr.ps1
.\scripts\evaluation\calibrate-faster-rcnn.ps1
.\scripts\evaluation\calibrate-retinanet.ps1
.\scripts\evaluation\calibrate-efficientdet.ps1
.\scripts\datasets\download-datasets.ps1
.\scripts\datasets\prepare-voc-dataset.ps1
.\scripts\development\start-app.ps1
.\scripts\monitoring\show-training-progress.ps1
.\scripts\models\download-models.ps1
.\scripts\models\publish-models-release.ps1
```

Si PowerShell bloquea scripts, usar `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ruta\del\script.ps1`.

