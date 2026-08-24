# Datasets y modelos propios

Este documento resume la estructura actual para trabajar datasets y modelos propios desde el backend.

## Registros principales

| Registro | Funcion |
|---|---|
| `ml/datasets/registry.json` | Catalogo de datasets disponibles o descargables. |
| `ml/registry.json` | Catalogo de modelos servibles, entrenados, experimentales o planificados. |
| `docs/REGISTRO_ENTRENAMIENTOS.md` | Historial consolidado de entrenamientos, metricas y evaluaciones. |

## Carpetas relevantes

| Carpeta | Uso |
|---|---|
| `ml/training/` | Codigo de entrenamiento, configuraciones y adaptadores de datasets. |
| `ml/inference/` | Workers de inferencia para YOLO y TorchVision. |
| `ml/models/` | Artefactos activos de modelos entrenados. No se versiona en Git. |
| `ml/evaluation/` | Script de evaluacion. Las salidas generadas se consideran temporales. |
| `scripts/` | Scripts PowerShell/BAT para preparar datasets, entrenar y revisar progreso. |

## Datasets registrados

| Dataset | Formato | Uso actual | Estado local tras limpieza |
|---|---|---|---|
| `custom-local-v1` | Estructura propia | Plantilla futura para datasets internos. | No requiere descarga. |
| `coco8-detect` | YOLO | Pruebas minimas de deteccion. | Regenerable. |
| `coco128-detect` | YOLO | Entrenamiento inicial de YOLO. | Regenerable. |
| `voc-detect` | YOLO/VOC convertido | Entrenamiento Faster R-CNN y RetinaNet. | Regenerable. |
| `penn-fudan-pedestrian` | Pedestrian/masks | Candidato futuro especializado. | No descargado. |
| `oxford-iiit-pets` | Clasificacion/segmentacion | Candidato futuro. | No descargado. |

Los datos pesados no deben quedar dentro del repositorio. Se descargan o regeneran cuando se vaya a entrenar.

## Comandos utiles

Desde `backend/`:

```bash
python manage_datasets.py list
python manage_datasets.py show coco128-detect
python manage_datasets.py download coco128-detect
```

Desde la raiz del proyecto:

```bash
python -m ml.training.train list-models
python -m ml.training.train list-datasets
python -m ml.training.train compatible-models coco128-detect
python -m ml.training.train validate-dataset coco128-detect
```

## Entrenamiento local

Los scripts locales viven en `scripts/`:

```powershell
.\scripts\prepare-voc-dataset.ps1
.\scripts\train-local-models.ps1
.\scripts\train-voc-managed.ps1
.\scripts\show-training-progress.ps1 -Model custom-faster-rcnn-detector
```

El entorno ML recomendado queda fuera del proyecto, en una ruta corta del usuario:

```text
C:\Users\<usuario>\.ciml\venv
```

Esto evita rutas largas de PyTorch dentro de OneDrive.

## Politica de limpieza

Se pueden eliminar y regenerar:

- `ml/datasets/downloads/`
- `ml/datasets/raw/`
- `ml/runs/`
- salidas `ml/evaluation/voc-validation-*`
- `frontend/node_modules/`
- `backend/.venv/`
- caches `__pycache__` y `.cache`
- pesos temporales en raiz como `yolo*.pt`

No eliminar sin reemplazo:

- `ml/registry.json`
- `ml/datasets/registry.json`
- `ml/models/*/model.pt`
- `ml/models/*/best.pt`
- `ml/models/*/checkpoint_latest.pt`
- `docs/REGISTRO_ENTRENAMIENTOS.md`
