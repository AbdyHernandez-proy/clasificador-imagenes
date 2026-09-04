# RetinaNet

## Estado

Modelo entrenado localmente por lotes.

- ID: `custom-retinanet-detector`
- Runtime: `torchvision`
- Arquitectura base: `retinanet_resnet50_fpn`
- Tarea: deteccion de objetos
- Dataset objetivo: `voc-detect`
- Artefacto final activo: `ml/models/custom-retinanet-detector/retinanet-detector.pt`
- Politica de artefactos: se conserva solo el artefacto final con nombre del modelo para despliegue
- Clases objetivo: 20 clases PASCAL VOC
- Estado actual: version final calibrada
- Servicio frontend/backend: activo

## Preflight completado

- Dataset `voc-detect`: disponible localmente.
- Split activo confirmado:
  - entrenamiento: 12,902 imagenes
  - validacion: 8,601 imagenes
  - proporcion: 60/40
- Splits oficiales conservados:
  - `official_train`: 16,551 imagenes
  - `official_val`: 4,952 imagenes
- Clases confirmadas: VOC20.
- Registro `ml/registry.json`: `status=ready`, `serve=true`.
- Artefacto entrenado disponible: `ml/models/custom-retinanet-detector/retinanet-detector.pt`.
- Historial de validacion consolidado en este documento y en `ml/registry.json`.
- Entorno ML detectado: `C:\Users\abdyh\.ciml\venv\Scripts\python.exe`.
- PyTorch/TorchVision disponibles con CUDA.

## Resultado del entrenamiento

- Epocas completadas: 6 de 6.
- Mejor epoca: 6.
- Mejor metrica principal: `precision_at_50 = 0.9168`.
- Los resultados relevantes del run de entrenamiento fueron consolidados en este documento y en `ml/registry.json`.
- Loss por epoca:
  - epoca 1: `1.353894`
  - epoca 2: `1.007696`
  - epoca 3: `0.715248`
  - epoca 4: `0.587615`
  - epoca 5: `0.456826`
  - epoca 6: `0.417097`
- Validacion de epoca 6:
  - `map50 = 0.4267`
  - `map50_95 = 0.2922`
  - `recall_at_50 = 0.4845`
  - falsos positivos: 54
  - falsos negativos: 633
  - predicciones totales: 649
  - imagenes evaluadas: 500
  - tiempo promedio de inferencia: `128.77ms`
Nota operativa: durante el entrenamiento hubo un checkpoint incompleto por un fallo cuDNN, pero la recuperacion se hizo desde el mejor checkpoint valido y el entrenamiento finalizo correctamente. Los checkpoints temporales y el historial externo se eliminaron despues de conservar `retinanet-detector.pt` y documentar las metricas.

## Calibracion de umbrales

Calibracion directa realizada sobre 500 imagenes del split de validacion usando el artefacto final `retinanet-detector.pt` y reconstruccion de inferencia alineada a `image_size=512`.

| Confidence | NMS | Max cajas | Precision@50 | Recall@50 | mAP@50 | mAP@50:95 | FP | FN | Predicciones |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | 0.35 | 8 | 0.8358 | 0.5888 | 0.5351 | 0.3532 | 142 | 505 | 865 |
| 0.50 | 0.35 | 8 | 0.8791 | 0.5505 | 0.4939 | 0.3321 | 93 | 552 | 769 |
| 0.55 | 0.35 | 8 | 0.9088 | 0.5195 | 0.4647 | 0.3144 | 64 | 590 | 702 |
| 0.60 | 0.35 | 8 | 0.9262 | 0.4805 | 0.4257 | 0.2917 | 47 | 638 | 637 |
| 0.65 | 0.35 | 8 | 0.9473 | 0.4389 | 0.3869 | 0.2705 | 30 | 689 | 569 |
| 0.70 | 0.35 | 7 | 0.9514 | 0.3664 | 0.3235 | 0.2322 | 23 | 778 | 473 |
| 0.75 | 0.35 | 6 | 0.9689 | 0.3046 | 0.2735 | 0.1991 | 12 | 854 | 386 |
| 0.80 | 0.35 | 5 | 0.9858 | 0.2256 | 0.2028 | 0.1500 | 4 | 951 | 281 |

Configuracion final aplicada:

- `confidence_threshold = 0.50`
- `nms_threshold = 0.35`
- `max_detections = 8`
- Motivo: conserva precision alta, mantiene mejor recall util que los umbrales mas estrictos y reduce falsos positivos frente al umbral mas permisivo.

Las salidas temporales de calibracion fueron eliminadas despues de documentar los resultados relevantes.

## Uso previsto

RetinaNet se usara como comparativo directo contra Faster R-CNN usando el mismo dataset VOC 60/40. Su interes principal es evaluar si focal loss reduce falsos positivos y mejora clases desbalanceadas u objetos pequenos/dificiles.

## Dataset

- Dataset: `voc-detect`
- Formato interno: YOLO convertido desde PASCAL VOC
- Entrenamiento: 60% = 12,902 imagenes
- Validacion: 40% = 8,601 imagenes
- Clases: VOC20

## Configuracion de entrenamiento usada

- Imagen: 512 px
- Batch: 1
- Epocas iniciales: 6
- Learning rate: `0.0005`
- Momentum: `0.9`
- Weight decay: `0.0005`
- Scheduler: StepLR con `lr_step_size=2`, `lr_gamma=0.5`
- Congelamiento inicial de backbone: 1 epoca
- Best metric: `precision_at_50`
- Validation limit: 500 imagenes por epoca
- Validation confidence: `0.60`
- Validation max detections: `8`
- Validation visual limit: 16 imagenes anotadas por epoca
- Min delta: `0.0005`
- Early stopping patience: 3 epocas sin mejora
- Validacion por epoca: si
- Salidas visuales por epoca: si
- Checkpoints temporales: usados durante entrenamiento y eliminados en el cierre final

## Comando de entrenamiento usado

```powershell
.\scripts\training\train-retinanet.ps1
```

Comando equivalente largo:

```powershell
.\scripts\training\train-voc-detectors.ps1 -Models retinanet -Epochs 6 -Batch 1 -ImageSize 512 -SessionSamples 1024 -SessionCount 64 -LogEvery 10 -CheckpointEvery 25 -Workers 0 -Device auto -LearningRate 0.0005 -Momentum 0.9 -WeightDecay 0.0005 -LrStepSize 2 -LrGamma 0.5 -FreezeBackboneEpochs 1 -ValidationLimit 500 -ValidationConfidence 0.60 -ValidationMaxDetections 8 -ValidationVisualLimit 16 -BestMetric precision_at_50 -MinDelta 0.0005 -EarlyStoppingPatience 3 -PublishPartial -ValidateEveryEpoch
```

## Criterios de aceptacion

- Reducir falsos positivos frente al primer comportamiento historico de RetinaNet.
- Mantener un recall util, evitando configuraciones demasiado estrictas.
- Guardar el artefacto final `retinanet-detector.pt` solo cuando mejore la metrica principal.
- Registrar calibracion final en `ml/registry.json`.

## Estado de cierre

- Artefacto final consolidado como `retinanet-detector.pt`.
- Umbral balanceado aplicado en `ml/registry.json`.
- Inferencia backend conectada mediante worker TorchVision persistente.
- Disponible desde el frontend como modelo solo imagen.
- Sin pendientes activos para esta version del modelo.

