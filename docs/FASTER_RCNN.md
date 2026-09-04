# Faster R-CNN

## Estado

Modelo backend finalizado en su version actual.

- ID: `custom-faster-rcnn-detector`
- Runtime: `torchvision`
- Arquitectura base: `fasterrcnn_resnet50_fpn`
- Tarea: deteccion de objetos
- Dataset usado: `voc-detect`
- Artefacto final activo: `ml/models/custom-faster-rcnn-detector/faster-rcnn-detector.pt`
- Politica de artefactos: se conserva solo el artefacto final con nombre del modelo para despliegue
- Clases: 20 clases PASCAL VOC
- Tamano de imagen de entrenamiento: 512 px
- Batch: 1
- Epocas finales: 6
- Mejor epoca: 6
- Metrica usada para seleccionar mejor modelo: `precision_at_50`

## Clases VOC

`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.

## Configuracion final de inferencia

Esta es la configuracion balanceada seleccionada para uso manual/backend:

- `confidence_threshold`: `0.80`
- `nms_threshold`: `0.40`
- `detections_per_img`: `8`
- `max_detections`: `8`

Esta configuracion mantiene hasta 8 cajas por imagen, reduce ruido frente a los umbrales bajos y conserva mas detecciones utiles que el modo de alta precision.

## Entrenamiento final

Dataset activo: PASCAL VOC convertido a YOLO/VOC interno.

- Split de entrenamiento: 60% = 12,902 imagenes
- Split de validacion: 40% = 8,601 imagenes
- Validacion por epoca usada durante entrenamiento: 500 imagenes
- Learning rate inicial: `0.0005`
- Momentum: `0.9`
- Weight decay: `0.0005`
- Scheduler: StepLR con `lr_step_size=2`, `lr_gamma=0.5`
- Congelamiento inicial de backbone: 1 epoca
- Checkpoints usados durante entrenamiento: si
- Salidas visuales por epoca: si, generadas durante entrenamiento y luego eliminadas como artefactos temporales
- Historiales externos de validacion eliminados despues de consolidar metricas en este documento y en `ml/registry.json`

## Perdida por epoca

| Epoca | Loss |
|---:|---:|
| 1 | 0.480296 |
| 2 | 0.410742 |
| 3 | 0.345209 |
| 4 | 0.296942 |
| 5 | 0.278076 |
| 6 | 0.252481 |

## Validacion de entrenamiento

| Epoca | Precision@50 | Recall@50 | mAP@50 | mAP@50:95 | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3345 | 0.2248 | 0.0369 | 0.0112 | 549 | 952 |
| 2 | 0.5473 | 0.6547 | 0.5421 | 0.2624 | 665 | 424 |
| 3 | 0.5965 | 0.6897 | 0.5964 | 0.3232 | 573 | 381 |
| 4 | 0.6440 | 0.7394 | 0.6618 | 0.3828 | 502 | 320 |
| 5 | 0.6788 | 0.7467 | 0.6930 | 0.4028 | 434 | 311 |
| 6 | 0.6991 | 0.7378 | 0.6750 | 0.4056 | 390 | 322 |

## Calibracion seleccionada

La calibracion final comparo umbrales sobre 120 imagenes del split de validacion. Se eligio modo balanceado:

| Ajuste | Confidence | NMS | Max cajas | Precision@50 | Recall@50 | FP | FN | Predicciones |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanceado final | 0.80 | 0.40 | 8 | 0.7045 | 0.4576 | 52 | 147 | 176 |

Se descarto el modo de alta precision `0.90 / 0.35 / 5` porque aunque alcanzo 0.7951 de precision, bajo el recall a 0.3579 y dejaba demasiados objetos reales sin detectar.

## Prueba manual con imagen propia

Desde la raiz del proyecto:

```powershell
.\scripts\prediction\predict-faster-rcnn.ps1 -Image "C:\ruta\a\imagen.jpg"
```

Si PowerShell bloquea scripts:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prediction\predict-faster-rcnn.ps1 -Image "C:\ruta\a\imagen.jpg"
```

La salida se guarda en:

```text
ml/evaluation/manual-predictions/
```

## Estado de cierre

- Artefacto final consolidado como `faster-rcnn-detector.pt`.
- Umbral balanceado aplicado en `ml/registry.json`.
- Inferencia backend conectada mediante worker TorchVision persistente.
- Disponible desde el frontend como modelo solo imagen.
- Sin pendientes activos para esta version del modelo.

