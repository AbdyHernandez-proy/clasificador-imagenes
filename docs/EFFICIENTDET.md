# EfficientDet

## Estado

Modelo entrenado localmente, calibrado en umbral balanceado y fijado en la epoca 8 como version final local.

- ID: `custom-efficientdet-detector`
- Runtime: `effdet`
- Arquitectura objetivo: `tf_efficientdet_d0`
- Tarea: deteccion de objetos
- Dataset objetivo: `voc-detect`
- Artefacto activo: `ml/models/custom-efficientdet-detector/efficientdet-detector.pt`
- Politica de artefactos: se conserva solo el peso final activo
- Estado actual: version final local seleccionada y activa
- Servicio frontend/backend: activo mediante worker EfficientDet persistente

## Uso previsto

EfficientDet se usara para comparar una familia de detectores escalables contra Faster R-CNN, RetinaNet y RT-DETR. Su valor principal esta en buscar un balance entre consumo, velocidad y precision.

## Dataset

- Dataset: `voc-detect`
- Formato interno: YOLO
- Entrenamiento: 60% = 12,902 imagenes
- Validacion: 40% = 8,601 imagenes
- Clases: 20 clases PASCAL VOC

## Resultado del entrenamiento final

- Epocas completadas: 8 de 8 seleccionadas
- Epoca final seleccionada: 8
- Criterio de seleccion: balance manual entre loss, recall, mAP50 y precision
- Precision@50: `0.6240`
- Recall@50: `0.1865`
- mAP@50: `0.1404`
- mAP@50:95: `0.0807`
- Loss final: `0.485938`
- Falsos positivos en validacion: `138`
- Falsos negativos en validacion: `999`
- Predicciones totales en validacion: `367`
- Tiempo medio de inferencia en validacion final: aproximadamente `46.31ms`

La epoca 8 se selecciono aunque la epoca 5 tenia una precision@50 apenas mayor (`0.6245` vs `0.6240`). La diferencia de precision es minima, mientras que la epoca 8 mejora loss, recall, mAP50 y mAP50:95. Para uso visual del proyecto, la epoca 8 ofrece una salida mas util que la version anterior.

Comparacion clave:

| Epoca | Loss | Precision@50 | Recall@50 | mAP@50 | mAP@50:95 | FP | FN | Predicciones |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.598739 | 0.6245 | 0.1409 | 0.1024 | 0.0584 | 104 | 1055 | 277 |
| 8 | 0.485938 | 0.6240 | 0.1865 | 0.1404 | 0.0807 | 138 | 999 | 367 |

La perdida bajo de forma sostenida, pero el recall sigue bajo frente a Faster R-CNN, RetinaNet y RT-DETR. El modelo aun detecta con cautela y deja objetos reales sin marcar cuando el umbral es alto.

## Calibracion seleccionada

Calibracion realizada sobre 500 imagenes del split de validacion.

| Ajuste | Confidence | Max cajas | Precision@50 | Recall@50 | mAP@50 | mAP@50:95 | FP | FN | Predicciones |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Muy sensible | 0.30 | 8 | 0.4684 | 0.3200 | 0.1814 | 0.0993 | 446 | 835 | 839 |
| Balanceado seleccionado | 0.35 | 8 | 0.5250 | 0.2647 | 0.1570 | 0.0864 | 294 | 903 | 619 |
| Actual previo | 0.50 | 8 | 0.6245 | 0.1409 | 0.1024 | 0.0584 | 104 | 1055 | 277 |

Configuracion aplicada:

- `confidence_threshold`: `0.35`
- `nms_threshold`: `0.50`
- `max_detections`: `8`

Se selecciono `0.35` porque reduce falsos positivos frente a `0.30`, pero conserva bastante mas recall que `0.50`. El ajuste `0.50` es demasiado estricto para el estado actual del modelo.

## Artefactos finales

- `ml/models/custom-efficientdet-detector/efficientdet-detector.pt`: artefacto activo final de la epoca 8.

Los checkpoints, logs de progreso, runs y salidas visuales temporales fueron eliminados despues de documentar las metricas relevantes. Si el modelo se entrena nuevamente, el pipeline vuelve a generar esos archivos durante la corrida.

## Implementacion actual

Ya existe:

- Configuracion tecnica: `ml/training/configs/efficientdet.json`
- Script de entrenamiento: `scripts/training/train-efficientdet.ps1`
- Modulo de validacion: `ml/training/train_efficientdet_preflight.py`
- Trainer real integrado en `ml/training/local_train.py`
- Worker de inferencia backend: `ml/inference/efficientdet_worker.py`
- Script de prediccion visual: `scripts/prediction/predict-efficientdet.ps1`
- Carpeta de artefactos esperada: `ml/models/custom-efficientdet-detector`

El backend carga el artefacto final a traves del worker EfficientDet, aplica los umbrales definidos en `ml/registry.json` y devuelve cajas normalizadas al frontend. El preflight y el trainer se mantienen para futuras rondas de entrenamiento, pero no son necesarios para usar el modelo final actual.

## Comandos

Validar preparacion:

```powershell
.\scripts\training\train-efficientdet.ps1 -PreflightOnly
```

Instalar dependencias EfficientDet:

```powershell
.\scripts\training\train-efficientdet.ps1 -Install
```

Ese comando solo instala dependencias. No ejecuta preflight ni entrenamiento.

Entrenar desde cero:

```powershell
.\scripts\training\train-efficientdet.ps1
```

Entrenar sin descargar backbone preentrenado:

```powershell
.\scripts\training\train-efficientdet.ps1 -NoPretrainedBackbone
```

Instalar y entrenar en el mismo comando, solo si se quiere hacer ambas cosas explicitamente:

```powershell
.\scripts\training\train-efficientdet.ps1 -Install -TrainAfterInstall
```

Entrenamiento equivalente desde el trainer central:

```powershell
.\scripts\training\train-local-models.ps1 -Models efficientdet -Dataset voc-detect -Epochs 8 -Batch 1 -ImageSize 512 -SessionSamples 1024 -SessionCount 64 -ValidateEveryEpoch -PublishPartial
```

Prediccion visual con el artefacto activo:

```powershell
.\scripts\prediction\predict-efficientdet.ps1 -Image "C:\ruta\a\imagen.jpg"
```

## Implementacion del trainer

- Dataset wrapper propio: convierte etiquetas YOLO/VOC a cajas `yxyx`, redimensiona a imagen cuadrada y conserva escala para validacion.
- Modelo: `tf_efficientdet_d0` con `DetBenchTrain`.
- Checkpoints: se generan durante entrenamiento, pero no se conservan en la version final limpia.
- Best model: `efficientdet-detector.pt` es el unico artefacto final conservado.
- Validacion: usa el mismo calculo interno de `map50`, `map50_95`, `precision_at_50` y `recall_at_50`.
- Salidas visuales: se generan por epoca si `ValidationVisualLimit` es mayor que cero.
- Debug: las corridas con `-MaxSamples` escriben en `_debug` y no publican el artefacto final.

## Configuracion inicial propuesta

- Imagen: 512 px
- Batch: 1
- Epocas finales seleccionadas: 8
- Workers: 0
- Device: `auto`
- Modelo base: `tf_efficientdet_d0`
- Learning rate: `0.0005`
- Weight decay: `0.0005`
- Scheduler: StepLR cada 2 epocas con gamma `0.5`
- Metrica best: `precision_at_50`

## Estado de cierre

- Artefacto final consolidado como `efficientdet-detector.pt`.
- Umbral balanceado aplicado en `ml/registry.json`.
- Inferencia backend conectada mediante `efficientdet_worker.py`.
- Disponible desde el frontend como modelo solo imagen.
- Sin pendientes activos para esta version del modelo.

