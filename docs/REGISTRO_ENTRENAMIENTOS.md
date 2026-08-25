# Registro de entrenamientos y evaluaciones

Fecha de consolidacion: 2026-08-24

Este documento concentra la informacion relevante de los entrenamientos y evaluaciones ejecutados hasta ahora. Las carpetas generadas de `ml/runs` y las salidas visuales de evaluacion pueden eliminarse sin perder el historial tecnico resumido aqui.

## Modelos activos

| Modelo | Estado | Dataset | Artefacto activo | Epocas | Uso actual |
|---|---|---|---|---:|---|
| YOLOv8 / YOLOv11 | Listo | COCO128 | `ml/models/custom-yolo-v8-v11-detector/best.pt` | 25 | Detector backend rapido para 80 clases COCO. |
| Faster R-CNN | Entrenado | PASCAL VOC | `ml/models/custom-faster-rcnn-detector/model.pt` | 10 | Detector backend experimental para 20 clases VOC. |
| RetinaNet | Entrenado | PASCAL VOC | `ml/models/custom-retinanet-detector/model.pt` | 5 | Detector backend experimental para comparar contra Faster. |
| EfficientDet | Planificado | COCO128 | Sin artefacto entrenado | 0 | Scaffold pendiente. |
| DETR / RT-DETR | Planificado | COCO128 | Sin artefacto entrenado | 0 | Scaffold pendiente. |

## Datasets usados

| Dataset | Uso | Estado posterior a limpieza |
|---|---|---|
| `coco128-detect` | Entrenamiento YOLO inicial. | Se conserva en registro; los zips/datos locales pueden regenerarse. |
| `voc-detect` | Entrenamiento Faster R-CNN y RetinaNet. | Se conserva en registro; los zips/datos locales pueden regenerarse. |

Nota: para volver a entrenar Faster/Retina localmente sera necesario volver a preparar/descargar VOC.

## Entrenamiento YOLOv8 / YOLOv11

- Modelo: `custom-yolo-v8-v11-detector`
- Dataset: `coco128-detect`
- Imagen: 640 px
- Batch: 1
- Epocas: 25
- Artefactos conservados:
  - `ml/models/custom-yolo-v8-v11-detector/best.pt`
  - `ml/models/custom-yolo-v8-v11-detector/last.pt`
  - `ml/models/custom-yolo-v8-v11-detector/metrics.json`

Metricas registradas en `ml/registry.json`:

| Punto | Epoca | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|---:|
| Mejor mAP@50 | 15 | 0.69572 | 0.55029 | 0.65453 | 0.46379 |
| Mejor mAP@50:95 | 22 | 0.63754 | 0.60068 | 0.64730 | 0.46381 |
| Final | 25 | 0.62052 | 0.58966 | 0.64809 | 0.46016 |

Lectura: YOLO es el mejor candidato para iterar rapido y para usar como detector principal del producto mientras Faster/Retina quedan como comparativos.

## Entrenamiento Faster R-CNN

- Modelo: `custom-faster-rcnn-detector`
- Dataset: `voc-detect`
- Imagen: 512 px
- Batch: 1
- Epocas completadas: 10
- Artefactos conservados:
  - `ml/models/custom-faster-rcnn-detector/model.pt`
  - `ml/models/custom-faster-rcnn-detector/checkpoints/checkpoint_latest.pt`
  - `ml/models/custom-faster-rcnn-detector/training_progress.json`
  - `ml/models/custom-faster-rcnn-detector/training_events.jsonl`

Perdida por epoca:

| Epoca | Loss |
|---:|---:|
| 1 | 0.4245 |
| 2 | 0.3614 |
| 3 | 0.3504 |
| 4 | 0.3503 |
| 5 | 0.3538 |
| 6 | 0.3537 |
| 7 | 0.3549 |
| 8 | 0.3542 |
| 9 | 0.3527 |
| 10 | 0.3534 |

Lectura: la perdida se estabilizo desde la epoca 3. Las epocas adicionales no produjeron mejora en validacion.

## Entrenamiento RetinaNet

- Modelo: `custom-retinanet-detector`
- Dataset: `voc-detect`
- Imagen: 512 px
- Batch: 1
- Epocas completadas: 5
- Artefactos conservados:
  - `ml/models/custom-retinanet-detector/model.pt`
  - `ml/models/custom-retinanet-detector/checkpoints/checkpoint_latest.pt`
  - `ml/models/custom-retinanet-detector/training_progress.json`
  - `ml/models/custom-retinanet-detector/training_events.jsonl`

Perdida por epoca:

| Epoca | Loss |
|---:|---:|
| 1 | 1.0675 |
| 2 | 0.7410 |
| 3 | 0.6694 |
| 4 | 0.6361 |
| 5 | 0.6288 |

Lectura: RetinaNet si bajo perdida de forma clara hasta la epoca 5, pero su validacion visual y numerica sigue por debajo de lo necesario para produccion.

## Evaluaciones VOC consolidadas

Todas las evaluaciones siguientes se ejecutaron sobre 250 imagenes del split `val` de PASCAL VOC.

| Evaluacion | Modelo | mAP@50 | mAP@50:95 | Precision@50 | Recall@50 | Predicciones | FP | FN | Lectura |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Inicial 2026-08-24 12:21 | Faster R-CNN | 0.2153 | 0.0985 | 0.2364 | 0.5289 | 1434 | 1095 | 302 | Mucho recall, demasiado ruido. |
| Ajuste final 5 epocas | Faster R-CNN | 0.1987 | 0.0919 | 0.3099 | 0.4540 | 939 | 648 | 350 | Mejor equilibrio visual que el inicial. |
| Evaluacion 10 epocas | Faster R-CNN | 0.1331 | 0.0657 | 0.2156 | 0.3931 | 1169 | 917 | 389 | Empeoro respecto a 5 epocas. |
| Inicial 2026-08-24 12:21 | RetinaNet | 0.2164 | 0.0955 | 0.1188 | 0.5835 | 3148 | 2774 | 267 | Demasiadas cajas falsas. |
| RetinaNet estricto | RetinaNet | 0.1125 | 0.0529 | 0.5710 | 0.2824 | 317 | 136 | 460 | Muy limpio, pero pierde demasiados objetos. |
| RetinaNet final | RetinaNet | 0.1521 | 0.0675 | 0.3547 | 0.3885 | 702 | 453 | 392 | Compromiso razonable, aun experimental. |

## Conclusion tecnica

- El checkpoint activo de Faster R-CNN quedo en 10 epocas, pero la mejor evaluacion observada fue la de 5 epocas.
- No se conserva un checkpoint historico de Faster a 5 epocas dentro de `ml/models`, por lo que para recuperar ese punto haria falta reentrenar con guardado de `best checkpoint` por validacion.
- Seguir entrenando Faster con la configuracion actual no es recomendable: aumento falsos positivos y bajo mAP/precision/recall.
- Antes de nuevos entrenamientos largos conviene implementar validacion por epoca, early stopping y guardado automatico del mejor checkpoint.
- Para iteracion rapida, YOLO debe ser el modelo principal; Faster y RetinaNet deben quedar como laboratorio/comparativa.

## Recomendaciones siguientes

1. Preparar flujo Colab para entrenamientos pesados con GPU externa.
2. Agregar `best checkpoint` por validacion antes de reentrenar Faster/Retina.
3. Usar subconjuntos balanceados para pruebas cortas antes de entrenar VOC completo.
4. Reentrenar Faster solo si cambiamos learning rate, estrategia de congelamiento o validacion por epoca.
5. Mantener datasets descargados fuera del repositorio y regenerarlos cuando se necesite entrenar.

## Recomendaciones siguientes

1. Ejecutar primero `faster_voc_gpu_tuned` en Colab y conservar el `best_model.pt`, no necesariamente el ultimo `model.pt`.
2. Revisar `validation_history.json` despues de cada corrida para confirmar si la mejora real ocurre antes de la ultima epoca.
3. Entrenar RetinaNet con `retinanet_voc_gpu_tuned` solo despues de validar Faster, para comparar con la misma metodologia.
4. Si el mAP no mejora, priorizar dataset/labels y umbrales de inferencia antes de aumentar epocas.
5. Mantener datasets descargados fuera del repositorio y regenerarlos cuando se necesite entrenar.

## Flujo implementado para proximos entrenamientos

- El entrenador TorchVision ahora acepta learning rate, momentum, weight decay, scheduler StepLR y congelamiento temporal del backbone.
- Faster R-CNN y RetinaNet pueden validar al cierre de cada epoca con `--validate-every-epoch`.
- Se guarda `checkpoint_latest.pt` para reanudar y `checkpoint_best.pt` / `best_model.pt` cuando mejora la metrica configurada.
- La metrica recomendada para elegir el mejor punto es `map50_95`, porque castiga cajas imprecisas y no solo detecciones faciles.
- `early_stopping_patience` evita repetir entrenamientos largos cuando el modelo deja de mejorar.
- `show-training-progress.ps1` muestra progreso, ultima validacion, mejor epoca y ruta del best checkpoint.
