# YOLOv8 / YOLOv11

## Estado

Modelo backend activo para deteccion general de objetos.

- ID: `custom-yolo-v8-v11-detector`
- Runtime: `ultralytics`
- Tarea: deteccion de objetos
- Dataset usado: `coco128-detect`
- Artefacto activo: `ml/models/custom-yolo-v8-v11-detector/yolo-v8-v11-detector.pt`
- Politica de artefactos: se conserva solo el artefacto final con nombre del modelo para despliegue
- Clases: 80 clases COCO
- Tamano de imagen: 640 px
- Batch: 1
- Epocas entrenadas: 25

## Uso previsto

YOLO es el detector backend mas liviano y practico del proyecto para iteracion rapida. Sirve como referencia principal cuando se necesita respuesta relativamente rapida sobre clases COCO comunes: personas, vehiculos, animales, comida, muebles y objetos cotidianos.

## Metricas registradas

| Punto | Epoca | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|---:|
| Mejor mAP@50 | 15 | 0.69572 | 0.55029 | 0.65453 | 0.46379 |
| Mejor mAP@50:95 | 22 | 0.63754 | 0.60068 | 0.64730 | 0.46381 |
| Final | 25 | 0.62052 | 0.58966 | 0.64809 | 0.46016 |

## Configuracion de inferencia

- Umbral de confianza actual: `0.25`
- El modelo se carga desde backend con Ultralytics.
- Si Ultralytics no esta instalado en el entorno del backend, la inferencia usa el entorno ML externo configurado por `CIML_PYTHON` o `C:\Users\<usuario>\.ciml\venv`.

## Notas tecnicas

- El dataset COCO128 es pequeno; el modelo sirve como base funcional, no como detector final de produccion.
- Si se desea mayor robustez, el siguiente paso es entrenar con un dataset mayor o propio.
- Este modelo detecta clases COCO; no comparte exactamente las mismas clases que Faster R-CNN y RetinaNet entrenados sobre PASCAL VOC.
- `last.pt` y `metrics.json` fueron eliminados de `ml/models` despues de consolidar las metricas relevantes en este documento y en `ml/registry.json`.

## Estado de cierre

- Artefacto final consolidado como `yolo-v8-v11-detector.pt`.
- Inferencia backend conectada mediante runtime Ultralytics.
- Disponible desde el frontend como modelo solo imagen.
- Sin pendientes activos para esta version del modelo.

