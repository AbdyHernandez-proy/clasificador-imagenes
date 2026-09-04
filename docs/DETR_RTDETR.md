# DETR / RT-DETR

## Estado

Modelo acoplado al flujo local de entrenamiento.

- ID: `custom-detr-rtdetr-detector`
- Runtime: `ultralytics`
- Arquitectura objetivo: `RTDETR`
- Pesos base recomendados: `rtdetr-l.pt`
- Tarea: deteccion de objetos
- Dataset objetivo: `voc-detect`
- Artefacto final activo: `ml/models/custom-detr-rtdetr-detector/rtdetr-detector.pt`
- Politica de artefactos: se conserva solo el artefacto final con nombre del modelo para despliegue
- Estado actual: finalizado
- Servicio frontend/backend: activo mediante runtime Ultralytics

## Uso previsto

RT-DETR se usara como comparativo de arquitectura transformer contra Faster R-CNN y RetinaNet. La idea es evaluar si logra detecciones mas limpias o mejor relacion precision/recall sin depender de anchors clasicos.

## Dataset

- Dataset: `voc-detect`
- Formato interno: YOLO
- Entrenamiento: 60% = 12,902 imagenes
- Validacion: 40% = 8,601 imagenes
- Clases: 20 clases PASCAL VOC

## Configuracion de entrenamiento

- Imagen: 512 px
- Batch: 1
- Epocas completadas: 10
- Workers: 0
- Device: `auto`
- Pesos base: `rtdetr-l.pt`

Esta configuracion replica el punto de partida de RetinaNet para que la comparacion sea razonable. RT-DETR usa el motor de Ultralytics: durante el entrenamiento guarda checkpoints por epoca, muestra progreso por epoca/lote y copia el artefacto final a `ml/models/custom-detr-rtdetr-detector`.

Nota: RT-DETR no usa `SessionSamples` como Faster R-CNN, RetinaNet o EfficientDet. El entrenamiento se controla por epocas de Ultralytics. Tras la limpieza final se conserva solo el artefacto final con nombre del modelo.

## Comandos

Instalar/verificar dependencias RT-DETR:

```powershell
.\scripts\training\train-rtdetr.ps1 -Install
```

Ese comando solo instala/verifica dependencias. No inicia entrenamiento.

Entrenar desde cero:

```powershell
.\scripts\training\train-rtdetr.ps1
```

Continuar entrenamiento si existen checkpoints completos de una corrida activa:

```powershell
.\scripts\training\train-rtdetr.ps1 -Resume
```

Instalar y entrenar en el mismo comando, solo si se quiere hacer ambas cosas explicitamente:

```powershell
.\scripts\training\train-rtdetr.ps1 -Install -TrainAfterInstall
```

Entrenamiento equivalente desde el trainer central:

```powershell
.\scripts\training\train-local-models.ps1 -Models rtdetr -Dataset voc-detect -Epochs 10 -Batch 1 -ImageSize 512 -Workers 0 -Resume
```

## Artefactos

Artefacto conservado para despliegue:

- El mejor peso generado por Ultralytics se consolido como `ml/models/custom-detr-rtdetr-detector/rtdetr-detector.pt`

Tras la aprobacion visual final se limpiaron los runs temporales, checkpoints pesados por epoca, pruebas smoke, salidas manuales, `last.pt`, `metrics.json` y `training_summary.json`. Las metricas relevantes quedaron consolidadas en este documento y en `ml/registry.json`.

## Resultado actual

Entrenamiento completado sobre `voc-detect` con split 60/40.

- Mejor epoca: 10
- Precision: `0.79488`
- Recall: `0.74635`
- mAP@50: `0.80380`
- mAP@50:95: `0.62674`

## Calibracion aplicada

La calibracion comparo varios umbrales sobre 250 imagenes del split de validacion. El ajuste aplicado prioriza una precision mas limpia con una perdida moderada de recall frente al ajuste anterior.

- Confidence: `0.45`
- IoU: `0.55`
- Max cajas: `8`
- Score de calibracion: `0.666995`
- Precision@50 calibrada: `0.6592`
- Recall@50 calibrado: `0.8068`
- mAP@50 calibrado: `0.7041`
- mAP@50:95 calibrado: `0.5590`
- Falsos positivos en muestra calibrada: `259`
- Predicciones en muestra calibrada: `760`
- Las salidas temporales de calibracion fueron limpiadas; se conservan aqui los valores aplicados.

## Criterios de aceptacion

- Entrenamiento completado a 10 epocas sin fallos.
- `rtdetr-detector.pt` conservado en `ml/models/custom-detr-rtdetr-detector`.
- Metricas finales registradas en este documento y en `ml/registry.json`.
- Inferencia visual manual probada con imagen real.
- Umbrales calibrados antes de exposicion frontend/backend.

## Estado de cierre

- `serve=true` en `ml/registry.json`.
- Artefacto final consolidado como `rtdetr-detector.pt`.
- Inferencia backend conectada mediante runtime Ultralytics.
- Disponible desde el frontend como modelo solo imagen.
- Sin pendientes activos para esta version del modelo.

