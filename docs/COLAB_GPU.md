# Entrenamiento remoto con Google Colab

El flujo Colab permite entrenar con una GPU remota y no consumir la GPU local. Esta preparado para entrenamientos largos con checkpoints, validacion por epoca, seleccion automatica del mejor modelo y early stopping.

## Notebook principal

```text
ml/training/colab_gpu_training.ipynb
```

El notebook ya esta preparado para:

- verificar GPU CUDA remota,
- clonar el repositorio desde GitHub,
- instalar dependencias ML,
- preparar `coco128-detect` o `voc-detect`,
- entrenar YOLO, Faster R-CNN o RetinaNet,
- validar Faster/Retina al cierre de cada epoca,
- guardar `checkpoint_latest.pt`, `checkpoint_best.pt`, `model.pt`, `best_model.pt` y `validation_history.json`,
- guardar artefactos en Google Drive.

## Uso recomendado

1. Abrir `ml/training/colab_gpu_training.ipynb`.
2. Seleccionar runtime con GPU.
3. Ejecutar las celdas en orden.
4. Para Faster R-CNN usar `SELECTED_PRESET = "faster_voc_gpu_tuned"`.
5. Para RetinaNet usar `SELECTED_PRESET = "retinanet_voc_gpu_tuned"`.

Por defecto el notebook usa:

```python
SOURCE_MODE = "git"
SELECTED_PRESET = "faster_voc_gpu_tuned"
```

## Presets disponibles

| Preset | Modelo | Dataset | Validacion | Uso |
|---|---|---|---|---|
| `yolo_coco128` | YOLO | COCO128 | Ultralytics interna | Prueba rapida de GPU y detector principal actual. |
| `faster_voc_gpu_tuned` | Faster R-CNN | PASCAL VOC | 500 imagenes por epoca | Reentrenamiento robusto con LR bajo, backbone congelado al inicio y best checkpoint. |
| `retinanet_voc_gpu_tuned` | RetinaNet | PASCAL VOC | 500 imagenes por epoca | Comparativa con LR mas conservador y control de falsos positivos. |

## Controles nuevos

| Parametro | Uso |
|---|---|
| `learning_rate` | Reduce o aumenta la velocidad de aprendizaje. En Faster/Retina se usa mas bajo que antes para evitar degradacion. |
| `freeze_backbone_epochs` | Congela el extractor ResNet al inicio para estabilizar las cabezas de deteccion. |
| `lr_step_size` / `lr_gamma` | Reduce el learning rate cada N epocas. |
| `validate_every_epoch` | Ejecuta validacion al cerrar cada epoca. |
| `best_metric` | Metrica que decide el mejor checkpoint. Por defecto `map50_95`. |
| `early_stopping_patience` | Detiene el entrenamiento si no mejora durante varias epocas. |
| `checkpoint_every` | Guarda checkpoint durante la epoca cada N batches. |

## Cuando GitHub no tenga los ultimos cambios

Si quieres entrenar cambios locales sin subirlos, genera un ZIP liviano:

```powershell
.\scripts\package-colab-project.ps1
```

Luego cambia en el notebook:

```python
SOURCE_MODE = "upload_zip"
```

Y sube el ZIP generado en:

```text
exports/colab/clasificador-imagenes-colab.zip
```

## Salida

Si `USE_DRIVE = True`, el notebook guarda resultados en:

```text
/content/drive/MyDrive/clasificador-imagenes-colab/run-...
```

Dentro quedan:

- `ml/models/`
- `ml/registry.json`
- `ml/datasets/registry.json`
- `docs/REGISTRO_ENTRENAMIENTOS.md`

Al terminar una corrida, descarga el ZIP generado en Drive y avisa para integrarlo al proyecto local.
