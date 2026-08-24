# Entrenamiento remoto con Google Colab

El flujo Colab permite entrenar con una GPU remota y no consumir la GPU local.

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
- guardar artefactos en Google Drive.

## Uso recomendado

1. Abrir `ml/training/colab_gpu_training.ipynb`.
2. Seleccionar kernel Colab / Python 3.
3. Confirmar que el runtime sea GPU.
4. Ejecutar las celdas en orden.

Por defecto usa:

```python
SOURCE_MODE = "git"
SELECTED_PRESET = "yolo_coco128"
```

Ese preset entrena YOLO con COCO128 y es el flujo mas rapido para validar que Colab funciona.

## Presets disponibles

| Preset | Modelo | Dataset | Uso |
|---|---|---|---|
| `yolo_coco128` | YOLO | COCO128 | Prueba principal rapida en GPU remota. |
| `faster_voc_short` | Faster R-CNN | PASCAL VOC | Entrenamiento por secciones para evitar corridas largas. |
| `retinanet_voc_short` | RetinaNet | PASCAL VOC | Entrenamiento por secciones para comparar contra Faster. |

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
