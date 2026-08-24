# Catalogo de modelos

## Frontend

Estos modelos se ejecutan en el navegador mediante TensorFlow.js. Sus pesos no se guardan en el repositorio.

| Modelo | Funcion | Uso |
|---|---|---|
| MobileNet | Clasificacion de imagen completa | Etiquetas generales de ImageNet. |
| COCO-SSD | Deteccion de objetos | Cajas de objetos comunes desde navegador. |
| BlazeFace | Deteccion de rostros | Rostros con baja latencia. |
| HandPose | Deteccion de manos | Mano y 21 puntos clave. |
| BodyPix | Segmentacion de persona | Separacion persona/fondo. |

## Backend activos

| Modelo | Estado | Funcion | Artefacto |
|---|---|---|---|
| `visual-profile-v1` | Listo | Perfil visual basico sin entrenamiento. | Interno. |
| `custom-yolo-v8-v11-detector` | Listo | Deteccion general COCO 80 clases. | `ml/models/custom-yolo-v8-v11-detector/best.pt` |
| `custom-faster-rcnn-detector` | Entrenado experimental | Deteccion VOC 20 clases. | `ml/models/custom-faster-rcnn-detector/model.pt` |
| `custom-retinanet-detector` | Entrenado experimental | Deteccion VOC 20 clases. | `ml/models/custom-retinanet-detector/model.pt` |

## Backend planificados

| Modelo | Estado | Objetivo |
|---|---|---|
| `custom-classifier-template` | Planificado | Base para clasificador propio. |
| `custom-efficientdet-detector` | Planificado | Comparar precision/consumo con EfficientDet. |
| `custom-detr-rtdetr-detector` | Planificado | Comparar arquitectura Transformer contra CNN/YOLO. |

## Lectura actual

- YOLO es el candidato principal para iteracion rapida y producto inicial.
- Faster R-CNN y RetinaNet funcionan desde backend, pero siguen siendo experimentales.
- Faster a 10 epocas no supero la evaluacion de 5 epocas; revisar `docs/REGISTRO_ENTRENAMIENTOS.md`.
- Antes de nuevos entrenamientos largos conviene usar Colab, validacion por epoca y guardado de mejor checkpoint.

## Fuentes utiles

- TensorFlow.js Models: https://github.com/tensorflow/tfjs-models
- Ultralytics YOLO: https://docs.ultralytics.com/
- TorchVision detection models: https://pytorch.org/vision/stable/models.html#object-detection
- ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/
