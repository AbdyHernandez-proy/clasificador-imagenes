export const BROWSER_MODELS = [
    {
        id: 'browser:mobilenet',
        name: 'MobileNet',
        usageLabel: 'Imagenes',
        runtime: 'browser',
        modelType: 'mobilenet',
        task: 'Clasificacion de imagenes',
        algorithm: 'CNN MobileNet',
        efficiency: 'Rapido',
        supportsWebcam: true,
        description: 'Clasifica la imagen completa y devuelve las etiquetas generales mas probables. Es adecuado para reconocer escenas u objetos dominantes, pero no localiza regiones especificas dentro de la imagen.'
    },
    {
        id: 'browser:coco-ssd',
        name: 'COCO-SSD',
        usageLabel: 'Objetos',
        runtime: 'browser',
        modelType: 'coco-ssd',
        task: 'Deteccion de objetos',
        algorithm: 'Single Shot Detector',
        efficiency: 'Medio',
        supportsWebcam: true,
        description: 'Detecta multiples objetos comunes, como personas, vehiculos, animales y elementos cotidianos. Devuelve una caja por objeto detectado y funciona bien para imagenes con varios elementos.'
    },
    {
        id: 'browser:blazeface',
        name: 'BlazeFace',
        usageLabel: 'Rostros',
        runtime: 'browser',
        modelType: 'blazeface',
        task: 'Deteccion de rostros',
        algorithm: 'Detector ligero inspirado en SSD',
        efficiency: 'Rapido',
        supportsWebcam: true,
        description: 'Detecta rostros con baja latencia y usa puntos faciales basicos para ajustar mejor el recuadro. Es util para webcam, selfies o validaciones rapidas de presencia facial.'
    },
    {
        id: 'browser:handpose',
        name: 'HandPose',
        usageLabel: 'Manos',
        runtime: 'browser',
        modelType: 'handpose',
        task: 'Deteccion de manos',
        algorithm: 'Regresion de 21 puntos de mano',
        efficiency: 'Lento',
        supportsWebcam: true,
        description: 'Estima 21 puntos clave de la mano para ubicar dedos, palma y muneca. Es mas pesado que los detectores simples, pero permite recuadros ajustados a la postura real de la mano.'
    },
    {
        id: 'browser:body-pix',
        name: 'BodyPix',
        usageLabel: 'Personas',
        runtime: 'browser',
        modelType: 'body-pix',
        task: 'Segmentacion de persona',
        algorithm: 'Segmentacion con MobileNet',
        efficiency: 'Lento',
        supportsWebcam: true,
        description: 'Segmenta pixeles pertenecientes a una persona y separa la silueta del fondo. Es util para mascaras, presencia humana y recorte aproximado, aunque consume mas recursos en webcam.'
    }
];

export const CANDIDATE_MODELS = [
    {
        id: 'candidate:deeplab-v3',
        name: 'DeepLab v3',
        task: 'Segmentacion semantica',
        algorithm: 'Atrous convolution / ASPP',
        reason: 'El paquete oficial de TensorFlow.js exige dependencias antiguas de tfjs-converter y no conviene forzarlo sobre TensorFlow.js 4.x.'
    },
    {
        id: 'candidate:yolo-onnx',
        name: 'YOLO via ONNX Runtime Web',
        task: 'Deteccion de objetos en imagen/video',
        algorithm: 'YOLO + ONNX Runtime Web',
        reason: 'Requiere integrar onnxruntime-web y almacenar pesos ONNX externos; incluso YOLOv8n ronda decenas de MB y debe validarse licencia/tamano antes de versionarlo.'
    },
    {
        id: 'candidate:movenet',
        name: 'MoveNet / Pose Detection',
        task: 'Deteccion de pose humana',
        algorithm: 'Pose estimation ultraligera',
        reason: 'El paquete oficial pose-detection falla en build con Vite 8 por una exportacion de MediaPipe; conviene integrarlo con una ruta aislada o CDN.'
    }
];
