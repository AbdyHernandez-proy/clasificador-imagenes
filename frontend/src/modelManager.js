const MODEL_URLS = {
    mobilenet: null,
    'coco-ssd': null,
    blazeface: null,
    handpose: null,
    'body-pix': null
};

export class ModelManager {
    constructor(modelUrls = MODEL_URLS) {
        this.models = {};
        this.modelUrls = modelUrls;
        this.currentModelType = null;
        this.currentModel = null;
        this.backendReady = false;
    }

    async loadModel(modelType) {
        try {
            await this.ensureTensorFlowBackend();

            // Si el modelo ya esta cargado, usarlo.
            if (this.models[modelType]) {
                this.currentModelType = modelType;
                this.currentModel = this.models[modelType];
                this.updateModelInfo();
                return;
            }

            // Cargar nuevo modelo.
            if (modelType === 'mobilenet') {
                console.log('Cargando MobileNet...');
                const mobilenet = await import('@tensorflow-models/mobilenet');
                const modelConfig = this.getModelConfig('mobilenet');
                const model = await mobilenet.load(modelConfig);
                this.models['mobilenet'] = model;
                this.currentModelType = 'mobilenet';
                this.currentModel = model;
            } else if (modelType === 'coco-ssd') {
                console.log('Cargando COCO-SSD...');
                const cocoSsd = await import('@tensorflow-models/coco-ssd');
                const modelConfig = this.getModelConfig('coco-ssd');
                const model = await cocoSsd.load(modelConfig);
                this.models['coco-ssd'] = model;
                this.currentModelType = 'coco-ssd';
                this.currentModel = model;
            } else if (modelType === 'blazeface') {
                console.log('Cargando BlazeFace...');
                const blazeface = await import('@tensorflow-models/blazeface');
                const model = await blazeface.load();
                this.models.blazeface = model;
                this.currentModelType = 'blazeface';
                this.currentModel = model;
            } else if (modelType === 'handpose') {
                console.log('Cargando HandPose...');
                const handpose = await import('@tensorflow-models/handpose');
                const model = await handpose.load({
                    detectionConfidence: 0.82,
                    scoreThreshold: 0.75,
                    iouThreshold: 0.3,
                    maxContinuousChecks: 5
                });
                this.models.handpose = model;
                this.currentModelType = 'handpose';
                this.currentModel = model;
            } else if (modelType === 'body-pix') {
                console.log('Cargando BodyPix...');
                const bodyPix = await import('@tensorflow-models/body-pix');
                const model = await bodyPix.load({
                    architecture: 'MobileNetV1',
                    outputStride: 16,
                    multiplier: 0.75,
                    quantBytes: 2
                });
                this.models['body-pix'] = model;
                this.currentModelType = 'body-pix';
                this.currentModel = model;
            } else {
                throw new Error(`Modelo no soportado: ${modelType}`);
            }

            this.updateModelInfo();
        } catch (error) {
            console.error(`Error cargando modelo ${modelType}:`, error);
            throw error;
        }
    }

    async ensureTensorFlowBackend() {
        if (this.backendReady) return;

        const tf = await import('@tensorflow/tfjs');

        try {
            await import('@tensorflow/tfjs-backend-webgl');
            const webglReady = await tf.setBackend('webgl');
            if (!webglReady) {
                throw new Error('WebGL no esta disponible.');
            }
        } catch (error) {
            console.warn('No se pudo usar WebGL; usando CPU para TensorFlow.js.', error);
            await import('@tensorflow/tfjs-backend-cpu');
            const cpuReady = await tf.setBackend('cpu');
            if (!cpuReady) {
                throw new Error('No se pudo inicializar TensorFlow.js.');
            }
        }

        await tf.ready();
        this.backendReady = true;
    }

    getModelConfig(modelType) {
        const modelUrl = this.modelUrls[modelType];
        return modelUrl ? { modelUrl } : undefined;
    }

    clearCurrentModel() {
        this.currentModelType = null;
        this.currentModel = null;
    }

    updateModelInfo() {
        const modelNameEl = document.getElementById('model-name');
        if (modelNameEl) {
            modelNameEl.textContent = this.getCurrentModelName();
        }
    }

    getCurrentModel() {
        return this.currentModel;
    }

    getCurrentModelType() {
        return this.currentModelType;
    }

    getCurrentModelName() {
        if (this.currentModelType === 'mobilenet') {
            return 'MobileNet';
        } else if (this.currentModelType === 'coco-ssd') {
            return 'COCO-SSD';
        } else if (this.currentModelType === 'blazeface') {
            return 'BlazeFace';
        } else if (this.currentModelType === 'handpose') {
            return 'HandPose';
        } else if (this.currentModelType === 'body-pix') {
            return 'BodyPix';
        }
        return 'Desconocido';
    }

    disposeModel(modelType) {
        if (this.models[modelType]?.dispose) {
            this.models[modelType].dispose();
        }

        if (this.models[modelType]) {
            delete this.models[modelType];
        }
    }

    disposeAll() {
        Object.keys(this.models).forEach(modelType => {
            this.disposeModel(modelType);
        });
    }
}
