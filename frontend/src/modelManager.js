const MODEL_URLS = {
    mobilenet: null,
    'coco-ssd': null
};

export class ModelManager {
    constructor(modelUrls = MODEL_URLS) {
        this.models = {};
        this.modelUrls = modelUrls;
        this.currentModelType = null;
        this.currentModel = null;
    }

    async loadModel(modelType) {
        try {
            // Si el modelo ya está cargado, usarlo
            if (this.models[modelType]) {
                this.currentModelType = modelType;
                this.currentModel = this.models[modelType];
                this.updateModelInfo();
                return;
            }

            // Cargar nuevo modelo
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
            } else {
                throw new Error(`Modelo no soportado: ${modelType}`);
            }

            this.updateModelInfo();
        } catch (error) {
            console.error(`Error cargando modelo ${modelType}:`, error);
            throw error;
        }
    }

    getModelConfig(modelType) {
        const modelUrl = this.modelUrls[modelType];
        return modelUrl ? { modelUrl } : undefined;
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
        }
        return 'Desconocido';
    }

    disposeModel(modelType) {
        if (this.models[modelType]) {
            this.models[modelType].dispose();
            delete this.models[modelType];
        }
    }

    disposeAll() {
        Object.keys(this.models).forEach(modelType => {
            this.disposeModel(modelType);
        });
    }
}
