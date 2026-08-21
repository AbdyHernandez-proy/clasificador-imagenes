import { ClassifierApiClient } from './apiClient.js';
import { requireElementById, requireSelector } from './dom.js';
import { ClassifierUI } from './ui.js';
import { ImageProcessor } from './imageProcessor.js';
import { ModelManager } from './modelManager.js';

const BROWSER_MODELS = [
    {
        id: 'browser:mobilenet',
        name: 'MobileNet (Navegador)',
        runtime: 'browser',
        modelType: 'mobilenet',
        supportsWebcam: true
    },
    {
        id: 'browser:coco-ssd',
        name: 'COCO-SSD (Navegador)',
        runtime: 'browser',
        modelType: 'coco-ssd',
        supportsWebcam: true
    }
];

class ImageClassifier {
    constructor() {
        this.ui = new ClassifierUI();
        this.imageProcessor = new ImageProcessor();
        this.modelManager = new ModelManager();
        this.apiClient = new ClassifierApiClient();
        this.availableModels = [...BROWSER_MODELS];
        this.currentModelInfo = BROWSER_MODELS[0];
        this.currentModel = null;
        this.webcamActive = false;
        this.webcamStream = null;
        this.processingFrame = false;
        this.runId = 0;
        
        this.initialize();
    }

    async initialize() {
        try {
            await this.loadBackendModels();
            this.ui.populateModelSelector(this.availableModels, this.currentModelInfo.id);
            await this.loadSelectedModel(this.currentModelInfo);
            this.ui.updateModelStatus('Listo');
            this.setupEventListeners();
        } catch (error) {
            console.error('Error al inicializar:', error);
            this.ui.updateModelStatus('Error al cargar modelo');
            this.ui.showError(this.getModelLoadErrorMessage(error));
        }
    }

    async loadBackendModels() {
        try {
            const response = await this.apiClient.listModels();
            const backendModels = (response.models || []).map((model) => ({
                id: `backend:${model.id}`,
                backendId: model.id,
                name: `${model.name || model.id} (Backend)`,
                runtime: 'backend',
                modelType: model.task || 'backend',
                supportsWebcam: false,
                metadata: model
            }));

            this.availableModels = [...BROWSER_MODELS, ...backendModels];
        } catch (error) {
            console.info('Backend no disponible; usando modelos del navegador.', error);
            this.availableModels = [...BROWSER_MODELS];
        }
    }

    setupEventListeners() {
        const imageInput = requireElementById('image-input');
        const modelSelect = requireElementById('model-select');
        const webcamToggle = requireElementById('webcam-toggle');

        imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        modelSelect.addEventListener('change', (e) => this.switchModel(e.target.value));
        webcamToggle.addEventListener('click', () => this.toggleWebcam());

        const uploadArea = requireSelector('.upload-area');
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
    }

    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!this.validateSelectedImage(file)) return;

        this.stopWebcam();
        await this.classifyImage(file);
    }

    async classifyImage(imageFile) {
        const runId = this.nextRunId();

        try {
            this.ui.showLoading();
            this.ui.clearDetections();
            const startTime = performance.now();
            const imageElement = await this.imageProcessor.loadImage(imageFile);
            const predictions = await this.getCurrentPredictions(imageElement, imageFile);

            if (!this.isCurrentRun(runId)) return;

            const endTime = performance.now();
            const processingTime = this.currentModelInfo.runtime === 'backend' && predictions.processing_time_ms
                ? predictions.processing_time_ms.toFixed(2)
                : (endTime - startTime).toFixed(2);
            const normalizedPredictions = this.normalizePredictions(predictions);

            this.ui.displayResults(normalizedPredictions, this.currentModelInfo.name);
            this.ui.updateStats(normalizedPredictions, processingTime);
            this.ui.showImage(imageElement);
            this.ui.renderDetections(normalizedPredictions, imageElement, this.modelManager.getCurrentModelType());
            this.ui.hideLoading();
        } catch (error) {
            if (!this.isCurrentRun(runId)) return;

            console.error('Error clasificando imagen:', error);
            this.ui.showError(error.message || 'Error al procesar la imagen.');
        }
    }

    async getCurrentPredictions(imageElement, imageFile) {
        if (this.currentModelInfo.runtime === 'backend') {
            return await this.apiClient.predict(imageFile, this.currentModelInfo.backendId);
        }

        const modelType = this.modelManager.getCurrentModelType();
        
        if (modelType === 'mobilenet') {
            return await this.currentModel.classify(imageElement, 5);
        } else if (modelType === 'coco-ssd') {
            return await this.currentModel.detect(imageElement);
        }

        throw new Error('No hay un modelo valido cargado para procesar la imagen.');
    }

    async switchModel(modelId) {
        const runId = this.nextRunId();
        const nextModel = this.availableModels.find((model) => model.id === modelId);

        if (!nextModel) {
            this.ui.showError('El modelo seleccionado no esta disponible.');
            return;
        }

        try {
            this.ui.updateModelStatus('Cambiando modelo...');
            this.stopWebcam({ invalidate: false });
            await this.loadSelectedModel(nextModel);
            if (!this.isCurrentRun(runId)) return;
            this.ui.updateModelStatus('Listo');
            this.ui.clearResults();
        } catch (error) {
            if (!this.isCurrentRun(runId)) return;

            console.error('Error cambiando modelo:', error);
            this.ui.updateModelStatus('Error al cambiar modelo');
            this.ui.showError(this.getModelLoadErrorMessage(error));
        }
    }

    async loadSelectedModel(modelInfo) {
        this.currentModelInfo = modelInfo;

        if (modelInfo.runtime === 'backend') {
            this.currentModel = null;
            this.modelManager.clearCurrentModel();
            this.ui.updateModelName(modelInfo.name);
            return;
        }

        await this.modelManager.loadModel(modelInfo.modelType);
        this.currentModel = this.modelManager.getCurrentModel();
    }

    async toggleWebcam() {
        if (this.webcamActive) {
            this.stopWebcam();
        } else {
            await this.startWebcam();
        }
    }

    async startWebcam() {
        if (!this.currentModelInfo.supportsWebcam) {
            this.ui.showError('La webcam solo esta disponible con modelos del navegador.');
            return;
        }

        if (!navigator.mediaDevices?.getUserMedia) {
            this.ui.showError('La webcam requiere un navegador compatible y ejecucion en localhost o HTTPS.');
            return;
        }

        const runId = this.nextRunId();

        try {
            this.webcamActive = true;
            this.ui.clearDetections();
            const video = requireElementById('webcam-video');
            const previewImage = requireElementById('preview-image');

            this.webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });

            if (!this.isCurrentRun(runId)) {
                this.webcamStream.getTracks().forEach(track => track.stop());
                this.webcamStream = null;
                return;
            }

            video.srcObject = this.webcamStream;
            video.style.display = 'block';
            previewImage.style.display = 'none';

            video.onloadedmetadata = () => {
                video.play();
                this.processWebcamFrames(video, runId);
            };

            requireElementById('webcam-toggle').textContent = 'Detener Webcam';
        } catch (error) {
            if (!this.isCurrentRun(runId)) return;

            console.error('Error accediendo a webcam:', error);
            this.ui.showError(this.getWebcamErrorMessage(error));
            this.webcamActive = false;
            this.processingFrame = false;
        }
    }

    async processWebcamFrames(video, runId) {
        if (!this.webcamActive || this.processingFrame || !this.isCurrentRun(runId)) return;

        this.processingFrame = true;

        try {
            const startTime = performance.now();
            const predictions = await this.getCurrentPredictions(video, null);

            if (!this.isCurrentRun(runId)) return;

            const endTime = performance.now();
            const processingTime = (endTime - startTime).toFixed(2);
            const normalizedPredictions = this.normalizePredictions(predictions);

            this.ui.displayResults(normalizedPredictions, this.currentModelInfo.name);
            this.ui.updateStats(normalizedPredictions, processingTime);
            this.ui.renderDetections(normalizedPredictions, video, this.modelManager.getCurrentModelType());
        } catch (error) {
            if (!this.isCurrentRun(runId)) return;

            console.error('Error procesando frame de webcam:', error);
            this.ui.showError(error.message || 'Error al procesar la imagen de la webcam.');
        } finally {
            this.processingFrame = false;

            if (this.webcamActive && this.isCurrentRun(runId)) {
                requestAnimationFrame(() => this.processWebcamFrames(video, runId));
            }
        }
    }

    stopWebcam(options = {}) {
        const { invalidate = true } = options;

        if (invalidate) {
            this.nextRunId();
        }

        if (this.webcamStream) {
            this.webcamStream.getTracks().forEach(track => track.stop());
            this.webcamStream = null;
        }

        const video = requireElementById('webcam-video');
        const previewImage = requireElementById('preview-image');
        
        video.style.display = 'none';
        previewImage.style.display = 'block';
        
        this.webcamActive = false;
        this.processingFrame = false;
        this.ui.clearDetections();
        requireElementById('webcam-toggle').textContent = 'Usar Webcam';
    }

    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('drag-over');
    }

    handleDragLeave(event) {
        event.currentTarget.classList.remove('drag-over');
    }

    handleDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('drag-over');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const imageFile = files[0];
            if (!this.validateSelectedImage(imageFile)) return;

            this.stopWebcam();
            this.classifyImage(imageFile);
        }
    }

    validateSelectedImage(file) {
        try {
            this.imageProcessor.validateImageFile(file);
            return true;
        } catch (error) {
            this.ui.showError(error.message);
            return false;
        }
    }

    normalizePredictions(predictions) {
        if (Array.isArray(predictions)) return predictions;
        return predictions.predictions || [];
    }

    nextRunId() {
        this.runId += 1;
        return this.runId;
    }

    isCurrentRun(runId) {
        return runId === this.runId;
    }

    getModelLoadErrorMessage(error) {
        if (!navigator.onLine) {
            return 'No hay conexion a internet para cargar el modelo de IA.';
        }

        return error.message || 'No se pudo cargar el modelo de IA.';
    }

    getWebcamErrorMessage(error) {
        if (error.name === 'NotAllowedError') {
            return 'Permiso de camara denegado. Autoriza la webcam en el navegador.';
        }

        if (error.name === 'NotFoundError') {
            return 'No se encontro una camara disponible en este equipo.';
        }

        if (error.name === 'NotReadableError') {
            return 'La camara esta ocupada o no se puede leer desde el navegador.';
        }

        return error.message || 'No se puede acceder a la webcam.';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    try {
        new ImageClassifier();
    } catch (error) {
        console.error('Error al iniciar la aplicacion:', error);
        const message = document.createElement('p');
        message.className = 'error-message';
        message.textContent = error.message || 'No se pudo iniciar la aplicacion.';
        document.body.prepend(message);
    }
});
