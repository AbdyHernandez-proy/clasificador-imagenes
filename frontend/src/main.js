import { requireElementById, requireSelector } from './dom.js';
import { ClassifierUI } from './ui.js';
import { ImageProcessor } from './imageProcessor.js';
import { ModelManager } from './modelManager.js';

class ImageClassifier {
    constructor() {
        this.ui = new ClassifierUI();
        this.imageProcessor = new ImageProcessor();
        this.modelManager = new ModelManager();
        this.currentModel = null;
        this.webcamActive = false;
        this.webcamStream = null;
        this.processingFrame = false;
        this.runId = 0;
        
        this.initialize();
    }

    async initialize() {
        try {
            // Cargar modelo inicial
            await this.modelManager.loadModel('mobilenet');
            this.currentModel = this.modelManager.getCurrentModel();
            this.ui.updateModelStatus('Listo');
            this.setupEventListeners();
        } catch (error) {
            console.error('Error al inicializar:', error);
            this.ui.updateModelStatus('Error al cargar modelo');
            this.ui.showError(this.getModelLoadErrorMessage(error));
        }
    }

    setupEventListeners() {
        const imageInput = requireElementById('image-input');
        const modelSelect = requireElementById('model-select');
        const webcamToggle = requireElementById('webcam-toggle');

        imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        modelSelect.addEventListener('change', (e) => this.switchModel(e.target.value));
        webcamToggle.addEventListener('click', () => this.toggleWebcam());

        // Drag and drop
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

            // Procesar imagen
            const imageElement = await this.imageProcessor.loadImage(imageFile);
            const predictions = await this.getCurrentPredictions(imageElement);

            if (!this.isCurrentRun(runId)) return;

            const endTime = performance.now();
            const processingTime = (endTime - startTime).toFixed(2);
            const modelName = this.modelManager.getCurrentModelName();

            // Mostrar resultados
            this.ui.displayResults(predictions, modelName);
            this.ui.updateStats(predictions, processingTime);
            this.ui.showImage(imageElement);
            this.ui.renderDetections(predictions, imageElement, this.modelManager.getCurrentModelType());
            this.ui.hideLoading();
        } catch (error) {
            if (!this.isCurrentRun(runId)) return;

            console.error('Error clasificando imagen:', error);
            this.ui.showError(error.message || 'Error al procesar la imagen.');
        }
    }

    async getCurrentPredictions(imageElement) {
        const modelType = this.modelManager.getCurrentModelType();
        
        if (modelType === 'mobilenet') {
            return await this.currentModel.classify(imageElement, 5);
        } else if (modelType === 'coco-ssd') {
            return await this.currentModel.detect(imageElement);
        }

        throw new Error('No hay un modelo valido cargado para procesar la imagen.');
    }

    async switchModel(modelName) {
        const runId = this.nextRunId();

        try {
            this.ui.updateModelStatus('Cambiando modelo...');
            this.stopWebcam({ invalidate: false });
            
            await this.modelManager.loadModel(modelName);
            if (!this.isCurrentRun(runId)) return;

            this.currentModel = this.modelManager.getCurrentModel();
            
            this.ui.updateModelStatus('Listo');
            this.ui.clearResults();
        } catch (error) {
            if (!this.isCurrentRun(runId)) return;

            console.error('Error cambiando modelo:', error);
            this.ui.updateModelStatus('Error al cambiar modelo');
            this.ui.showError(this.getModelLoadErrorMessage(error));
        }
    }

    async toggleWebcam() {
        if (this.webcamActive) {
            this.stopWebcam();
        } else {
            await this.startWebcam();
        }
    }

    async startWebcam() {
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

            // Obtener acceso a la webcam
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

            // Procesar frames
            video.onloadedmetadata = () => {
                video.play();
                this.processWebcamFrames(video, runId);
            };

            requireElementById('webcam-toggle').textContent = '⏹️ Detener Webcam';
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
            const predictions = await this.getCurrentPredictions(video);

            if (!this.isCurrentRun(runId)) return;

            const endTime = performance.now();
            const processingTime = (endTime - startTime).toFixed(2);

            this.ui.displayResults(predictions, this.modelManager.getCurrentModelName());
            this.ui.updateStats(predictions, processingTime);
            this.ui.renderDetections(predictions, video, this.modelManager.getCurrentModelType());
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
        requireElementById('webcam-toggle').textContent = '📹 Usar Webcam';
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

// Inicializar cuando el DOM esté listo
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
