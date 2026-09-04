import { ClassifierApiClient } from './apiClient.js';
import { requireElementById, requireSelector } from './dom.js';
import { ClassifierUI } from './ui.js';
import { ImageProcessor } from './imageProcessor.js';
import { BROWSER_MODELS, CUSTOM_MODEL_SLOTS } from './modelCatalog.js';
import { ModelManager } from './modelManager.js';

const CONNECTABLE_BACKEND_STATUSES = new Set(['ready', 'trained']);

const MODEL_STATUS_LABELS = {
    ready: 'Listo',
    trained: 'Entrenado',
    training: 'Entrenando',
    partial: 'Parcial',
    planned: 'Planificado',
    failed: 'Con error',
    unavailable: 'No disponible'
};

const TASK_LABELS = {
    object_detection: 'Deteccion de objetos',
    image_classification: 'Clasificacion de imagenes',
    image_profile_classification: 'Perfil visual'
};

class ImageClassifier {
    constructor() {
        this.ui = new ClassifierUI();
        this.imageProcessor = new ImageProcessor();
        this.modelManager = new ModelManager();
        this.apiClient = new ClassifierApiClient();
        this.availableModels = this.buildModelCatalog();
        this.currentModelInfo = BROWSER_MODELS[0];
        this.currentModel = null;
        this.webcamActive = false;
        this.webcamStream = null;
        this.processingFrame = false;
        this.lastWebcamInferenceAt = 0;
        this.uploadedImageActive = false;
        this.currentImageFile = null;
        this.runId = 0;
        this.eventListenersReady = false;
        
        this.initialize();
    }

    async initialize() {
        try {
            await this.loadBackendModels();
            this.ui.populateModelSelector(this.availableModels, this.currentModelInfo.id);
            this.setupEventListeners();
            await this.loadSelectedModel(this.currentModelInfo);
            this.ui.updateModelStatus('Listo');
            this.preloadImageOnlyModels();
        } catch (error) {
            console.error('Error al inicializar:', error);
            this.ui.updateModelStatus('Error al cargar modelo');
            this.ui.showError(this.getModelLoadErrorMessage(error));
        }
    }

    async loadBackendModels() {
        const servedModelsResult = await Promise.allSettled([
            this.apiClient.listModels(),
            this.apiClient.listModelRegistry()
        ]);
        const modelsResponse = servedModelsResult[0].status === 'fulfilled' ? servedModelsResult[0].value : null;
        const registryResponse = servedModelsResult[1].status === 'fulfilled' ? servedModelsResult[1].value : null;

        if (!modelsResponse && !registryResponse) {
            console.info('Backend no disponible; usando modelos del navegador y espacios backend en espera.');
            this.availableModels = this.buildModelCatalog();
            return;
        }

        this.availableModels = this.buildModelCatalog({
            backendAvailable: true,
            servedModels: modelsResponse?.models || [],
            registryModels: registryResponse?.models || []
        });
    }

    buildModelCatalog({ backendAvailable = false, servedModels = [], registryModels = [] } = {}) {
        const servedById = new Map(servedModels.map((model) => [model.id, model]));
        const registryById = new Map(registryModels.map((model) => [model.id, model]));
        const customModels = CUSTOM_MODEL_SLOTS.map((slot) => {
            const backendModel = registryById.get(slot.backendId) || servedById.get(slot.backendId);
            return this.mergeBackendSlot(slot, backendModel, backendAvailable);
        });

        return [...BROWSER_MODELS, ...customModels];
    }

    async preloadImageOnlyModels() {
        const hasSelectableBackendModels = this.availableModels.some(
            (model) => model.runtime === 'backend' && model.selectable !== false
        );

        if (!hasSelectableBackendModels) return;

        try {
            this.ui.updateModelStatus('Listo - preparando modelos');
            const status = await this.apiClient.preloadModels();
            if (status.running) {
                this.pollPreloadStatus();
            } else {
                this.ui.updateModelStatus('Listo');
            }
        } catch (error) {
            console.info('No se pudo iniciar la precarga de modelos backend.', error);
            this.ui.updateModelStatus('Listo');
        }
    }

    async pollPreloadStatus() {
        try {
            const status = await this.apiClient.getPreloadStatus();
            const completedModels = status.models?.filter((model) => model.status === 'ready' || model.status === 'failed').length || 0;
            const totalModels = status.models?.length || 0;

            if (status.running) {
                this.ui.updateModelStatus(`Preparando modelos ${completedModels}/${totalModels}`);
                window.setTimeout(() => this.pollPreloadStatus(), 4000);
                return;
            }

            this.ui.updateModelStatus('Listo');
        } catch (error) {
            console.info('No se pudo consultar la precarga de modelos backend.', error);
            this.ui.updateModelStatus('Listo');
        }
    }

    mergeBackendSlot(slot, backendModel, backendAvailable) {
        if (!backendModel) {
            return {
                ...slot,
                selectable: false,
                unavailableReason: backendAvailable
                    ? 'Este modelo aun no esta registrado como servible en el backend.'
                    : 'Inicia el backend local para conectar este modelo.',
                metadata: null
            };
        }

        return {
            ...slot,
            ...this.normalizeBackendModel(backendModel, backendAvailable, slot)
        };
    }

    normalizeBackendModel(model, backendAvailable, fallback = {}) {
        const status = model.status || fallback.status || 'unavailable';
        const selectable = this.isBackendModelSelectable(model, backendAvailable);
        const task = TASK_LABELS[model.task] || fallback.task || model.task || 'Inferencia backend';

        return {
            id: fallback.id || `backend:${model.id}`,
            backendId: model.id,
            name: model.name || fallback.name || model.id,
            usageLabel: fallback.usageLabel || this.getUsageLabel(model.task),
            runtime: 'backend',
            modelType: model.task || fallback.modelType || 'backend',
            supportsWebcam: false,
            task,
            algorithm: fallback.algorithm || model.runtime || 'backend',
            efficiency: this.normalizeEfficiency(model.efficiency || fallback.efficiency || 'Medio'),
            modeLabel: fallback.modeLabel || (model.supportsWebcam ? 'Imagen/camara' : 'Solo imagen'),
            status,
            statusLabel: MODEL_STATUS_LABELS[status] || status,
            selectable,
            unavailableReason: selectable ? '' : this.getBackendUnavailableReason(model, backendAvailable),
            description: fallback.description || model.description || 'Modelo registrado en el backend interno.',
            metadata: model
        };
    }

    isBackendModelSelectable(model, backendAvailable) {
        return Boolean(
            backendAvailable &&
            model?.serve !== false &&
            CONNECTABLE_BACKEND_STATUSES.has(model?.status || 'unavailable')
        );
    }

    getBackendUnavailableReason(model, backendAvailable) {
        if (!backendAvailable) return 'Inicia el backend local para usar modelos propios.';
        if (model?.serve === false) return 'Este modelo aun no esta expuesto para inferencia.';

        const status = model?.status || 'unavailable';
        if (status === 'planned') return 'Este espacio esta preparado, pero el modelo aun no ha sido entrenado.';
        if (status === 'partial' || status === 'training') return 'Este modelo sigue en entrenamiento o validacion.';
        if (status === 'failed') return 'El ultimo estado registrado del modelo tiene error.';

        return 'Este modelo aun no esta listo para usarse desde la interfaz.';
    }

    getUsageLabel(task) {
        if (task === 'object_detection') return 'Objetos';
        if (task === 'image_classification') return 'Imagenes';
        if (task === 'image_profile_classification') return 'Perfil';

        return 'Backend';
    }

    normalizeEfficiency(efficiency) {
        const normalized = (efficiency || 'Medio').toLowerCase();

        if (normalized === 'rapido') return 'Rapido';
        if (normalized.includes('lento')) return 'Lento';

        return 'Medio';
    }

    setupEventListeners() {
        if (this.eventListenersReady) return;

        const imageInput = requireElementById('image-input');
        const modelSelect = requireElementById('model-select');
        const webcamStart = requireElementById('webcam-start');
        const webcamStop = requireElementById('webcam-stop');
        const clearImageButton = requireElementById('clear-image-button');
        const activeModelOverlay = requireElementById('active-model-overlay');
        const activeModelTrigger = requireElementById('active-model-trigger');
        const activeModelMenu = requireElementById('active-model-menu');

        imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        modelSelect.addEventListener('click', (e) => this.handleModelOptionClick(e));
        modelSelect.addEventListener('keydown', (e) => this.handleModelOptionKeydown(e));
        webcamStart.addEventListener('click', () => this.startWebcam());
        webcamStop.addEventListener('click', () => this.stopWebcam());
        clearImageButton.addEventListener('click', () => this.clearUploadedImage());
        activeModelTrigger.addEventListener('click', (e) => {
            e.stopPropagation();
            this.ui.toggleActiveModelMenu();
        });
        activeModelMenu.addEventListener('click', (e) => this.handleActiveModelMenuClick(e));
        document.addEventListener('click', (e) => {
            if (!activeModelOverlay.contains(e.target)) {
                this.ui.closeActiveModelMenu();
            }
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.ui.closeActiveModelMenu();
            }
        });

        const uploadArea = requireSelector('.upload-area');
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        this.eventListenersReady = true;
    }

    handleActiveModelMenuClick(event) {
        const option = event.target.closest('.active-model-menu-item');
        if (!option) return;

        this.ui.closeActiveModelMenu();
        this.switchModel(option.dataset.modelId);
    }

    handleModelOptionClick(event) {
        const option = event.target.closest('.model-option');
        if (!option || option.disabled) return;

        this.switchModel(option.dataset.modelId);
    }

    handleModelOptionKeydown(event) {
        const option = event.target.closest('.model-option');
        if (!option || option.disabled) return;

        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            this.switchModel(option.dataset.modelId);
            return;
        }

        const directionByKey = {
            ArrowRight: 1,
            ArrowDown: 1,
            ArrowLeft: -1,
            ArrowUp: -1
        };
        const direction = directionByKey[event.key];
        if (!direction) return;

        const options = [...event.currentTarget.querySelectorAll('.model-option:not(:disabled)')];
        const currentIndex = options.indexOf(option);
        if (currentIndex === -1) return;

        const nextIndex = (currentIndex + direction + options.length) % options.length;
        const nextOption = options[nextIndex];

        event.preventDefault();
        nextOption.focus();
        this.switchModel(nextOption.dataset.modelId);
    }

    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!this.validateSelectedImage(file)) return;

        this.uploadedImageActive = true;
        this.currentImageFile = file;
        this.ui.updateSelectedFileName(file.name);
        this.stopWebcam();
        await this.classifyImage(file);
    }

    async classifyImage(imageFile) {
        const runId = this.nextRunId();

        try {
            this.ui.showLoading(this.getInferenceStatusMessage());
            this.ui.showActiveModel(this.currentModelInfo.name, true);
            this.ui.clearDetections();
            const startTime = performance.now();
            const imageElement = await this.imageProcessor.loadImage(imageFile);
            this.ui.showImage(imageElement);
            const predictions = await this.getCurrentPredictions(imageElement, imageFile);

            if (!this.isCurrentRun(runId)) return;

            const endTime = performance.now();
            const processingTime = this.currentModelInfo.runtime === 'backend' && predictions.processing_time_ms
                ? predictions.processing_time_ms.toFixed(2)
                : (endTime - startTime).toFixed(2);
            const normalizedPredictions = this.normalizePredictionsForDisplay(predictions, imageElement);

            this.ui.displayResults(normalizedPredictions, this.currentModelInfo.name);
            this.ui.updateStats(normalizedPredictions, processingTime);
            this.ui.renderDetections(normalizedPredictions, imageElement, this.modelManager.getCurrentModelType());
            this.ui.hideLoading();
            this.ui.showActiveModel(this.currentModelInfo.name, false);
        } catch (error) {
            if (!this.isCurrentRun(runId)) return;

            console.error('Error clasificando imagen:', error);
            this.ui.showActiveModel(this.currentModelInfo.name, false);
            this.ui.showError(error.message || 'Error al procesar la imagen.');
        }
    }

    async getCurrentPredictions(imageElement, imageFile) {
        if (this.currentModelInfo.runtime === 'backend') {
            return await this.apiClient.predict(imageFile, this.currentModelInfo.backendId);
        }

        const modelType = this.modelManager.getCurrentModelType();
        
        if (modelType === 'mobilenet') {
            const classifications = await this.currentModel.classify(imageElement, 5);
            return this.normalizeImageClassifications(classifications, imageElement);
        } else if (modelType === 'coco-ssd') {
            const detections = await this.currentModel.detect(imageElement);
            return this.normalizeObjectDetections(detections, imageElement);
        } else if (modelType === 'blazeface') {
            const faces = await this.currentModel.estimateFaces(imageElement, false);
            return this.normalizeFaces(faces, imageElement);
        } else if (modelType === 'handpose') {
            const hands = await this.currentModel.estimateHands(imageElement);
            return this.normalizeHands(hands, imageElement);
        } else if (modelType === 'body-pix') {
            const segmentation = await this.currentModel.segmentPerson(imageElement);
            return this.normalizeBodySegmentation(segmentation, imageElement);
        }

        throw new Error('No hay un modelo valido cargado para procesar la imagen.');
    }

    async switchModel(modelId) {
        const runId = this.nextRunId();
        const nextModel = this.availableModels.find((model) => model.id === modelId);
        this.ui.closeActiveModelMenu();

        if (!nextModel) {
            this.ui.showError('El modelo seleccionado no esta disponible.');
            return;
        }

        if (nextModel.selectable === false) {
            this.ui.showError(nextModel.unavailableReason || 'Este modelo aun no esta listo para usarse.');
            return;
        }

        try {
            this.ui.updateModelStatus('Cambiando modelo...');
            this.stopWebcam({ invalidate: false });
            await this.loadSelectedModel(nextModel);
            if (!this.isCurrentRun(runId)) return;
            this.ui.updateModelStatus('Listo');

            if (this.currentImageFile) {
                await this.classifyImage(this.currentImageFile);
            } else {
                this.ui.clearResults();
            }
        } catch (error) {
            if (!this.isCurrentRun(runId)) return;

            console.error('Error cambiando modelo:', error);
            this.ui.updateModelStatus('Error al cambiar modelo');
            this.ui.showError(this.getModelLoadErrorMessage(error));
        }
    }

    async loadSelectedModel(modelInfo) {
        this.currentModelInfo = modelInfo;
        this.ui.setSelectedModelOption(modelInfo.id);
        this.ui.populateActiveModelMenu(this.availableModels, modelInfo.id);

        if (modelInfo.runtime === 'backend') {
            this.currentModel = null;
            this.modelManager.clearCurrentModel();
            this.ui.updateModelName(modelInfo.name);
            this.ui.updateModelDetails(modelInfo);
            this.ui.updateWebcamAvailability(modelInfo);
            return;
        }

        await this.modelManager.loadModel(modelInfo.modelType);
        this.currentModel = this.modelManager.getCurrentModel();
        this.ui.updateModelName(modelInfo.name);
        this.ui.updateModelDetails(modelInfo);
        this.ui.updateWebcamAvailability(modelInfo);
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
            this.lastWebcamInferenceAt = 0;
            this.ui.updateWebcamControls(this.currentModelInfo, true);
            this.ui.showActiveModel(this.currentModelInfo.name, true);
            this.ui.hideEmptyPreview();
            this.ui.clearDetections();
            const video = requireElementById('webcam-video');
            const previewImage = requireElementById('preview-image');

            video.muted = true;
            video.autoplay = true;
            video.playsInline = true;

            this.webcamStream = await this.requestWebcamStream();

            if (!this.isCurrentRun(runId)) {
                this.webcamStream.getTracks().forEach(track => track.stop());
                this.webcamStream = null;
                return;
            }

            video.srcObject = this.webcamStream;
            video.style.display = 'block';
            previewImage.style.display = 'none';

            await this.waitForVideoMetadata(video);
            await video.play();

            if (!this.isCurrentRun(runId)) return;

            this.processWebcamFrames(video, runId);
        } catch (error) {
            if (!this.isCurrentRun(runId)) return;

            console.error('Error accediendo a webcam:', error);
            this.stopWebcamTracks();
            this.ui.showError(this.getWebcamErrorMessage(error));
            this.webcamActive = false;
            this.processingFrame = false;
            this.ui.updateWebcamControls(this.currentModelInfo, false);
        }
    }

    async requestWebcamStream() {
        try {
            return await navigator.mediaDevices.getUserMedia({
                audio: false,
                video: this.getWebcamVideoConstraints()
            });
        } catch (error) {
            if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
                throw error;
            }

            return await navigator.mediaDevices.getUserMedia({
                audio: false,
                video: true
            });
        }
    }

    getWebcamVideoConstraints() {
        if (this.modelManager.getCurrentModelType() === 'handpose') {
            return {
                width: { ideal: 480 },
                height: { ideal: 360 },
                facingMode: { ideal: 'environment' }
            };
        }

        return {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: { ideal: 'environment' }
        };
    }

    getWebcamInferenceInterval() {
        const modelType = this.modelManager.getCurrentModelType();

        if (modelType === 'handpose') return 220;
        if (modelType === 'body-pix') return 260;
        if (modelType === 'mobilenet') return 180;

        return 80;
    }

    waitForVideoMetadata(video) {
        if (video.readyState >= HTMLMediaElement.HAVE_METADATA && video.videoWidth && video.videoHeight) {
            return Promise.resolve();
        }

        return new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                cleanup();
                reject(new Error('La camara no entrego video a tiempo.'));
            }, 8000);

            const cleanup = () => {
                clearTimeout(timeoutId);
                video.removeEventListener('loadedmetadata', handleLoadedMetadata);
                video.removeEventListener('error', handleError);
            };

            const handleLoadedMetadata = () => {
                cleanup();
                resolve();
            };

            const handleError = () => {
                cleanup();
                reject(new Error('El navegador no pudo preparar el video de la camara.'));
            };

            video.addEventListener('loadedmetadata', handleLoadedMetadata, { once: true });
            video.addEventListener('error', handleError, { once: true });
        });
    }

    async processWebcamFrames(video, runId) {
        if (!this.webcamActive || this.processingFrame || !this.isCurrentRun(runId)) return;

        const now = performance.now();
        const minimumInterval = this.getWebcamInferenceInterval();

        if (now - this.lastWebcamInferenceAt < minimumInterval) {
            requestAnimationFrame(() => this.processWebcamFrames(video, runId));
            return;
        }

        this.lastWebcamInferenceAt = now;

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
            this.ui.showActiveModel(this.currentModelInfo.name, false);
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

        this.stopWebcamTracks();

        const video = requireElementById('webcam-video');
        const previewImage = requireElementById('preview-image');
        
        video.style.display = 'none';
        previewImage.style.display = this.currentImageFile ? 'block' : 'none';
        
        this.webcamActive = false;
        this.processingFrame = false;
        this.lastWebcamInferenceAt = 0;
        this.ui.clearDetections();
        if (this.currentImageFile) {
            this.ui.hideEmptyPreview();
            this.ui.showActiveModel(this.currentModelInfo.name, false);
        } else {
            this.ui.showEmptyPreview();
            this.ui.hideActiveModel();
        }
        this.ui.updateWebcamControls(this.currentModelInfo, false);
    }

    stopWebcamTracks() {
        if (!this.webcamStream) return;

        this.webcamStream.getTracks().forEach(track => track.stop());
        this.webcamStream = null;
    }

    handleDragOver(event) {
        event.preventDefault();
        if (this.uploadedImageActive) return;
        event.currentTarget.classList.add('drag-over');
    }

    handleDragLeave(event) {
        event.currentTarget.classList.remove('drag-over');
    }

    handleDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('drag-over');

        if (this.uploadedImageActive) return;
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const imageFile = files[0];
            if (!this.validateSelectedImage(imageFile)) return;

            this.uploadedImageActive = true;
            this.currentImageFile = imageFile;
            this.ui.updateSelectedFileName(imageFile.name);
            this.stopWebcam();
            this.classifyImage(imageFile);
        }
    }

    clearUploadedImage() {
        const imageInput = requireElementById('image-input');

        this.uploadedImageActive = false;
        this.currentImageFile = null;
        imageInput.value = '';
        this.nextRunId();
        this.ui.clearUploadedImageState();
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

    getInferenceStatusMessage() {
        if (this.currentModelInfo.runtime === 'backend') {
            return `Ejecutando ${this.currentModelInfo.name} desde backend...`;
        }

        return `Ejecutando ${this.currentModelInfo.name}...`;
    }

    normalizePredictionsForDisplay(predictionsPayload, sourceElement) {
        const predictions = this.normalizePredictions(predictionsPayload);

        if (this.currentModelInfo.runtime !== 'backend') {
            return predictions;
        }

        const backendImageSize = predictionsPayload?.image;
        const previewSize = this.getSourceSize(sourceElement);

        if (!backendImageSize?.width || !backendImageSize?.height || !previewSize.width || !previewSize.height) {
            return predictions;
        }

        const scaleX = previewSize.width / backendImageSize.width;
        const scaleY = previewSize.height / backendImageSize.height;

        return predictions
            .map((prediction) => {
                const scaledBbox = this.scaleBbox(prediction.bbox, scaleX, scaleY);

                return {
                    ...prediction,
                    bbox: scaledBbox ? this.clampValidBbox(scaledBbox, previewSize) : scaledBbox
                };
            })
            .filter((prediction) => !prediction.bbox || this.isValidBbox(prediction.bbox));
    }

    scaleBbox(bbox, scaleX, scaleY) {
        if (!this.isValidBbox(bbox)) return bbox;

        return [
            bbox[0] * scaleX,
            bbox[1] * scaleY,
            bbox[2] * scaleX,
            bbox[3] * scaleY
        ];
    }

    normalizeImageClassifications(classifications, sourceElement) {
        const sourceSize = this.getSourceSize(sourceElement);
        const imageBbox = this.fullImageBbox(sourceSize);

        return classifications.map((classification, index) => ({
            ...classification,
            type: 'image_classification',
            bbox: index === 0 ? imageBbox : null,
            details: index === 0
                ? 'La clasificacion aplica a la imagen completa; este modelo no localiza objetos individuales.'
                : undefined
        }));
    }

    normalizeObjectDetections(detections, sourceElement) {
        const sourceSize = this.getSourceSize(sourceElement);

        return detections
            .map((detection) => {
                const bbox = this.clampValidBbox(detection.bbox, sourceSize);

                return {
                    ...detection,
                    type: 'object_detection',
                    bbox
                };
            })
            .filter((detection) => this.isValidBbox(detection.bbox));
    }

    normalizeFaces(faces, sourceElement) {
        const sourceSize = this.getSourceSize(sourceElement);

        return faces.map((face, index) => {
            const topLeft = this.toPoint(face.topLeft);
            const bottomRight = this.toPoint(face.bottomRight);
            const confidence = Array.isArray(face.probability) ? face.probability[0] : face.probability;
            const rawBbox = [
                topLeft.x,
                topLeft.y,
                bottomRight.x - topLeft.x,
                bottomRight.y - topLeft.y
            ];

            return {
                label: `rostro ${index + 1}`,
                confidence: confidence ?? 0,
                type: 'face_detection',
                bbox: this.refineFaceBbox(face, rawBbox, sourceSize)
            };
        });
    }

    refineFaceBbox(face, rawBbox, sourceSize) {
        const landmarkBbox = this.bboxFromLandmarks(face.landmarks, rawBbox);
        const bbox = landmarkBbox || this.tightenBbox(rawBbox, {
            offsetX: 0.08,
            offsetY: 0.03,
            scaleX: 0.84,
            scaleY: 0.84
        });

        return this.clampBbox(bbox, sourceSize);
    }

    bboxFromLandmarks(landmarks, rawBbox) {
        if (!Array.isArray(landmarks) || landmarks.length < 2) return null;

        const points = landmarks.map((landmark) => this.toPoint(landmark));
        const xValues = points.map((point) => point.x);
        const yValues = points.map((point) => point.y);
        const minX = Math.min(...xValues);
        const maxX = Math.max(...xValues);
        const minY = Math.min(...yValues);
        const maxY = Math.max(...yValues);
        const landmarkWidth = Math.max(maxX - minX, rawBbox[2] * 0.45);
        const landmarkHeight = Math.max(maxY - minY, rawBbox[3] * 0.35);
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const width = Math.min(rawBbox[2] * 0.9, landmarkWidth * 1.55);
        const height = Math.min(rawBbox[3] * 0.88, landmarkHeight * 2.15);

        return [
            centerX - width / 2,
            centerY - height * 0.58,
            width,
            height
        ];
    }

    tightenBbox(bbox, options) {
        const [x, y, width, height] = bbox;
        const nextWidth = width * options.scaleX;
        const nextHeight = height * options.scaleY;

        return [
            x + width * options.offsetX,
            y + height * options.offsetY,
            nextWidth,
            nextHeight
        ];
    }

    clampBbox(bbox, sourceSize) {
        const [x, y, width, height] = bbox;
        const maxWidth = sourceSize.width || x + width;
        const maxHeight = sourceSize.height || y + height;
        const nextX = Math.max(0, Math.min(x, maxWidth));
        const nextY = Math.max(0, Math.min(y, maxHeight));
        const nextWidth = Math.max(0, Math.min(width, maxWidth - nextX));
        const nextHeight = Math.max(0, Math.min(height, maxHeight - nextY));

        return [nextX, nextY, nextWidth, nextHeight];
    }

    clampValidBbox(bbox, sourceSize) {
        if (!this.isValidBbox(bbox)) return null;

        const clampedBbox = this.clampBbox(bbox, sourceSize);
        return this.isValidBbox(clampedBbox) ? clampedBbox : null;
    }

    normalizeHands(hands, sourceElement) {
        const sourceSize = this.getSourceSize(sourceElement);

        return hands.map((hand, index) => ({
            label: `mano ${index + 1}`,
            confidence: hand.handInViewConfidence ?? 0,
            type: 'hand_landmarks',
            bbox: this.refineHandBbox(hand, sourceSize),
            details: `${hand.landmarks?.length || 0} puntos clave`
        }));
    }

    refineHandBbox(hand, sourceSize) {
        const boxBbox = this.bboxFromBox(hand.boundingBox);
        const landmarkBbox = this.bboxFromHandLandmarks(hand.landmarks);
        const bbox = this.isValidBbox(landmarkBbox)
            ? landmarkBbox
            : this.getReasonableFallbackBbox(boxBbox, sourceSize);

        if (!this.isValidBbox(bbox)) return null;

        return this.clampBbox(this.expandBbox(bbox, 0.06, 0.08), sourceSize);
    }

    bboxFromHandLandmarks(landmarks) {
        if (!Array.isArray(landmarks) || landmarks.length === 0) return null;

        const points = landmarks.map((landmark) => this.toPoint(landmark));
        const xValues = points.map((point) => point.x);
        const yValues = points.map((point) => point.y);
        const minX = Math.min(...xValues);
        const maxX = Math.max(...xValues);
        const minY = Math.min(...yValues);
        const maxY = Math.max(...yValues);

        return [minX, minY, maxX - minX, maxY - minY];
    }

    expandBbox(bbox, horizontalRatio, verticalRatio) {
        const [x, y, width, height] = bbox;
        const extraX = width * horizontalRatio;
        const extraY = height * verticalRatio;

        return [
            x - extraX,
            y - extraY,
            width + extraX * 2,
            height + extraY * 2
        ];
    }

    getReasonableFallbackBbox(bbox, sourceSize) {
        if (!this.isValidBbox(bbox)) return null;
        if (!sourceSize.width || !sourceSize.height) return bbox;

        const widthRatio = bbox[2] / sourceSize.width;
        const heightRatio = bbox[3] / sourceSize.height;
        const areaRatio = (bbox[2] * bbox[3]) / (sourceSize.width * sourceSize.height);

        if (widthRatio > 0.45 || heightRatio > 0.55 || areaRatio > 0.22) {
            return null;
        }

        return bbox;
    }

    fullImageBbox(sourceSize) {
        if (!sourceSize.width || !sourceSize.height) return null;

        return [0, 0, sourceSize.width, sourceSize.height];
    }

    normalizeBodySegmentation(segmentation, sourceElement) {
        const totalPixels = segmentation.data?.length || 0;
        const personPixels = totalPixels
            ? segmentation.data.reduce((total, value) => total + (value ? 1 : 0), 0)
            : 0;
        const personRatio = totalPixels ? personPixels / totalPixels : 0;
        const sourceSize = this.getSourceSize(sourceElement);
        const segmentationBbox = this.bboxFromSegmentation(segmentation, sourceSize);

        if (!personPixels) return [];

        return [
            {
                label: 'persona segmentada',
                confidence: Math.min(0.5 + personRatio, 0.99),
                type: 'person_segmentation',
                bbox: segmentationBbox,
                details: `${(personRatio * 100).toFixed(1)}% de pixeles pertenecen a persona`
            }
        ];
    }

    bboxFromSegmentation(segmentation, sourceSize) {
        if (!segmentation.data?.length) return null;

        const maskWidth = segmentation.width || sourceSize.width;
        const maskHeight = segmentation.height || sourceSize.height;
        if (!maskWidth || !maskHeight) return null;

        let minX = maskWidth;
        let minY = maskHeight;
        let maxX = -1;
        let maxY = -1;

        segmentation.data.forEach((value, index) => {
            if (!value) return;

            const x = index % maskWidth;
            const y = Math.floor(index / maskWidth);
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        });

        if (maxX < minX || maxY < minY) return null;

        const scaleX = sourceSize.width && maskWidth ? sourceSize.width / maskWidth : 1;
        const scaleY = sourceSize.height && maskHeight ? sourceSize.height / maskHeight : 1;
        const bbox = [
            minX * scaleX,
            minY * scaleY,
            (maxX - minX + 1) * scaleX,
            (maxY - minY + 1) * scaleY
        ];

        return this.clampBbox(this.expandBbox(bbox, 0.02, 0.03), sourceSize);
    }

    bboxFromBox(box) {
        if (!box) return null;

        const topLeft = this.toPoint(box.topLeft);
        const bottomRight = this.toPoint(box.bottomRight);

        return [
            topLeft.x,
            topLeft.y,
            bottomRight.x - topLeft.x,
            bottomRight.y - topLeft.y
        ];
    }

    isValidBbox(bbox) {
        return Array.isArray(bbox)
            && bbox.length === 4
            && bbox.every(Number.isFinite)
            && bbox[2] > 0
            && bbox[3] > 0;
    }

    toPoint(point) {
        if (Array.isArray(point)) {
            return { x: point[0], y: point[1] };
        }

        return { x: point?.x || 0, y: point?.y || 0 };
    }

    getSourceSize(sourceElement) {
        if (sourceElement instanceof HTMLVideoElement) {
            return { width: sourceElement.videoWidth, height: sourceElement.videoHeight };
        }

        if (sourceElement instanceof HTMLCanvasElement) {
            return { width: sourceElement.width, height: sourceElement.height };
        }

        if (sourceElement instanceof HTMLImageElement) {
            return { width: sourceElement.naturalWidth, height: sourceElement.naturalHeight };
        }

        return { width: 0, height: 0 };
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

        if (error.name === 'OverconstrainedError') {
            return 'La camara no soporta la configuracion solicitada.';
        }

        if (error.name === 'SecurityError') {
            return 'La webcam requiere abrir la app desde localhost o HTTPS.';
        }

        if (error.name === 'AbortError') {
            return 'El navegador interrumpio el acceso a la camara. Intenta de nuevo.';
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
