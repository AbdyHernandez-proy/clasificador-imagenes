import { requireElementById, requireSelector } from './dom.js';

export class ClassifierUI {
    constructor() {
        this.resultsContainer = requireElementById('results-container');
        this.loadingSpinner = requireElementById('loading-spinner');
        this.previewEmptyState = requireElementById('preview-empty-state');
        this.previewImage = requireElementById('preview-image');
        this.modelStatus = requireElementById('model-status');
        this.activeModelOverlay = requireElementById('active-model-overlay');
        this.activeModelTrigger = requireElementById('active-model-trigger');
        this.activeModelMenu = requireElementById('active-model-menu');
        this.activeModelStatus = requireElementById('active-model-status');
        this.activeModelName = requireElementById('active-model-name');
        this.webcamNote = requireElementById('webcam-note');
        this.confidenceValue = requireElementById('confidence-value');
        this.timeValue = requireElementById('time-value');
        this.imageContainer = requireSelector('.image-container');
        this.uploadArea = requireSelector('.upload-area');
        this.canvasContainer = requireElementById('canvas-container');
        this.detectionCanvas = requireElementById('detection-canvas');
    }

    showLoading(message = 'Procesando...') {
        this.loadingSpinner.style.display = 'inline-block';
        this.setStatusMessage(message, 'loading-state');
    }

    hideLoading() {
        this.loadingSpinner.style.display = 'none';
    }

    showActiveModel(modelName, isRunning = false) {
        this.activeModelStatus.textContent = isRunning ? 'Ejecutando modelo' : 'Modelo activo';
        this.activeModelName.textContent = modelName || 'Modelo seleccionado';
        this.activeModelOverlay.hidden = false;
        this.activeModelOverlay.classList.toggle('is-running', isRunning);
    }

    hideActiveModel() {
        this.activeModelOverlay.hidden = true;
        this.activeModelOverlay.classList.remove('is-running');
        this.closeActiveModelMenu();
    }

    toggleActiveModelMenu() {
        this.setActiveModelMenuOpen(this.activeModelMenu.hidden);
    }

    closeActiveModelMenu() {
        this.setActiveModelMenuOpen(false);
    }

    setActiveModelMenuOpen(isOpen) {
        this.activeModelMenu.hidden = !isOpen;
        this.activeModelTrigger.setAttribute('aria-expanded', String(isOpen));
    }

    showEmptyPreview() {
        this.previewEmptyState.hidden = false;
    }

    hideEmptyPreview() {
        this.previewEmptyState.hidden = true;
    }

    displayResults(predictions, modelType) {
        this.resultsContainer.replaceChildren();

        if (!predictions || predictions.length === 0) {
            this.setStatusMessage(`No se detectaron objetos con ${modelType}.`, 'no-results-state');
            return;
        }

        const fragment = document.createDocumentFragment();

        predictions.forEach((prediction, index) => {
            const confidence = this.getPredictionConfidence(prediction);
            const item = document.createElement('div');
            item.className = 'result-item';

            const header = document.createElement('div');
            header.className = 'result-header';

            const rank = document.createElement('span');
            rank.className = 'result-rank';
            rank.textContent = `#${index + 1}`;

            const label = document.createElement('span');
            label.className = 'result-label';
            label.textContent = this.getPredictionLabel(prediction);

            const bar = document.createElement('div');
            bar.className = 'result-bar';

            const progress = document.createElement('div');
            progress.className = 'result-progress';
            progress.style.width = `${confidence * 100}%`;

            const probability = document.createElement('span');
            probability.className = 'result-probability';
            probability.textContent = `${(confidence * 100).toFixed(2)}%`;

            header.append(rank, label);
            bar.append(progress);
            item.append(header, bar, probability);

            if (prediction.details) {
                const details = document.createElement('p');
                details.className = 'result-details';
                details.textContent = prediction.details;
                item.append(details);
            }

            fragment.append(item);
        });

        this.resultsContainer.append(fragment);
    }

    updateStats(predictions, processingTime) {
        if (predictions && predictions.length > 0) {
            const confidence = this.getPredictionConfidence(predictions[0]);
            this.confidenceValue.textContent = (confidence * 100).toFixed(2) + '%';
        } else {
            this.confidenceValue.textContent = '-';
        }

        this.timeValue.textContent = processingTime + 'ms';
    }

    showImage(imageElement) {
        this.hideEmptyPreview();

        if (imageElement instanceof HTMLImageElement) {
            this.previewImage.src = imageElement.src;
            this.previewImage.style.display = 'block';
        } else if (imageElement instanceof HTMLCanvasElement) {
            this.previewImage.src = imageElement.toDataURL();
            this.previewImage.style.display = 'block';
        }
    }

    renderDetections(predictions, sourceElement) {
        if (!Array.isArray(predictions) || predictions.length === 0 || !predictions.some((prediction) => prediction.bbox)) {
            this.clearDetections();
            return;
        }

        const sourceSize = this.getSourceSize(sourceElement);
        if (!sourceSize.width || !sourceSize.height) {
            this.clearDetections();
            return;
        }

        const containerWidth = this.imageContainer.clientWidth;
        const containerHeight = this.imageContainer.clientHeight;
        const scale = Math.min(containerWidth / sourceSize.width, containerHeight / sourceSize.height);
        const drawWidth = sourceSize.width * scale;
        const drawHeight = sourceSize.height * scale;
        const offsetX = (containerWidth - drawWidth) / 2;
        const offsetY = (containerHeight - drawHeight) / 2;

        this.detectionCanvas.width = containerWidth;
        this.detectionCanvas.height = containerHeight;
        this.canvasContainer.style.display = 'block';

        const ctx = this.detectionCanvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, containerWidth, containerHeight);
        ctx.lineWidth = 2;
        ctx.font = '12px system-ui, sans-serif';

        predictions.forEach((prediction) => {
            if (!prediction.bbox) return;

            const [x, y, width, height] = prediction.bbox;
            const boxX = offsetX + x * scale;
            const boxY = offsetY + y * scale;
            const boxWidth = width * scale;
            const boxHeight = height * scale;
            const label = `${this.getPredictionLabel(prediction)} ${(this.getPredictionConfidence(prediction) * 100).toFixed(1)}%`;
            const labelWidth = ctx.measureText(label).width + 8;
            const labelY = Math.max(0, boxY - 20);

            ctx.strokeStyle = '#10b981';
            ctx.fillStyle = 'rgba(16, 185, 129, 0.18)';
            ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);
            ctx.fillRect(boxX, boxY, boxWidth, boxHeight);

            ctx.fillStyle = '#10b981';
            ctx.fillRect(boxX, labelY, labelWidth, 18);
            ctx.fillStyle = '#0f172a';
            ctx.fillText(label, boxX + 4, labelY + 13);
        });
    }

    clearDetections() {
        const ctx = this.detectionCanvas.getContext('2d');
        if (ctx) {
            ctx.clearRect(0, 0, this.detectionCanvas.width, this.detectionCanvas.height);
        }
        this.canvasContainer.style.display = 'none';
    }

    populateModelSelector(models, selectedModelId) {
        const modelSelect = requireElementById('model-select');
        modelSelect.replaceChildren();

        const realtimeModels = models.filter((model) => model.runtime === 'browser');
        const imageOnlyModels = models.filter((model) => model.runtime === 'backend');

        this.appendModelGroup(
            modelSelect,
            'Modelos para imagenes y camara',
            'Clasificacion con imagenes cargadas y uso en tiempo real.',
            realtimeModels,
            selectedModelId
        );
        this.appendModelGroup(
            modelSelect,
            'Modelos solo para imagenes',
            'Modelos propios del backend para analizar imagenes cargadas.',
            imageOnlyModels,
            selectedModelId
        );

        this.setSelectedModelOption(selectedModelId);
    }

    appendModelGroup(modelSelect, title, description, models, selectedModelId) {
        if (!models.length) return;

        const group = document.createElement('section');
        group.className = 'model-group';

        const heading = document.createElement('div');
        heading.className = 'model-group-heading';

        const titleElement = document.createElement('span');
        titleElement.className = 'model-group-title';
        titleElement.textContent = title;

        const descriptionElement = document.createElement('span');
        descriptionElement.className = 'model-group-description';
        descriptionElement.textContent = description;

        const grid = document.createElement('div');
        grid.className = 'model-group-grid';

        heading.append(titleElement, descriptionElement);
        group.append(heading, grid);

        models.forEach((model) => {
            const option = document.createElement('button');
            const isSelected = model.id === selectedModelId;
            const isEnabled = model.selectable !== false;

            option.type = 'button';
            option.className = 'model-option';
            option.classList.add(model.runtime === 'backend' ? 'model-option-backend' : 'model-option-browser');
            option.classList.toggle('is-disabled', !isEnabled);
            option.dataset.modelId = model.id;
            option.disabled = !isEnabled;
            option.title = isEnabled ? model.description || model.name : model.unavailableReason || 'Modelo no disponible';
            option.setAttribute('role', 'radio');
            option.setAttribute('aria-checked', String(isSelected));
            option.setAttribute('aria-disabled', String(!isEnabled));
            option.tabIndex = isEnabled && isSelected ? 0 : -1;

            const header = document.createElement('span');
            header.className = 'model-option-header';

            const name = document.createElement('span');
            name.className = 'model-option-name';
            name.textContent = model.name;

            const usage = document.createElement('span');
            usage.className = 'model-option-usage';
            usage.textContent = model.usageLabel || model.task || 'Modelo';

            const efficiency = document.createElement('span');
            efficiency.className = `model-option-efficiency efficiency-${this.getEfficiencyClass(model.efficiency)}`;
            efficiency.textContent = model.efficiency || 'Medio';

            const status = document.createElement('span');
            status.className = `model-option-status status-${this.getStatusClass(model.status)}`;
            status.textContent = model.statusLabel || 'Listo';

            const mode = document.createElement('span');
            mode.className = `model-option-mode mode-${model.runtime === 'backend' ? 'image-only' : 'realtime'}`;
            mode.textContent = model.modeLabel || (model.supportsWebcam ? 'Imagen/camara' : 'Solo imagen');

            const metadata = document.createElement('span');
            metadata.className = 'model-option-metadata';
            metadata.append(mode, efficiency, status);

            const description = document.createElement('span');
            description.className = 'model-option-description';
            description.textContent = isEnabled
                ? model.description || model.task || 'Modelo disponible'
                : model.unavailableReason || model.description || model.task || 'Modelo no disponible';

            header.append(name, usage);
            option.append(header, metadata, description);
            grid.append(option);
        });

        modelSelect.append(group);
    }

    setSelectedModelOption(modelId) {
        const modelSelect = requireElementById('model-select');
        const options = modelSelect.querySelectorAll('.model-option');

        options.forEach((option) => {
            const isSelected = option.dataset.modelId === modelId;
            option.classList.toggle('is-selected', isSelected);
            option.setAttribute('aria-checked', String(isSelected));
            option.tabIndex = !option.disabled && isSelected ? 0 : -1;
        });
    }

    populateActiveModelMenu(models, selectedModelId) {
        const options = models.filter((model) => model.id !== selectedModelId && model.selectable !== false);
        this.activeModelMenu.replaceChildren();

        if (options.length === 0) {
            const empty = document.createElement('span');
            empty.className = 'active-model-menu-empty';
            empty.textContent = 'No hay otros modelos disponibles';
            this.activeModelMenu.append(empty);
            return;
        }

        options.forEach((model) => {
            const option = document.createElement('button');
            option.type = 'button';
            option.className = 'active-model-menu-item';
            option.dataset.modelId = model.id;
            option.setAttribute('role', 'option');

            const name = document.createElement('strong');
            name.textContent = model.name;

            const meta = document.createElement('span');
            meta.textContent = `${model.usageLabel || model.task || 'Modelo'} · ${model.efficiency || 'Medio'}`;

            option.append(name, meta);
            this.activeModelMenu.append(option);
        });
    }

    updateModelName(name) {
        const modelName = requireElementById('model-name');
        modelName.textContent = name;
    }

    updateModelDetails(model) {
        requireElementById('model-task').textContent = model.task || 'Modelo';
        requireElementById('model-algorithm').textContent = model.algorithm || model.runtime || 'Algoritmo no especificado';
        requireElementById('model-status-detail').textContent = `Estado: ${model.statusLabel || 'Listo'}`;
        requireElementById('model-status-detail').className = `model-status-detail status-${this.getStatusClass(model.status)}`;
        requireElementById('model-efficiency').textContent = `Eficiencia: ${model.efficiency || 'Medio'}`;
        requireElementById('model-efficiency').className = `model-efficiency efficiency-${this.getEfficiencyClass(model.efficiency)}`;

        const description = requireElementById('model-description');
        const descriptionText = document.createElement('span');
        descriptionText.className = 'model-description-text';
        descriptionText.textContent = model.description || 'Sin descripcion disponible.';
        description.replaceChildren(descriptionText);

        if (model.runtime === 'backend') {
            const classes = this.getDetectableClasses(model);
            if (classes.length > 0) {
                description.append(this.createDetectableClassesBlock(classes));
            }
        }
    }

    getDetectableClasses(model) {
        const labels = model.metadata?.labels || model.labels || [];
        if (!Array.isArray(labels)) return [];

        return labels
            .map((label) => String(label).trim())
            .filter(Boolean);
    }

    createDetectableClassesBlock(classes) {
        const wrapper = document.createElement('span');
        wrapper.className = 'model-classes';

        const label = document.createElement('span');
        label.className = 'model-classes-title';
        label.textContent = 'Clases detectables';

        const list = document.createElement('span');
        list.className = 'model-classes-list';

        classes.forEach((className) => {
            const chip = document.createElement('span');
            chip.className = 'model-class-chip';
            chip.textContent = className;
            list.append(chip);
        });

        wrapper.append(label, list);
        return wrapper;
    }

    getEfficiencyClass(efficiency) {
        const normalized = (efficiency || 'medio').toLowerCase();

        if (normalized === 'rapido') return 'fast';
        if (normalized === 'lento') return 'slow';

        return 'medium';
    }

    getStatusClass(status) {
        const normalized = (status || 'ready').toLowerCase();

        if (normalized === 'ready' || normalized === 'trained') return 'ready';
        if (normalized === 'training' || normalized === 'partial') return 'training';
        if (normalized === 'planned') return 'planned';
        if (normalized === 'failed') return 'failed';

        return 'unavailable';
    }

    updateWebcamAvailability(model) {
        this.updateWebcamControls(model, false);
    }

    updateWebcamControls(model, isActive = false) {
        const webcamStart = requireElementById('webcam-start');
        const webcamStop = requireElementById('webcam-stop');
        const supportsWebcam = Boolean(model.supportsWebcam);

        webcamStart.disabled = !supportsWebcam || isActive;
        webcamStop.disabled = !supportsWebcam || !isActive;
        webcamStart.title = supportsWebcam
            ? 'Iniciar captura de camara'
            : 'La webcam solo esta disponible con modelos del navegador';
        webcamStop.title = supportsWebcam
            ? 'Finalizar captura de camara'
            : 'La webcam solo esta disponible con modelos del navegador';
        this.webcamNote.hidden = supportsWebcam;
        this.webcamNote.textContent = supportsWebcam
            ? ''
            : 'Camara desactivada para modelos solo imagen.';
    }

    updateSelectedFileName(fileName) {
        const imageInput = requireElementById('image-input');

        requireElementById('selected-file-name').textContent = fileName || 'Ningun archivo seleccionado';
        this.uploadArea.classList.toggle('has-file', Boolean(fileName));
        requireElementById('clear-image-button').hidden = !fileName;
        imageInput.disabled = Boolean(fileName);
    }

    updateModelStatus(status) {
        this.modelStatus.textContent = status;
    }

    clearResults() {
        this.setStatusMessage('Carga una imagen para comenzar');
        this.confidenceValue.textContent = '-';
        this.timeValue.textContent = '-';
        this.previewImage.src = '';
        this.previewImage.style.display = 'none';
        this.clearDetections();
        this.hideActiveModel();
        this.showEmptyPreview();
    }

    clearUploadedImageState() {
        this.hideLoading();
        this.updateSelectedFileName('');
        this.clearResults();
    }

    showError(message) {
        const errorMessage = document.createElement('p');
        errorMessage.className = 'error-message';
        errorMessage.textContent = `Error: ${message}`;
        this.resultsContainer.replaceChildren(errorMessage);
        this.hideLoading();
    }

    setStatusMessage(message, className = 'empty-state') {
        const status = document.createElement('p');
        status.className = className;
        status.textContent = message;
        this.resultsContainer.replaceChildren(status);
    }

    getPredictionLabel(prediction) {
        return prediction.className || prediction.class || prediction.label || 'Sin etiqueta';
    }

    getPredictionConfidence(prediction) {
        return prediction.probability ?? prediction.score ?? prediction.confidence ?? 0;
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
}
