import { requireElementById, requireSelector } from './dom.js';

export class ClassifierUI {
    constructor() {
        this.resultsContainer = requireElementById('results-container');
        this.loadingSpinner = requireElementById('loading-spinner');
        this.previewImage = requireElementById('preview-image');
        this.modelStatus = requireElementById('model-status');
        this.confidenceValue = requireElementById('confidence-value');
        this.timeValue = requireElementById('time-value');
        this.imageContainer = requireSelector('.image-container');
        this.canvasContainer = requireElementById('canvas-container');
        this.detectionCanvas = requireElementById('detection-canvas');
    }

    showLoading() {
        this.loadingSpinner.style.display = 'inline-block';
        this.setStatusMessage('Procesando...');
    }

    hideLoading() {
        this.loadingSpinner.style.display = 'none';
    }

    displayResults(predictions, modelType) {
        this.resultsContainer.replaceChildren();

        if (!predictions || predictions.length === 0) {
            this.setStatusMessage('Sin resultados');
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
            fragment.append(item);
        });

        this.resultsContainer.append(fragment);
    }

    updateStats(predictions, processingTime) {
        if (predictions && predictions.length > 0) {
            const confidence = this.getPredictionConfidence(predictions[0]);
            this.confidenceValue.textContent = (confidence * 100).toFixed(2) + '%';
        }

        this.timeValue.textContent = processingTime + 'ms';
    }

    showImage(imageElement) {
        if (imageElement instanceof HTMLImageElement) {
            this.previewImage.src = imageElement.src;
            this.previewImage.style.display = 'block';
        } else if (imageElement instanceof HTMLCanvasElement) {
            this.previewImage.src = imageElement.toDataURL();
            this.previewImage.style.display = 'block';
        }
    }

    renderDetections(predictions, sourceElement, modelType) {
        if (modelType !== 'coco-ssd' || !Array.isArray(predictions) || predictions.length === 0) {
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
            const label = `${prediction.class} ${(prediction.score * 100).toFixed(1)}%`;
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

        models.forEach((model) => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            option.selected = model.id === selectedModelId;
            modelSelect.append(option);
        });
    }

    updateModelName(name) {
        const modelName = requireElementById('model-name');
        modelName.textContent = name;
    }

    updateModelStatus(status) {
        this.modelStatus.textContent = status;
    }

    clearResults() {
        this.setStatusMessage('Carga una imagen para comenzar');
        this.confidenceValue.textContent = '-';
        this.timeValue.textContent = '-';
        this.previewImage.src = '';
        this.clearDetections();
    }

    showError(message) {
        const errorMessage = document.createElement('p');
        errorMessage.className = 'error-message';
        errorMessage.textContent = `Error: ${message}`;
        this.resultsContainer.replaceChildren(errorMessage);
        this.hideLoading();
    }

    setStatusMessage(message) {
        const status = document.createElement('p');
        status.className = 'empty-state';
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
