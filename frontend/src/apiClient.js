const DEFAULT_API_BASE_URL = 'http://localhost:8000';

export class ClassifierApiClient {
    constructor(baseUrl = DEFAULT_API_BASE_URL) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
    }

    async listModels() {
        const response = await this.fetchWithTimeout(`${this.baseUrl}/models`, { timeout: 2500 });
        if (!response.ok) {
            throw new Error('No se pudieron cargar los modelos del backend.');
        }

        return await response.json();
    }

    async listModelRegistry() {
        const response = await this.fetchWithTimeout(`${this.baseUrl}/models/registry`, { timeout: 2500 });
        if (!response.ok) {
            throw new Error('No se pudo cargar el registro completo de modelos.');
        }

        return await response.json();
    }

    async preloadModels() {
        const response = await this.fetchWithTimeout(`${this.baseUrl}/models/preload`, {
            method: 'POST',
            timeout: 2500
        });
        if (!response.ok) {
            throw new Error('No se pudo iniciar la precarga de modelos.');
        }

        return await response.json();
    }

    async getPreloadStatus() {
        const response = await this.fetchWithTimeout(`${this.baseUrl}/models/preload`, { timeout: 2500 });
        if (!response.ok) {
            throw new Error('No se pudo consultar la precarga de modelos.');
        }

        return await response.json();
    }

    async predict(imageFile, modelId) {
        const formData = new FormData();
        formData.append('image', imageFile);
        if (modelId) {
            formData.append('model_id', modelId);
        }

        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            body: formData
        });

        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || 'Error al clasificar la imagen desde el backend.');
        }

        return payload;
    }

    async fetchWithTimeout(url, options = {}) {
        const { timeout = 2500, ...fetchOptions } = options;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
            return await fetch(url, { ...fetchOptions, signal: controller.signal });
        } finally {
            clearTimeout(timeoutId);
        }
    }
}
