export const ALLOWED_IMAGE_TYPES = [
    'image/jpeg',
    'image/png',
    'image/webp',
    'image/gif',
    'image/bmp'
];

export const ALLOWED_IMAGE_EXTENSIONS = ['JPG', 'JPEG', 'PNG', 'WEBP', 'GIF', 'BMP'];

export class ImageProcessor {
    constructor() {
        this.maxWidth = 640;
        this.maxHeight = 480;
        this.maxFileSize = 10 * 1024 * 1024;
    }

    validateImageFile(file) {
        if (!file) {
            throw new Error('No se selecciono ningun archivo.');
        }

        if (!file.type || !ALLOWED_IMAGE_TYPES.includes(file.type)) {
            throw new Error(`Formato no permitido. Usa: ${ALLOWED_IMAGE_EXTENSIONS.join(', ')}.`);
        }

        if (file.size > this.maxFileSize) {
            throw new Error('La imagen supera el tamano maximo permitido de 10 MB.');
        }
    }

    async loadImage(file) {
        this.validateImageFile(file);

        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    try {
                        const resized = this.resizeImage(img);
                        resolve(resized);
                    } catch (error) {
                        reject(error);
                    }
                };
                img.onerror = () => reject(new Error('El archivo seleccionado no es una imagen valida o esta danado.'));
                img.src = e.target.result;
            };
            reader.onerror = () => reject(new Error('No se pudo leer el archivo seleccionado.'));
            reader.readAsDataURL(file);
        });
    }

    resizeImage(img) {
        const canvas = document.createElement('canvas');
        let width = img.width;
        let height = img.height;

        if (width > height) {
            if (width > this.maxWidth) {
                height = Math.round((height * this.maxWidth) / width);
                width = this.maxWidth;
            }
        } else {
            if (height > this.maxHeight) {
                width = Math.round((width * this.maxHeight) / height);
                height = this.maxHeight;
            }
        }

        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
            throw new Error('No se pudo preparar la imagen para procesamiento.');
        }

        ctx.drawImage(img, 0, 0, width, height);

        return canvas;
    }
}
