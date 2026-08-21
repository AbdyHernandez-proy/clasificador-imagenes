# Roadmap

## Completado

1. Conectar frontend con backend.
   - El frontend consume `GET /models` cuando la API esta disponible.
   - El selector combina modelos del navegador y modelos del backend.
   - Las imagenes se envian a `POST /predict` cuando se selecciona un modelo backend.
   - Si el backend no esta disponible, la interfaz mantiene los modelos del navegador como fallback.

2. Implementar inferencia backend real.
   - El backend valida imagenes recibidas por `POST /predict`.
   - `visual-profile-v1` ejecuta inferencia interna con Pillow.
   - La respuesta incluye `model_id`, `model_name`, metadatos de imagen, predicciones y tiempo de procesamiento.

## Pendiente

3. Crear estructura formal de datasets.
4. Crear scripts de entrenamiento.
5. Exportar modelos propios a `ml/models`.
6. Registrar modelos en `ml/registry.json`.
7. Mejorar selector de modelos con metadatos visibles para el usuario.
8. Agregar pruebas automatizadas.
9. Definir despliegue backend.
