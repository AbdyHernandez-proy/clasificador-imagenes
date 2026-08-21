# Auditoria y pasos a seguir

## Estado actual

El proyecto fue reestructurado para separar responsabilidades:

- `frontend/`: interfaz web.
- `backend/`: API interna sin usuarios ni autenticacion.
- `ml/`: datasets, modelos, entrenamiento, inferencia y registro de modelos.
- `docs/`: documentacion tecnica.

No se detectaron incidentes activos en el frontend despues del ultimo build exitoso. La nueva estructura deja componentes pendientes de construir para completar el objetivo de producto interno con modelos propios.

## Componentes pendientes

### 1. Conectar frontend con backend

- Reemplazar o complementar la inferencia actual del navegador con llamadas al backend.
- Consumir `GET /models` para poblar el selector de modelos.
- Enviar imagenes a `POST /predict`.
- Mostrar la respuesta normalizada del backend.

### 2. Implementar inferencia backend real

- Crear cargador de modelos en `ml/inference`.
- Leer `ml/registry.json`.
- Cargar el modelo seleccionado.
- Preprocesar imagenes en backend.
- Ejecutar prediccion.
- Devolver etiquetas, confianza y metadatos.

### 3. Definir formato final de modelos

Elegir uno o varios formatos soportados:

- TensorFlow/Keras (`.keras`).
- PyTorch (`.pt`).
- ONNX (`.onnx`).
- TensorFlow.js exportado si se mantiene compatibilidad web.

### 4. Crear estructura formal de datasets

Propuesta:

```text
ml/datasets/<dataset-id>/
|-- train/
|-- validation/
|-- test/
`-- metadata.json
```

Pendiente:

- Validar imagenes.
- Contar clases.
- Detectar datasets desbalanceados.
- Separar train/validation/test.

### 5. Construir scripts de entrenamiento

En `ml/training/` crear:

- `split_dataset.py`
- `train_classifier.py`
- `evaluate.py`
- `export_model.py`
- `config.yaml`

### 6. Registrar modelos entrenados

Cada entrenamiento debe generar:

```text
ml/models/<model-name>/<version>/
|-- model.keras
|-- labels.json
|-- metrics.json
|-- config.yaml
`-- training.log
```

Y actualizar `ml/registry.json`.

### 7. Normalizar respuesta de prediccion

El backend deberia responder siempre con una estructura estable:

```json
{
  "model_id": "frutas-v1",
  "predictions": [
    { "label": "manzana", "confidence": 0.95 }
  ],
  "processing_time_ms": 123
}
```

### 8. Pruebas

Agregar pruebas para:

- Lectura de `registry.json`.
- Validacion de imagenes.
- Endpoint `/models`.
- Endpoint `/predict`.
- Build del frontend.

### 9. Despliegue futuro

- Mantener frontend en GitHub Pages o moverlo a un hosting de frontend.
- Desplegar backend en Render, Railway, Fly.io, Cloud Run o VPS.
- Definir almacenamiento persistente para `ml/models` y `ml/datasets` si se despliega fuera de la maquina local.

## Decisiones tomadas

- No usar base de datos inicialmente.
- No implementar usuarios.
- No implementar autenticacion.
- No implementar cola de entrenamiento.
- Usar carpetas y archivos (`ml/registry.json`) como fuente de verdad inicial.
