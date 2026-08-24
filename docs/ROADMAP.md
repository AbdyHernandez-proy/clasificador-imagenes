# Roadmap

## Estado actual

El proyecto ya cuenta con una base funcional separada en frontend, backend y area ML.

- Frontend Vite con modelos del navegador y fallback cuando el backend no esta disponible.
- Selector visual de modelos con metadatos de uso.
- Carga de imagenes, arrastrar y soltar, webcam, vista previa y resultados.
- Modelos del navegador: MobileNet, COCO-SSD, BlazeFace, HandPose y BodyPix.
- Backend FastAPI con `GET /health`, `GET /models` y `POST /predict`.
- Registro de modelos en `ml/registry.json`.
- Modelo backend interno de referencia: `visual-profile-v1`.
- Auditoria actual: sin incidentes activos.

## Objetivo de producto completo

El proyecto se considera completo cuando permita usar la web como interfaz de interaccion y mantener fuera del frontend todo el flujo interno de modelos propios:

- Crear y organizar datasets.
- Entrenar modelos propios.
- Guardar artefactos entrenados.
- Registrar modelos disponibles.
- Ejecutar inferencia backend con modelos reales.
- Seleccionar modelos desde la web.
- Procesar imagenes desde archivo o webcam.
- Validar el flujo con pruebas automatizadas.
- Documentar instalacion, entrenamiento, inferencia y despliegue.

No forman parte del alcance actual:

- Manejo de usuarios.
- Login o autenticacion.
- Cola de entrenamiento.
- Panel web de administracion de datasets.
- Base de datos obligatoria.

## Completado

1. Conectar frontend con backend.
   - El frontend consume `GET /models` cuando la API esta disponible.
   - El selector combina modelos del navegador y modelos del backend.
   - Las imagenes se envian a `POST /predict` cuando se selecciona un modelo backend.
   - Si el backend no esta disponible, la interfaz mantiene los modelos del navegador como fallback.

2. Implementar inferencia backend de referencia.
   - El backend valida imagenes recibidas por `POST /predict`.
   - `visual-profile-v1` ejecuta inferencia interna con Pillow.
   - La respuesta incluye `model_id`, `model_name`, metadatos de imagen, predicciones y tiempo de procesamiento.
   - Este modelo valida el flujo completo, pero no sustituye un modelo entrenado propio.

3. Mejorar experiencia de seleccion de modelos.
   - El selector ya no depende de una lista desplegable.
   - Cada modelo muestra su uso principal: imagenes, objetos, rostros, manos o personas.
   - El selector se comporta como grupo de botones tipo radio.

## Pendiente para completar el proyecto

### 1. Repositorio interno de datasets

Crear estructura formal dentro de `ml/` para almacenar datasets de entrenamiento, validacion y pruebas.

Estructura propuesta:

```text
ml/
|-- datasets/
|   |-- raw/
|   |-- processed/
|   |-- splits/
|   `-- README.md
|-- models/
|-- training/
|-- inference/
`-- registry.json
```

Criterios de completado:

- Carpeta por dataset.
- Convencion de clases documentada.
- Separacion minima entre entrenamiento, validacion y pruebas.
- Archivo de metadatos por dataset.
- Reglas claras para no versionar imagenes pesadas innecesarias.

### 2. Scripts de preparacion de datos

Crear herramientas internas para preparar datasets antes del entrenamiento.

Debe incluir:

- Validacion de formatos permitidos.
- Redimensionamiento opcional.
- Normalizacion de nombres.
- Separacion train/validation/test.
- Reporte de cantidad de imagenes por clase.
- Deteccion de clases vacias o desbalanceadas.

Criterios de completado:

- Un comando reproducible para preparar un dataset.
- Salida documentada en `ml/datasets/processed/`.
- Reporte legible para revisar calidad de datos.

### 3. Entrenamiento de modelos propios

Crear scripts para entrenar al menos un modelo propio de clasificacion.

Ruta propuesta:

```text
ml/training/
|-- train_classifier.py
|-- evaluate_classifier.py
`-- config.example.json
```

Primer objetivo tecnico:

- Clasificador de imagenes por carpetas de clases.
- Transfer learning con un backbone preentrenado.
- Exportacion de metricas de entrenamiento.
- Guardado del modelo entrenado en `ml/models/`.

Criterios de completado:

- Entrenamiento ejecutable por comando.
- Configuracion externa sin editar codigo.
- Metricas basicas: accuracy, loss y matriz de confusion.
- Artefacto entrenado versionado o documentado.

### 4. Formato estandar de modelos

Definir como se guarda cada modelo propio.

Estructura propuesta:

```text
ml/models/
`-- nombre-modelo/
    |-- version.txt
    |-- model/
    |-- labels.json
    |-- metrics.json
    `-- model_card.md
```

Criterios de completado:

- Cada modelo tiene etiquetas.
- Cada modelo tiene version.
- Cada modelo tiene descripcion de uso.
- Cada modelo tiene metricas.
- Cada modelo indica runtime esperado.

### 5. Registro de modelos robusto

Ampliar `ml/registry.json` para que no solo liste modelos, sino que describa como cargarlos.

Campos propuestos:

- `id`
- `name`
- `version`
- `runtime`
- `task`
- `description`
- `labels`
- `artifact_path`
- `input_size`
- `preprocessing`
- `metrics_path`
- `enabled`

Criterios de completado:

- El backend ignora modelos deshabilitados.
- El frontend muestra solo modelos disponibles.
- El backend valida que los artefactos existan antes de exponer el modelo.
- Los errores de registro son claros.

### 6. Inferencia backend con modelos reales

Implementar runtimes backend adicionales para cargar modelos entrenados.

Opciones iniciales:

- `pytorch`
- `onnx`
- `tensorflow_saved_model`
- `sklearn_or_pillow_builtin`

Primer runtime recomendado:

- ONNX para inferencia portable y ligera.

Criterios de completado:

- `POST /predict` puede ejecutar al menos un modelo propio entrenado.
- El resultado mantiene el formato normalizado usado por el frontend.
- El backend valida tamano, formato, canal de color e input esperado.
- Los errores de modelo no rompen toda la API.

### 7. Pruebas automatizadas

Agregar pruebas minimas para frontend y backend.

Backend:

- `GET /health`.
- `GET /models`.
- `POST /predict` con imagen valida.
- `POST /predict` con archivo invalido.
- Modelo inexistente.

Frontend:

- Build de Vite.
- Validacion de archivos.
- Render de selector de modelos.
- Fallback cuando backend no responde.

Criterios de completado:

- Comandos documentados.
- Pruebas ejecutables localmente.
- El flujo de build no queda verde si algo basico se rompe.

### 8. Documentacion operativa

Actualizar documentacion para cubrir uso real interno.

Debe incluir:

- Instalacion frontend.
- Instalacion backend.
- Como preparar datasets.
- Como entrenar un modelo.
- Como registrar un modelo.
- Como ejecutar inferencia.
- Como desplegar frontend.
- Como exponer backend para pruebas remotas si aplica.

Criterios de completado:

- `README.md` explica el proyecto sin mezclar auditoria.
- `docs/ROADMAP.md` mantiene los pasos pendientes.
- `docs/MODELOS.md` describe modelos disponibles y candidatos.
- Cada script nuevo tiene ejemplo de uso.

### 9. Despliegue backend

Definir como se ejecutara la API fuera del entorno local.

Opciones:

- Servidor local interno.
- VPS.
- Render/Fly/Railway u otro proveedor.
- Tunnel temporal solo para pruebas.

Criterios de completado:

- URL backend configurable desde frontend.
- CORS limitado al dominio usado.
- Variables de entorno documentadas.
- Instrucciones para iniciar, detener y actualizar la API.

### 10. Revision final de producto

Antes de considerar el proyecto completo:

- Ejecutar auditoria tecnica.
- Ejecutar pruebas automatizadas.
- Probar frontend con backend apagado.
- Probar frontend con backend encendido.
- Probar al menos un modelo del navegador.
- Probar al menos un modelo backend propio.
- Verificar README, ROADMAP y MODELOS.
- Verificar que no haya secretos, datasets pesados o archivos temporales versionados.

## Orden recomendado de ejecucion

1. Crear estructura `ml/datasets`, `ml/training`, `ml/inference` y `ml/models`.
2. Crear script de preparacion de datasets.
3. Crear script de entrenamiento de clasificador propio.
4. Definir formato estandar de modelo exportado.
5. Ampliar `ml/registry.json`.
6. Implementar runtime backend real.
7. Registrar y probar el primer modelo propio.
8. Agregar pruebas automatizadas.
9. Completar documentacion operativa.
10. Definir despliegue backend.
