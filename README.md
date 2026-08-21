# Clasificador de Imagenes

Proyecto interno para clasificar imagenes con una arquitectura separada en frontend, backend y area de machine learning. La web funciona como interfaz de uso; los modelos, datasets y entrenamiento viven fuera del frontend.

## Objetivo

Construir una plataforma interna donde se puedan crear, almacenar, entrenar y usar modelos propios de clasificacion de imagenes.

El usuario de la web solo debe:

- Escoger un modelo disponible.
- Subir una imagen o usar webcam.
- Ver predicciones y resultados.

La administracion de datasets, entrenamiento y modelos se maneja desde carpetas y scripts internos, no desde cuentas de usuario ni autenticacion.

## Arquitectura

```text
Clasificador_Imagenes/
|-- frontend/              # Interfaz web Vite/TensorFlow.js actual
|-- backend/               # API interna FastAPI
|-- ml/                    # Datasets, modelos, entrenamiento e inferencia
|-- docs/                  # Documentacion tecnica y roadmap
|-- .github/workflows/     # Deploy del frontend a GitHub Pages
|-- SETUP.md
`-- README.md
```

## Frontend

Ubicacion: `frontend/`

Responsabilidades actuales:

- Mostrar la interfaz de clasificacion.
- Cargar el listado de modelos desde el backend cuando este disponible.
- Permitir seleccionar modelos del navegador o del backend.
- Permitir subir imagenes o usar webcam.
- Validar imagenes antes de procesarlas.
- Mostrar predicciones, confianza y tiempo de procesamiento.
- Dibujar cajas de deteccion para COCO-SSD.
- Mantener GitHub Pages funcionando para pruebas remotas con modelos del navegador.
- Usar fallback automatico a modelos del navegador si el backend no esta disponible.

Comandos:

```bash
cd frontend
npm install
npm run dev
npm run build
npm run preview
```

## Backend

Ubicacion: `backend/`

Responsabilidades actuales:

- Exponer API interna para la web.
- Listar modelos registrados en `ml/registry.json`.
- Recibir imagenes para inferencia.
- Ejecutar el modelo seleccionado desde backend.
- Devolver predicciones normalizadas al frontend.
- Validar tipo y tamano de imagen antes de procesar.

Endpoints:

- `GET /health`
- `GET /models`
- `POST /predict`

Ejecucion local:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## ML

Ubicacion: `ml/`

Responsabilidades:

- Guardar datasets.
- Guardar modelos entrenados.
- Mantener scripts de entrenamiento.
- Mantener scripts de inferencia.
- Registrar modelos disponibles en `ml/registry.json`.

Estructura:

```text
ml/
|-- datasets/
|-- models/
|-- training/
|-- inference/
`-- registry.json
```

## Registro de modelos

No se usara base de datos inicialmente. El backend leera `ml/registry.json`.

Registro actual:

```json
{
  "default_model": "visual-profile-v1",
  "models": [
    {
      "id": "visual-profile-v1",
      "name": "Perfil visual backend",
      "version": "v1",
      "runtime": "backend_builtin",
      "task": "image_profile_classification"
    }
  ]
}
```

## Modelos actuales

### MobileNet (Navegador)

Modelo preentrenado de TensorFlow.js para clasificacion general de imagenes. Corre directamente en el navegador, soporta imagenes cargadas y webcam, y devuelve etiquetas con probabilidad.

### COCO-SSD (Navegador)

Modelo preentrenado de TensorFlow.js para deteccion de objetos. Corre directamente en el navegador, soporta imagenes cargadas y webcam, y permite dibujar cajas de deteccion sobre la imagen.

### Perfil visual backend

Modelo interno de referencia registrado como `visual-profile-v1`. Corre en FastAPI con Pillow y clasifica propiedades visuales basicas: brillo, color dominante y orientacion. No es todavia un modelo entrenado propio; existe para validar el flujo backend completo mientras se construyen datasets, entrenamiento y exportacion de modelos reales.

## GitHub Pages

El workflow publica solo el frontend desde `frontend/dist`.

URL actual:

```text
https://abdyhernandez-proy.github.io/clasificador-imagenes/
```

## Sin base de datos, usuarios ni autenticacion

Por decision de arquitectura, el proyecto no incluye inicialmente:

- Usuarios.
- Login.
- Autenticacion.
- Cola de entrenamiento.
- Base de datos.

La persistencia inicial se resuelve con carpetas y archivos versionables.
