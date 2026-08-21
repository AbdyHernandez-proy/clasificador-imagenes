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
|-- AUDITORIA.md           # Estado y pasos siguientes
|-- SETUP.md
`-- README.md
```

## Frontend

Ubicacion: `frontend/`

Responsabilidades actuales:

- Mostrar la interfaz de clasificacion.
- Permitir seleccionar modelo.
- Permitir subir imagenes o usar webcam.
- Validar imagenes antes de procesarlas.
- Mostrar predicciones, confianza y tiempo de procesamiento.
- Dibujar cajas de deteccion para COCO-SSD.
- Mantener GitHub Pages funcionando para pruebas remotas.

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

Responsabilidades previstas:

- Exponer API interna para la web.
- Listar modelos registrados en `ml/registry.json`.
- Recibir imagenes para inferencia.
- Ejecutar el modelo seleccionado desde backend.
- Devolver predicciones normalizadas al frontend.

Endpoints iniciales:

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

Ejemplo futuro:

```json
{
  "default_model": "frutas-v1",
  "models": [
    {
      "id": "frutas-v1",
      "name": "Clasificador de frutas",
      "version": "v1",
      "path": "ml/models/frutas/v1/model.keras",
      "labels": "ml/models/frutas/v1/labels.json",
      "metrics": "ml/models/frutas/v1/metrics.json"
    }
  ]
}
```

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

## Auditoria y siguientes pasos

Ver `AUDITORIA.md`.
