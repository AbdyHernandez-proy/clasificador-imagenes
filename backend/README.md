# Backend

API interna para listar modelos disponibles y recibir imagenes para inferencia.

## Ejecutar en desarrollo

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints iniciales

- `GET /health`: verifica que el backend responde.
- `GET /models`: lista modelos registrados en `ml/registry.json`.
- `POST /predict`: punto de entrada para inferencia backend. Queda preparado para conectar el runtime del modelo activo.

## Nota

Este backend no maneja usuarios, autenticacion ni base de datos. Lee configuracion desde archivos del proyecto.
