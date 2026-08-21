# Guia Rapida de Inicio

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

La web local abre en `http://localhost:3000`.

## Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API local:

```text
http://localhost:8000/health
http://localhost:8000/models
```

## Build del frontend

```powershell
cd frontend
npm run build
```

## Estructura nueva

- `frontend/`: interfaz web.
- `backend/`: API interna.
- `ml/`: datasets, modelos, entrenamiento e inferencia.
- `docs/`: roadmap y documentacion tecnica.
