from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.services.inference import InferenceService
from app.services.model_registry import ModelRegistry

app = FastAPI(title="Clasificador de Imagenes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry()
inference_service = InferenceService(registry)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict:
    return {
        "default_model": registry.get_default_model_id(),
        "models": registry.list_models()
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...), model_id: str | None = None) -> dict:
    return await inference_service.predict(image=image, model_id=model_id)
