from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.services.dataset_registry import DatasetRegistry
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
dataset_registry = DatasetRegistry()
inference_service = InferenceService(registry)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("shutdown")
def shutdown_workers() -> None:
    inference_service.close()


@app.get("/models")
def list_models() -> dict:
    return {
        "default_model": registry.get_default_model_id(),
        "models": [model_public_summary(model) for model in registry.list_models()]
    }


@app.get("/models/registry")
def list_model_registry() -> dict:
    return registry.load()


@app.post("/models/preload")
def preload_models() -> dict:
    return inference_service.preload_models_async()


@app.get("/models/preload")
def preload_status() -> dict:
    return inference_service.get_preload_status()


@app.get("/datasets")
def list_datasets() -> dict:
    return {
        "datasets": dataset_registry.list_datasets()
    }


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    dataset = dataset_registry.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' no existe.")
    return {"dataset": dataset}


@app.post("/predict")
async def predict(image: UploadFile = File(...), model_id: str | None = Form(default=None)) -> dict:
    return await inference_service.predict(image=image, model_id=model_id)


def model_public_summary(model: dict) -> dict:
    keys = [
        "id",
        "name",
        "version",
        "runtime",
        "task",
        "status",
        "serve",
        "description",
        "efficiency",
        "confidence_threshold",
        "iou_threshold",
        "nms_threshold",
        "max_detections",
    ]
    return {key: model[key] for key in keys if key in model}
