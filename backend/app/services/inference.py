from fastapi import UploadFile

from app.services.model_registry import ModelRegistry


class InferenceService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    async def predict(self, image: UploadFile, model_id: str | None = None) -> dict:
        selected_model_id = model_id or self.registry.get_default_model_id()
        if not selected_model_id:
            return {
                "status": "not_configured",
                "message": "No hay un modelo predeterminado configurado en ml/registry.json.",
                "predictions": []
            }

        model = self.registry.get_model(selected_model_id)
        if not model:
            return {
                "status": "not_found",
                "message": f"El modelo '{selected_model_id}' no existe en ml/registry.json.",
                "predictions": []
            }

        return {
            "status": "pending_runtime",
            "message": "El modelo existe en el registro, pero falta conectar el runtime de inferencia.",
            "model": model,
            "filename": image.filename,
            "predictions": []
        }
