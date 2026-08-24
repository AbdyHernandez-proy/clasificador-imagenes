import json
from pathlib import Path
from typing import Any

from app.core.paths import REGISTRY_PATH


class ModelRegistry:
    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self.registry_path = registry_path

    def load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"schema_version": 1, "models": [], "default_model": None}

        with self.registry_path.open("r", encoding="utf-8") as registry_file:
            return json.load(registry_file)

    def save(self, registry: dict[str, Any]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("w", encoding="utf-8") as registry_file:
            json.dump(registry, registry_file, ensure_ascii=False, indent=2)
            registry_file.write("\n")

    def list_models(self, serving_only: bool = True) -> list[dict[str, Any]]:
        models = self.load().get("models", [])
        if not serving_only:
            return models
        return [
            model for model in models
            if model.get("serve", True) and model.get("status", "ready") == "ready"
        ]

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        for model in self.list_models(serving_only=False):
            if model.get("id") == model_id:
                return model
        return None

    def get_default_model_id(self) -> str | None:
        return self.load().get("default_model")

    def register_model(self, model: dict[str, Any]) -> dict[str, Any]:
        if not model.get("id"):
            raise ValueError("El modelo necesita un campo 'id'.")

        registry = self.load()
        models = registry.setdefault("models", [])
        existing_index = next(
            (index for index, item in enumerate(models) if item.get("id") == model["id"]),
            None
        )

        if existing_index is None:
            models.append(model)
        else:
            models[existing_index] = {**models[existing_index], **model}

        self.save(registry)
        return model
