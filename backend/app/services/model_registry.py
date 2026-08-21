import json
from pathlib import Path
from typing import Any

from app.core.paths import REGISTRY_PATH


class ModelRegistry:
    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self.registry_path = registry_path

    def load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"models": [], "default_model": None}

        with self.registry_path.open("r", encoding="utf-8") as registry_file:
            return json.load(registry_file)

    def list_models(self) -> list[dict[str, Any]]:
        return self.load().get("models", [])

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        for model in self.list_models():
            if model.get("id") == model_id:
                return model
        return None

    def get_default_model_id(self) -> str | None:
        return self.load().get("default_model")
