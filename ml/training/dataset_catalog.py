from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.training.paths import DATASET_REGISTRY_PATH, PROJECT_ROOT


def load_dataset_registry(registry_path: Path = DATASET_REGISTRY_PATH) -> dict[str, Any]:
    with registry_path.open("r", encoding="utf-8-sig") as registry_file:
        return json.load(registry_file)


def list_datasets() -> list[dict[str, Any]]:
    return load_dataset_registry().get("datasets", [])


def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    for dataset in list_datasets():
        if dataset.get("id") == dataset_id:
            return dataset
    return None


def resolve_project_path(relative_or_absolute: str | None) -> Path | None:
    if not relative_or_absolute:
        return None

    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
