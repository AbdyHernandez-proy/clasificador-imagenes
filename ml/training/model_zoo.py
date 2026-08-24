from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.training.paths import MODEL_CONFIGS_ROOT


def load_detector_configs(config_root: Path = MODEL_CONFIGS_ROOT) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    for config_path in sorted(config_root.glob("*.json")):
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        config["config_path"] = str(config_path)
        configs.append(config)

    return configs


def get_detector_config(model_id: str) -> dict[str, Any] | None:
    for config in load_detector_configs():
        if config.get("id") == model_id:
            return config
    return None


def compatible_detectors_for_dataset(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_format = dataset.get("format")
    dataset_task = dataset.get("task")
    compatible: list[dict[str, Any]] = []

    for config in load_detector_configs():
        formats = config.get("compatible_dataset_formats", [])
        tasks = config.get("compatible_dataset_tasks", [])
        if dataset_format in formats and dataset_task in tasks:
            compatible.append(config)

    return compatible
