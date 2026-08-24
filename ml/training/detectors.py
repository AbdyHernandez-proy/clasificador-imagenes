from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.training.dataset_catalog import get_dataset, resolve_project_path
from ml.training.datasets.yolo_detection import summarize_yolo_dataset
from ml.training.model_zoo import get_detector_config


@dataclass(frozen=True)
class TrainingPlan:
    model_id: str
    model_name: str
    dataset_id: str
    dataset_path: str
    framework: str
    status: str
    notes: list[str]
    dataset_summary: dict[str, int] | None = None


class DetectorScaffold:
    def __init__(self, model_id: str, dataset_id: str | None = None):
        self.config = get_detector_config(model_id)
        if not self.config:
            raise ValueError(f"Modelo no registrado en ml/training/configs: {model_id}")

        self.dataset_id = dataset_id or self.config.get("default_dataset_id")
        self.dataset = get_dataset(self.dataset_id)
        if not self.dataset:
            raise ValueError(f"Dataset no registrado: {self.dataset_id}")

    def build_plan(self) -> TrainingPlan:
        dataset_path = resolve_project_path(self.dataset.get("dataset_path"))
        notes = [
            "Scaffold creado: aun no ejecuta entrenamiento real.",
            "El siguiente paso es instalar el framework elegido y conectar el trainer especifico.",
            "El artefacto entrenado debe guardarse en ml/models/ y registrarse en ml/registry.json."
        ]
        dataset_summary = None

        if self.dataset.get("format") == "yolo" and dataset_path:
            train_split = self.dataset.get("splits", {}).get("train")
            if train_split:
                dataset_summary = summarize_yolo_dataset(dataset_path, train_split)

        return TrainingPlan(
            model_id=self.config["id"],
            model_name=self.config["name"],
            dataset_id=self.dataset_id,
            dataset_path=str(dataset_path) if dataset_path else "",
            framework=self.config["framework"],
            status=self.config["status"],
            notes=notes,
            dataset_summary=dataset_summary
        )

    def train(self) -> None:
        raise NotImplementedError(
            f"El entrenamiento real para {self.config['name']} aun no esta implementado. "
            "Usa build_plan() para validar dataset/configuracion antes de conectar el framework."
        )


def build_detector_plan(model_id: str, dataset_id: str | None = None) -> dict[str, Any]:
    plan = DetectorScaffold(model_id=model_id, dataset_id=dataset_id).build_plan()
    return {
        "model_id": plan.model_id,
        "model_name": plan.model_name,
        "dataset_id": plan.dataset_id,
        "dataset_path": plan.dataset_path,
        "framework": plan.framework,
        "status": plan.status,
        "dataset_summary": plan.dataset_summary,
        "notes": plan.notes
    }
