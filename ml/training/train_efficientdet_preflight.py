from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime

from ml.training.dataset_catalog import resolve_project_path
from ml.training.datasets.yolo_detection import build_yolo_index
from ml.training.local_train import get_dataset_classes, get_required_dataset
from ml.training.paths import MODEL_ARTIFACTS_ROOT


MODEL_ID = "custom-efficientdet-detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida preparacion de EfficientDet.")
    parser.add_argument("--dataset-id", default="voc-detect")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--base-model", default="tf_efficientdet_d0")
    return parser.parse_args()


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    args = parse_args()
    dataset = get_required_dataset(args.dataset_id)
    dataset_root = resolve_project_path(dataset.get("dataset_path"))
    if not dataset_root or not dataset_root.exists():
        raise SystemExit(f"Dataset no disponible localmente: {dataset.get('dataset_path')}")

    train_split = dataset.get("splits", {}).get("train")
    val_split = dataset.get("splits", {}).get("val")
    if not train_split:
        raise SystemExit("Dataset sin split train.")

    train_samples = build_yolo_index(dataset_root=dataset_root, image_split=train_split)
    val_samples = build_yolo_index(dataset_root=dataset_root, image_split=val_split) if val_split else []
    classes = get_dataset_classes(dataset)

    artifact_dir = MODEL_ARTIFACTS_ROOT / MODEL_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": MODEL_ID,
        "status": "preflight_ready",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": args.dataset_id,
        "base_model": args.base_model,
        "image_size": args.imgsz,
        "class_count": len(classes),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "dependencies": {
            "effdet": has_module("effdet"),
            "timm": has_module("timm"),
            "pycocotools": has_module("pycocotools"),
        },
        "next_step": "Ejecutar entrenamiento EfficientDet, revisar validacion por epoca y calibrar umbrales antes de activar servicio.",
    }
    report_path = artifact_dir / "preflight_report.json"
    try:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except PermissionError as exc:
        report["report_write_warning"] = str(exc)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if not report["dependencies"]["effdet"]:
        raise SystemExit("Falta dependencia effdet. Ejecuta scripts\\training\\train-efficientdet.ps1 -Install antes de entrenar.")


if __name__ == "__main__":
    main()
