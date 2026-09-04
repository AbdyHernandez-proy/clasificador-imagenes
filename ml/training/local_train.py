from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from ml.training.dataset_catalog import get_dataset, resolve_project_path
from ml.training.datasets.coco_classes import COCO_CLASSES
from ml.training.datasets.yolo_detection import build_yolo_index, resolve_label_path
from ml.training.paths import MODEL_ARTIFACTS_ROOT, MODEL_REGISTRY_PATH, PROJECT_ROOT, RUNS_ROOT

MODEL_ALIASES = {
    "yolo": "custom-yolo-v8-v11-detector",
    "yolo-v8-v11": "custom-yolo-v8-v11-detector",
    "faster-rcnn": "custom-faster-rcnn-detector",
    "faster_rcnn": "custom-faster-rcnn-detector",
    "retinanet": "custom-retinanet-detector",
    "rtdetr": "custom-detr-rtdetr-detector",
    "detr": "custom-detr-rtdetr-detector",
    "efficientdet": "custom-efficientdet-detector"
}

DEFAULT_MODELS = ["yolo", "faster-rcnn", "retinanet", "rtdetr"]

FINAL_ARTIFACT_FILENAMES = {
    "custom-yolo-v8-v11-detector": "yolo-v8-v11-detector.pt",
    "custom-faster-rcnn-detector": "faster-rcnn-detector.pt",
    "custom-retinanet-detector": "retinanet-detector.pt",
    "custom-detr-rtdetr-detector": "rtdetr-detector.pt",
    "custom-efficientdet-detector": "efficientdet-detector.pt",
}


def final_artifact_filename(model_id: str) -> str:
    return FINAL_ARTIFACT_FILENAMES.get(model_id, f"{model_id}.pt")


def latest_artifact_filename(model_id: str) -> str:
    return f"{Path(final_artifact_filename(model_id)).stem}-latest.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script local desechable para entrenar/probar detectores propios."
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Lista: yolo,faster-rcnn,retinanet,rtdetr,efficientdet o all.")
    parser.add_argument("--dataset-id", default="coco128-detect")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="auto", help="auto, cpu o indice CUDA como 0.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 usa todo el dataset.")
    parser.add_argument("--session-samples", type=int, default=0, help="0 procesa el dataset completo. Mayor a 0 procesa solo una seccion y guarda checkpoint.")
    parser.add_argument("--session-count", type=int, default=1, help="Cantidad de secciones consecutivas a procesar en el mismo proceso Python.")
    parser.add_argument("--log-every", type=int, default=25, help="Cantidad de lotes entre logs de progreso.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="0 guarda solo al final de cada seccion. Mayor a 0 guarda cada N lotes.")
    parser.add_argument("--resume", action="store_true", help="Reanuda desde checkpoint_latest.pt si existe.")
    parser.add_argument("--publish-partial", action="store_true", help="Publica el artefacto latest aunque el objetivo de epocas no haya terminado.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0025, help="Learning rate para entrenadores TorchVision.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum SGD para entrenadores TorchVision.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay para entrenadores TorchVision.")
    parser.add_argument("--lr-step-size", type=int, default=0, help="Epocas entre pasos del scheduler StepLR. 0 desactiva scheduler.")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="Factor de reduccion del scheduler StepLR.")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0, help="Congela el backbone durante N epocas iniciales en TorchVision.")
    parser.add_argument("--validate-every-epoch", action="store_true", help="Valida al cerrar cada epoca y guarda el artefacto final del modelo.")
    parser.add_argument("--validation-limit", type=int, default=250, help="Imagenes de validacion por epoca. 0 usa todo el split val.")
    parser.add_argument("--validation-confidence", type=float, default=-1.0, help="Umbral de validacion. -1 usa el del registro del modelo.")
    parser.add_argument("--validation-max-detections", type=int, default=0, help="Max detecciones por imagen. 0 usa el registro del modelo.")
    parser.add_argument("--validation-visual-limit", type=int, default=12, help="Cantidad de imagenes anotadas a guardar por validacion. 0 desactiva salidas visuales.")
    parser.add_argument("--best-metric", default="map50_95", choices=["map50_95", "map50", "precision_at_50", "recall_at_50"], help="Metrica usada para elegir el artefacto final del modelo.")
    parser.add_argument("--min-delta", type=float, default=0.0001, help="Mejora minima para reemplazar best checkpoint.")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Epocas sin mejora antes de detener. 0 desactiva early stopping.")
    parser.add_argument("--validate-before-training", action="store_true", help="Evalua el checkpoint base antes de entrenar y lo guarda como best si corresponde.")
    parser.add_argument("--allow-efficientdet", action="store_true", help="Permite entrenar EfficientDet desde el trainer local.")
    parser.add_argument("--efficientdet-base-model", default="tf_efficientdet_d0", help="Arquitectura base EfficientDet.")
    parser.add_argument("--efficientdet-pretrained-backbone", action="store_true", help="Usa backbone preentrenado para EfficientDet si el entorno puede descargar/cachear pesos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = get_required_dataset(args.dataset_id)
    dataset_root = resolve_project_path(dataset.get("dataset_path"))
    if not dataset_root or not dataset_root.exists():
        raise SystemExit(f"Dataset no disponible localmente: {dataset.get('dataset_path')}")

    run_root = RUNS_ROOT / f"local-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    data_yaml_path = write_yolo_data_yaml(dataset=dataset, dataset_root=dataset_root, run_root=run_root)
    selected_models = parse_model_list(args.models)
    results: list[dict[str, Any]] = []

    for model_key in selected_models:
        print(f"\n=== Entrenando/preparando {model_key} ===", flush=True)
        try:
            if model_key == "yolo":
                result = train_ultralytics_yolo(args, data_yaml_path, run_root, dataset)
            elif model_key == "rtdetr":
                result = train_ultralytics_rtdetr(args, data_yaml_path, run_root)
            elif model_key == "faster-rcnn":
                result = train_torchvision_detector(args, dataset, dataset_root, run_root, model_key)
            elif model_key == "retinanet":
                result = train_torchvision_detector(args, dataset, dataset_root, run_root, model_key)
            elif model_key == "efficientdet":
                result = train_efficientdet_detector(args, dataset, dataset_root, run_root)
            else:
                result = {"model": model_key, "status": "skipped", "reason": "Modelo no reconocido."}
        except Exception as exc:  # noqa: BLE001 - script local: registrar y seguir con el siguiente modelo.
            result = {"model": model_key, "status": "failed", "error": str(exc) or repr(exc)}

        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        results.append(result)

    report_path = run_root / "training_report.json"
    report_path.write_text(json.dumps({"dataset": dataset["id"], "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReporte final: {report_path}", flush=True)


def get_required_dataset(dataset_id: str) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise SystemExit(f"Dataset no registrado: {dataset_id}")
    if dataset.get("format") != "yolo":
        raise SystemExit(f"Este script local solo soporta datasets YOLO por ahora. Formato recibido: {dataset.get('format')}")
    return dataset


def parse_model_list(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return ["yolo", "faster-rcnn", "retinanet", "rtdetr", "efficientdet"]

    models: list[str] = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key in {"custom-yolo-v8-v11-detector", "custom-detr-rtdetr-detector"}:
            key = "yolo" if "yolo" in key else "rtdetr"
        elif key == "custom-faster-rcnn-detector":
            key = "faster-rcnn"
        elif key == "custom-retinanet-detector":
            key = "retinanet"
        elif key == "custom-efficientdet-detector":
            key = "efficientdet"
        models.append(key)
    return models


def write_yolo_data_yaml(dataset: dict[str, Any], dataset_root: Path, run_root: Path) -> Path:
    classes = get_dataset_classes(dataset)
    train_split = materialize_ultralytics_split(dataset, dataset_root, dataset.get("splits", {}).get("train"), run_root, "train")
    val_split = materialize_ultralytics_split(
        dataset,
        dataset_root,
        dataset.get("splits", {}).get("val") or dataset.get("splits", {}).get("train"),
        run_root,
        "val",
    )
    config = {
        "path": str(dataset_root),
        "train": str(train_split),
        "val": str(val_split),
        "nc": len(classes),
        "names": classes
    }
    yaml_path = run_root / f"{dataset['id']}.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def materialize_ultralytics_split(
    dataset: dict[str, Any],
    dataset_root: Path,
    split_value: str | None,
    run_root: Path,
    split_name: str,
) -> Path | str:
    if not split_value:
        raise ValueError(f"Dataset sin split {split_name}.")

    split_path = dataset_root / split_value
    if not split_path.is_file():
        return split_value

    output_dir = run_root / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}_absolute.txt"
    lines = []
    for raw_line in split_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        image_path = Path(line)
        if not image_path.is_absolute():
            image_path = dataset_root / image_path
        lines.append(str(resolve_ultralytics_image_path(dataset, dataset_root, image_path).absolute()))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def resolve_ultralytics_image_path(dataset: dict[str, Any], dataset_root: Path, image_path: Path) -> Path:
    try:
        relative_path = image_path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return image_path

    parts = list(relative_path.parts)
    if "JPEGImages" not in parts:
        return image_path

    image_dir = image_path.parent
    label_dir = image_dir.parent / "labels"
    label_path = resolve_label_path(
        dataset_root=dataset_root,
        image_path=image_path,
        image_dir=None,
        label_dir=None,
    )
    if not label_path.exists():
        return image_path

    images_index = parts.index("JPEGImages")
    view_root = dataset_root / "ultralytics_views" / str(dataset["id"])
    view_parent = view_root / Path(*parts[:images_index])
    view_image_dir = view_parent / "images"
    view_label_dir = view_parent / "labels"
    view_image_path = view_image_dir / image_path.name
    view_label_path = view_label_dir / label_path.name

    ensure_link_or_hardlink(source=image_dir, target=view_image_dir, directory=True)
    ensure_link_or_hardlink(source=label_dir, target=view_label_dir, directory=True)

    if not view_image_path.exists():
        ensure_link_or_hardlink(source=image_path, target=view_image_path, directory=False)
    if not view_label_path.exists():
        ensure_link_or_hardlink(source=label_path, target=view_label_path, directory=False)

    return view_image_path


def ensure_link_or_hardlink(source: Path, target: Path, directory: bool) -> None:
    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, target, target_is_directory=directory)
        return
    except OSError:
        pass

    if directory:
        target.mkdir(parents=True, exist_ok=True)
        return

    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def get_dataset_classes(dataset: dict[str, Any]) -> list[str]:
    classes = dataset.get("classes") or dataset.get("names")
    if isinstance(classes, dict):
        return [classes[key] for key in sorted(classes, key=lambda item: int(item))]
    if isinstance(classes, list) and classes:
        return [str(class_name) for class_name in classes]
    return COCO_CLASSES


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def train_ultralytics_yolo(
    args: argparse.Namespace,
    data_yaml_path: Path,
    run_root: Path,
    dataset: dict[str, Any]
) -> dict[str, Any]:
    from ultralytics import YOLO

    model_id = "custom-yolo-v8-v11-detector"
    classes = get_dataset_classes(dataset)
    model = YOLO("yolo11n.pt")
    result = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=resolve_device(args.device),
        workers=args.workers,
        project=str(run_root),
        name=model_id,
        exist_ok=True
    )
    artifact_dir = MODEL_ARTIFACTS_ROOT / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(result.save_dir)
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    artifact_best_path = artifact_dir / final_artifact_filename(model_id)
    artifact_last_path = artifact_dir / latest_artifact_filename(model_id)

    if best_path.exists():
        shutil.copy2(best_path, artifact_best_path)
    if last_path.exists():
        shutil.copy2(last_path, artifact_last_path)

    metrics_summary = summarize_ultralytics_results(run_dir / "results.csv")
    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_model_registry(
        "custom-yolo-v8-v11-detector",
        {
            "status": "trained",
            "artifact_path": relative_project_path(artifact_best_path),
            "metrics": metrics_summary,
            "labels": classes,
            "training": {
                "dataset_id": args.dataset_id,
                "epochs": args.epochs,
                "batch": args.batch,
                "image_size": args.imgsz,
                "run_dir": relative_project_path(run_dir)
            }
        }
    )

    return {
        "model": model_id,
        "status": "trained",
        "run_dir": str(run_dir),
        "expected_best": str(best_path),
        "artifact_dir": str(artifact_dir),
        "artifact_path": str(artifact_best_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics_summary
    }


def train_ultralytics_rtdetr(args: argparse.Namespace, data_yaml_path: Path, run_root: Path) -> dict[str, Any]:
    from ultralytics import RTDETR

    model_id = "custom-detr-rtdetr-detector"
    if args.max_samples > 0:
        raise ValueError("RT-DETR no soporta --max-samples en este wrapper. Ejecuta un entrenamiento normal o crea un split temporal del dataset.")
    if args.session_samples > 0:
        print("custom-detr-rtdetr-detector ignora --session-samples; Ultralytics guarda checkpoints por epoca.", flush=True)

    artifact_dir = MODEL_ARTIFACTS_ROOT / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_best_path = artifact_dir / final_artifact_filename(model_id)
    artifact_last_path = artifact_dir / latest_artifact_filename(model_id)

    resume_checkpoint = find_latest_ultralytics_epoch_checkpoint(model_id) or artifact_last_path
    if args.resume and resume_checkpoint.exists():
        resume_checkpoint = prepare_ultralytics_resume_checkpoint(
            checkpoint_path=resume_checkpoint,
            target_epochs=args.epochs,
            data_yaml_path=data_yaml_path,
            model_id=model_id
        )

    if args.resume and resume_checkpoint.exists():
        model = RTDETR(str(resume_checkpoint))
        resume_training = True
        print(f"{model_id} reanudando desde {resume_checkpoint}", flush=True)
    else:
        model = RTDETR("rtdetr-l.pt")
        resume_training = False
        if args.resume:
            print(f"{model_id} no encontro artefacto latest; iniciando desde rtdetr-l.pt", flush=True)

    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": resolve_device(args.device),
        "workers": args.workers,
        "project": str(run_root),
        "name": model_id,
        "exist_ok": True,
        "save_period": 1,
    }
    if resume_training:
        train_kwargs["resume"] = True

    result = model.train(**train_kwargs)
    run_dir = Path(result.save_dir)
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"

    if best_path.exists():
        shutil.copy2(best_path, artifact_best_path)
    if last_path.exists():
        shutil.copy2(last_path, artifact_last_path)

    metrics_summary = summarize_ultralytics_results(run_dir / "results.csv")
    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "model": model_id,
        "status": "trained" if artifact_best_path.exists() else "completed_without_best",
        "run_dir": str(run_dir),
        "artifact_path": str(artifact_best_path) if artifact_best_path.exists() else None,
        "latest_artifact_path": str(artifact_last_path) if artifact_last_path.exists() else None,
        "metrics_path": str(metrics_path),
        "metrics": metrics_summary,
    }
    (artifact_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    update_model_registry(
        model_id,
        {
            "status": "ready" if artifact_best_path.exists() else "planned",
            "artifact_path": relative_project_path(artifact_best_path) if artifact_best_path.exists() else relative_project_path(artifact_last_path),
            "latest_artifact_path": relative_project_path(artifact_last_path) if artifact_last_path.exists() else None,
            "best_artifact_path": relative_project_path(artifact_best_path) if artifact_best_path.exists() else None,
            "metrics": metrics_summary,
            "confidence_threshold": 0.45,
            "iou_threshold": 0.55,
            "max_detections": 8,
            "labels": get_dataset_classes(get_required_dataset(args.dataset_id)),
            "training": {
                "dataset_id": args.dataset_id,
                "epochs": args.epochs,
                "target_epochs": args.epochs,
                "batch": args.batch,
                "image_size": args.imgsz,
                "base_weights": "rtdetr-l.pt",
                "run_dir": relative_project_path(run_dir),
                "training_record": "docs/DETR_RTDETR.md",
                "confidence_threshold": 0.45,
                "iou_threshold": 0.55,
                "max_detections": 8,
                "status": "calibrated",
            },
        },
    )

    return summary


def summarize_ultralytics_results(results_csv_path: Path) -> dict[str, Any]:
    if not results_csv_path.exists():
        return {"status": "metrics_not_found"}

    import csv

    with results_csv_path.open("r", encoding="utf-8", newline="") as results_file:
        rows = list(csv.DictReader(results_file))

    if not rows:
        return {"status": "metrics_empty"}

    def number(row: dict[str, str], key: str) -> float:
        return float(row.get(key, "0") or 0)

    best_map50 = max(rows, key=lambda row: number(row, "metrics/mAP50(B)"))
    best_map50_95 = max(rows, key=lambda row: number(row, "metrics/mAP50-95(B)"))
    final = rows[-1]

    return {
        "status": "available",
        "epochs": len(rows),
        "best_map50": {
            "epoch": int(float(best_map50["epoch"])),
            "precision": number(best_map50, "metrics/precision(B)"),
            "recall": number(best_map50, "metrics/recall(B)"),
            "map50": number(best_map50, "metrics/mAP50(B)"),
            "map50_95": number(best_map50, "metrics/mAP50-95(B)")
        },
        "best_map50_95": {
            "epoch": int(float(best_map50_95["epoch"])),
            "precision": number(best_map50_95, "metrics/precision(B)"),
            "recall": number(best_map50_95, "metrics/recall(B)"),
            "map50": number(best_map50_95, "metrics/mAP50(B)"),
            "map50_95": number(best_map50_95, "metrics/mAP50-95(B)")
        },
        "final": {
            "epoch": int(float(final["epoch"])),
            "precision": number(final, "metrics/precision(B)"),
            "recall": number(final, "metrics/recall(B)"),
            "map50": number(final, "metrics/mAP50(B)"),
            "map50_95": number(final, "metrics/mAP50-95(B)")
        }
    }


def train_torchvision_detector(
    args: argparse.Namespace,
    dataset: dict[str, Any],
    dataset_root: Path,
    run_root: Path,
    model_key: str
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn

    from ml.training.torchvision_validation import evaluate_torchvision_model, select_validation_samples

    classes = get_dataset_classes(dataset)
    train_split = dataset.get("splits", {}).get("train")
    if not train_split:
        raise ValueError("Dataset sin split train.")

    print(f"{model_key} preparando indice del dataset {dataset['id']}...", flush=True)
    samples = build_yolo_index(dataset_root, train_split)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    if not samples:
        raise ValueError("No hay muestras para entrenar.")

    validation_samples = []
    if args.validate_every_epoch:
        validation_split = dataset.get("splits", {}).get("val") or dataset.get("splits", {}).get("validation")
        if validation_split:
            validation_samples = select_validation_samples(build_yolo_index(dataset_root, validation_split), args.validation_limit)
            print(f"{model_key} validacion configurada: {len(validation_samples)} muestras.", flush=True)
        else:
            print(f"{model_key} sin split de validacion; se omite validacion por epoca.", flush=True)

    print(f"{model_key} indice listo: {len(samples)} muestras.", flush=True)
    device = torch.device("cuda:0" if resolve_device(args.device) != "cpu" and torch.cuda.is_available() else "cpu")
    num_classes = len(classes) + 1

    print(f"{model_key} construyendo arquitectura Torchvision...", flush=True)
    if model_key == "faster-rcnn":
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=num_classes,
            min_size=args.imgsz,
            max_size=args.imgsz
        )
        model_id = "custom-faster-rcnn-detector"
    else:
        model = retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=num_classes,
            min_size=args.imgsz,
            max_size=args.imgsz
        )
        model_id = "custom-retinanet-detector"

    artifact_dir = MODEL_ARTIFACTS_ROOT / model_id
    checkpoint_dir = artifact_dir / "checkpoints"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
    best_checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
    latest_artifact_path = artifact_dir / latest_artifact_filename(model_id)
    best_artifact_path = artifact_dir / final_artifact_filename(model_id)
    validation_history_path = artifact_dir / "validation_history.json"

    state = initial_training_state(model_id=model_id, dataset_id=dataset["id"], total_samples=len(samples))
    checkpoint: dict[str, Any] | None = None

    print(f"{model_id} moviendo modelo a {device}...", flush=True)
    model.to(device)

    if args.resume:
        checkpoint, loaded_checkpoint_path = load_resume_checkpoint(
            model_id=model_id,
            candidates=[checkpoint_path, best_checkpoint_path],
            device=device,
        )
        if checkpoint is None:
            raise RuntimeError(f"{model_id} no encontro un checkpoint valido para reanudar.")
    else:
        loaded_checkpoint_path = None

    if checkpoint:
        print(f"{model_id} aplicando pesos...", flush=True)
        model.load_state_dict(checkpoint["state_dict"])
        state = checkpoint.get("training_state", state)
        state = sync_training_state_with_dataset(state, total_samples=len(samples))
        print(f"{model_id} reanudado desde {loaded_checkpoint_path}", flush=True)

    if int(state.get("current_epoch", 0)) < args.freeze_backbone_epochs:
        set_backbone_trainable(model, False)
        state["backbone_frozen"] = True
    else:
        set_backbone_trainable(model, True)
        state["backbone_frozen"] = False

    print(f"{model_id} preparando optimizador lr={args.lr} momentum={args.momentum} weight_decay={args.weight_decay}...", flush=True)
    optimizer = create_torchvision_optimizer(model, args)
    scheduler = create_torchvision_scheduler(optimizer, args)
    if checkpoint:
        try:
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and checkpoint.get("scheduler_state"):
                scheduler.load_state_dict(checkpoint["scheduler_state"])
        except ValueError as exc:
            state["optimizer_resume_warning"] = str(exc)
            print(f"{model_id} no pudo reusar el optimizador previo; se continua con uno nuevo: {exc}", flush=True)

    loss_history: list[float] = list(state.get("loss_history", []))
    validation_history: list[dict[str, Any]] = list(state.get("validation_history", []))

    if args.validate_before_training and args.validate_every_epoch and validation_samples:
        bootstrap_best_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            classes=classes,
            state=state,
            validation_samples=validation_samples,
            validation_history=validation_history,
            validation_history_path=validation_history_path,
            best_checkpoint_path=best_checkpoint_path,
            best_artifact_path=best_artifact_path,
            model_id=model_id,
            args=args,
            device=device,
        )

    if int(state.get("current_epoch", 0)) >= args.epochs:
        return {
            "model": model_id,
            "status": "already_complete",
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
            "training_state": state
        }

    session_count = max(int(args.session_count), 1)
    completed_training = False

    for session_number in range(1, session_count + 1):
        if int(state.get("current_epoch", 0)) >= args.epochs or state.get("early_stopped"):
            completed_training = int(state.get("current_epoch", 0)) >= args.epochs
            break

        epoch_index = int(state.get("current_epoch", 0))
        if epoch_index >= args.freeze_backbone_epochs and state.get("backbone_frozen"):
            print(f"{model_id} descongelando backbone en epoca {epoch_index + 1}...", flush=True)
            set_backbone_trainable(model, True)
            optimizer = create_torchvision_optimizer(model, args)
            scheduler = create_torchvision_scheduler(optimizer, args)
            state["backbone_frozen"] = False
            state["optimizer_recreated_after_unfreeze"] = True

        model.train()
        start_index = int(state.get("next_sample_index", 0))
        epoch_order = shuffled_epoch_indices(len(samples), epoch_index)
        session_size = len(samples) - start_index if args.session_samples <= 0 else args.session_samples
        end_index = min(start_index + session_size, len(samples))
        segment_indices = epoch_order[start_index:end_index]
        segment_samples = [samples[index] for index in segment_indices]

        if not segment_samples:
            raise ValueError("No hay muestras en la seccion solicitada.")

        train_dataset = TorchvisionYoloDataset(segment_samples)
        data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=lambda batch: tuple(zip(*batch))
        )

        print(
            f"{model_id} session {session_number}/{session_count} "
            f"epoch {epoch_index + 1}/{args.epochs} "
            f"items {start_index + 1}-{end_index}/{len(samples)} "
            f"batches={len(data_loader)} device={device} lr={current_learning_rate(optimizer):.6f}",
            flush=True
        )

        epoch_loss_sum = float(state.get("epoch_loss_sum", 0.0))
        epoch_batches = int(state.get("epoch_batches", 0))
        processed_in_segment = 0
        start_time = time.time()

        write_training_progress(
            artifact_dir=artifact_dir,
            run_root=run_root,
            state=state,
            event={
                "event": "session_started",
                "model": model_id,
                "session": session_number,
                "session_count": session_count,
                "epoch": epoch_index + 1,
                "target_epochs": args.epochs,
                "start_index": start_index,
                "end_index": end_index,
                "total_samples": len(samples),
                "batch": args.batch,
                "image_size": args.imgsz,
                "device": str(device),
                "learning_rate": current_learning_rate(optimizer),
                "backbone_frozen": bool(state.get("backbone_frozen"))
            }
        )

        try:
            for batch_number, (images, targets) in enumerate(data_loader, start=1):
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                batch_loss = float(losses.detach().cpu())
                batch_size = len(images)
                processed_in_segment += batch_size
                epoch_loss_sum += batch_loss
                epoch_batches += 1

                state.update(
                    {
                        "status": "running",
                        "current_epoch": epoch_index,
                        "target_epochs": args.epochs,
                        "next_sample_index": min(start_index + processed_in_segment, len(samples)),
                        "total_samples": len(samples),
                        "epoch_loss_sum": epoch_loss_sum,
                        "epoch_batches": epoch_batches,
                        "last_loss": batch_loss,
                        "last_batch": batch_number,
                        "last_session": session_number,
                        "last_learning_rate": current_learning_rate(optimizer),
                        "last_update": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_seconds": round(time.time() - start_time, 2)
                    }
                )

                should_log = batch_number == 1 or batch_number == len(data_loader) or batch_number % max(args.log_every, 1) == 0
                should_checkpoint = (
                    batch_number == len(data_loader)
                    or (args.checkpoint_every > 0 and batch_number % args.checkpoint_every == 0)
                )

                if should_log:
                    progress_percent = (state["next_sample_index"] / max(len(samples), 1)) * 100
                    message = (
                        f"{model_id} session {session_number}/{session_count} "
                        f"epoch {epoch_index + 1}/{args.epochs} "
                        f"batch {batch_number}/{len(data_loader)} "
                        f"items {state['next_sample_index']}/{len(samples)} "
                        f"({progress_percent:.2f}%) loss={batch_loss:.4f}"
                    )
                    print(message, flush=True)
                    write_training_progress(
                        artifact_dir=artifact_dir,
                        run_root=run_root,
                        state=state,
                        event={
                            "event": "batch_progress",
                            "model": model_id,
                            "session": session_number,
                            "session_count": session_count,
                            "epoch": epoch_index + 1,
                            "batch": batch_number,
                            "total_batches": len(data_loader),
                            "next_sample_index": state["next_sample_index"],
                            "total_samples": len(samples),
                            "loss": batch_loss,
                            "progress_percent": round(progress_percent, 4),
                            "learning_rate": current_learning_rate(optimizer)
                        }
                    )

                if should_checkpoint:
                    save_torchvision_checkpoint(
                        checkpoint_path=checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        classes=classes,
                        state=state
                    )
        except Exception as exc:
            state.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "last_update": datetime.now().isoformat(timespec="seconds")
                }
            )
            write_training_progress(
                artifact_dir=artifact_dir,
                run_root=run_root,
                state=state,
                event={
                    "event": "failed",
                    "model": model_id,
                    "session": session_number,
                    "epoch": epoch_index + 1,
                    "next_sample_index": state.get("next_sample_index", start_index),
                    "error": str(exc)
                }
            )
            save_torchvision_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                classes=classes,
                state=state
            )
            raise

        completed_epoch = int(state["next_sample_index"]) >= len(samples)
        if completed_epoch:
            epoch_average_loss = epoch_loss_sum / max(epoch_batches, 1)
            loss_history.append(epoch_average_loss)
            epoch_index += 1
            state.update(
                {
                    "current_epoch": epoch_index,
                    "next_sample_index": 0,
                    "loss_history": loss_history,
                    "epoch_loss_sum": 0.0,
                    "epoch_batches": 0,
                    "last_epoch_loss": epoch_average_loss,
                    "last_update": datetime.now().isoformat(timespec="seconds")
                }
            )
            print(
                f"{model_id} epoch {epoch_index}/{args.epochs} completada "
                f"loss={epoch_average_loss:.4f}",
                flush=True
            )

            if args.validate_every_epoch and validation_samples:
                confidence_threshold, max_detections = resolve_validation_settings(model_id, args)
                print(
                    f"{model_id} validando epoca {epoch_index} "
                    f"conf={confidence_threshold} max_det={max_detections}...",
                    flush=True
                )
                validation_summary = evaluate_torchvision_model(
                    model=model,
                    samples=validation_samples,
                    classes=classes,
                    device=device,
                    confidence_threshold=confidence_threshold,
                    max_detections=max_detections,
                    visual_output_dir=(
                        run_root / "visuals" / model_id / f"epoch_{epoch_index:03d}"
                        if args.validation_visual_limit > 0
                        else None
                    ),
                    visual_limit=args.validation_visual_limit,
                )
                validation_record = build_validation_record(
                    epoch=epoch_index,
                    loss=epoch_average_loss,
                    summary=validation_summary,
                    best_metric=args.best_metric,
                )
                validation_history.append(validation_record)
                validation_history_path.write_text(json.dumps(validation_history, ensure_ascii=False, indent=2), encoding="utf-8")
                state["validation_history"] = validation_history
                state["last_validation"] = validation_record
                state["best_metric"] = args.best_metric

                best_value = state.get("best_metric_value")
                current_value = float(validation_record["metric_value"])
                improved = best_value is None or current_value > float(best_value) + args.min_delta
                if improved:
                    state["best_metric_value"] = current_value
                    state["best_epoch"] = epoch_index
                    state["best_checkpoint_path"] = relative_project_path(best_checkpoint_path)
                    state["best_artifact_path"] = relative_project_path(best_artifact_path)
                    state["epochs_without_improvement"] = 0
                    save_torchvision_checkpoint(
                        checkpoint_path=best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        classes=classes,
                        state=state
                    )
                    save_torchvision_artifact(
                        artifact_path=best_artifact_path,
                        model=model,
                        classes=classes,
                        state=state,
                        metrics={"validation": validation_record, "validation_history": validation_history},
                    )
                    print(f"{model_id} nuevo best {args.best_metric}={current_value:.6f} en epoca {epoch_index}.", flush=True)
                else:
                    state["epochs_without_improvement"] = int(state.get("epochs_without_improvement", 0)) + 1
                    print(
                        f"{model_id} sin mejora en {args.best_metric}; "
                        f"racha={state['epochs_without_improvement']}",
                        flush=True
                    )

                if args.early_stopping_patience > 0 and int(state.get("epochs_without_improvement", 0)) >= args.early_stopping_patience:
                    state["early_stopped"] = True
                    state["early_stopped_epoch"] = epoch_index
                    state["status"] = "early_stopped"

            if scheduler is not None:
                scheduler.step()
                state["last_learning_rate"] = current_learning_rate(optimizer)
        else:
            state["loss_history"] = loss_history

        completed_training = int(state["current_epoch"]) >= args.epochs
        if state.get("early_stopped"):
            state["status"] = "early_stopped"
        else:
            state["status"] = "completed" if completed_training else "partial"
        state["last_update"] = datetime.now().isoformat(timespec="seconds")

        save_torchvision_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            classes=classes,
            state=state
        )

        write_training_progress(
            artifact_dir=artifact_dir,
            run_root=run_root,
            state=state,
            event={
                "event": "session_completed",
                "model": model_id,
                "session": session_number,
                "session_count": session_count,
                "status": state["status"],
                "next_sample_index": state["next_sample_index"],
                "total_samples": len(samples),
                "checkpoint_path": relative_project_path(checkpoint_path),
                "best_checkpoint_path": state.get("best_checkpoint_path")
            }
        )

        if completed_training or state.get("early_stopped"):
            break

    metrics = {
        "loss_history": loss_history,
        "validation_history": validation_history,
        "training_state": state
    }
    metrics_path = run_root / f"{model_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    update_model_registry_training_progress(
        model_id,
        {
            "status": state["status"],
            "dataset_id": dataset["id"],
            "target_epochs": args.epochs,
            "current_epoch": state["current_epoch"],
            "next_sample_index": state["next_sample_index"],
            "total_samples": len(samples),
            "checkpoint_path": relative_project_path(checkpoint_path),
            "best_checkpoint_path": state.get("best_checkpoint_path"),
            "progress_path": relative_project_path(artifact_dir / "training_progress.json"),
            "events_path": relative_project_path(artifact_dir / "training_events.jsonl"),
            "validation_history_path": relative_project_path(validation_history_path) if validation_history_path.exists() else None,
            "run_dir": relative_project_path(run_root),
            "last_update": state["last_update"]
        }
    )

    artifact_path = best_artifact_path if best_artifact_path.exists() else latest_artifact_path
    if completed_training or args.publish_partial or state.get("early_stopped"):
        save_torchvision_artifact(
            artifact_path=latest_artifact_path,
            model=model,
            classes=classes,
            state=state,
            metrics=metrics,
        )
        update_model_registry(
            model_id,
            {
                "status": "trained" if (completed_training or state.get("early_stopped")) else state["status"],
                "artifact_path": relative_project_path(artifact_path),
                "latest_artifact_path": relative_project_path(latest_artifact_path),
                "best_artifact_path": relative_project_path(best_artifact_path) if best_artifact_path.exists() else None,
                "metrics": metrics,
                "labels": classes,
                "training": {
                    "dataset_id": dataset["id"],
                    "epochs": state["current_epoch"],
                    "target_epochs": args.epochs,
                    "batch": args.batch,
                    "image_size": args.imgsz,
                    "learning_rate": args.lr,
                    "momentum": args.momentum,
                    "weight_decay": args.weight_decay,
                    "lr_step_size": args.lr_step_size,
                    "lr_gamma": args.lr_gamma,
                    "freeze_backbone_epochs": args.freeze_backbone_epochs,
                    "best_metric": args.best_metric,
                    "best_metric_value": state.get("best_metric_value"),
                    "best_epoch": state.get("best_epoch"),
                    "checkpoint_path": relative_project_path(checkpoint_path),
                    "best_checkpoint_path": state.get("best_checkpoint_path"),
                    "validation_history_path": relative_project_path(validation_history_path) if validation_history_path.exists() else None,
                    "run_dir": relative_project_path(run_root),
                    "status": state["status"]
                }
            }
        )

    return {
        "model": model_id,
        "status": state["status"],
        "artifact_path": str(artifact_path) if artifact_path.exists() else None,
        "latest_artifact_path": str(latest_artifact_path) if latest_artifact_path.exists() else None,
        "best_artifact_path": str(best_artifact_path) if best_artifact_path.exists() else None,
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
        "metrics_path": str(metrics_path),
        "validation_history_path": str(validation_history_path) if validation_history_path.exists() else None,
        "progress_path": str(artifact_dir / "training_progress.json"),
        "events_path": str(artifact_dir / "training_events.jsonl"),
        "training_state": state
    }

def initial_training_state(model_id: str, dataset_id: str, total_samples: int) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "status": "new",
        "current_epoch": 0,
        "target_epochs": 0,
        "next_sample_index": 0,
        "total_samples": total_samples,
        "epoch_loss_sum": 0.0,
        "epoch_batches": 0,
        "loss_history": [],
        "last_loss": None,
        "last_batch": 0,
        "last_update": datetime.now().isoformat(timespec="seconds")
    }


def find_latest_ultralytics_epoch_checkpoint(model_id: str) -> Path | None:
    candidates: dict[Path, int] = {}
    artifact_dir = MODEL_ARTIFACTS_ROOT / model_id
    summary_path = artifact_dir / "training_summary.json"

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run_dir = Path(str(summary.get("run_dir", "")))
            collect_ultralytics_epoch_checkpoints(run_dir / "weights", candidates)
        except (OSError, json.JSONDecodeError):
            pass

    for weights_dir in RUNS_ROOT.glob(f"*/{model_id}/weights"):
        collect_ultralytics_epoch_checkpoints(weights_dir, candidates)

    if not candidates:
        return None

    return max(candidates, key=lambda path: (candidates[path], path.stat().st_mtime))


def collect_ultralytics_epoch_checkpoints(weights_dir: Path, candidates: dict[Path, int]) -> None:
    if not weights_dir.exists():
        return

    for checkpoint_path in weights_dir.glob("epoch*.pt"):
        epoch_text = checkpoint_path.stem.removeprefix("epoch")
        if not epoch_text.isdigit():
            continue
        candidates[checkpoint_path] = int(epoch_text)


def prepare_ultralytics_resume_checkpoint(
    checkpoint_path: Path,
    target_epochs: int,
    data_yaml_path: Path,
    model_id: str
) -> Path:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return checkpoint_path

    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, dict):
        return checkpoint_path

    current_target = int(train_args.get("epochs") or 0)
    if current_target >= target_epochs:
        return checkpoint_path

    continuation_dir = MODEL_ARTIFACTS_ROOT / model_id / "resume_checkpoints"
    continuation_dir.mkdir(parents=True, exist_ok=True)
    epoch_index = int(checkpoint.get("epoch", -1))
    continuation_path = continuation_dir / f"epoch{epoch_index}_resume_to_epoch{target_epochs}.pt"

    patched_args = dict(train_args)
    patched_args["epochs"] = target_epochs
    patched_args["data"] = str(data_yaml_path)
    checkpoint["train_args"] = patched_args
    torch.save(checkpoint, continuation_path)

    print(
        f"{model_id} preparo checkpoint de continuacion: {continuation_path} "
        f"(objetivo {current_target} -> {target_epochs} epocas)",
        flush=True
    )
    return continuation_path


def sync_training_state_with_dataset(state: dict[str, Any], total_samples: int) -> dict[str, Any]:
    previous_total = int(state.get("total_samples", total_samples) or total_samples)
    state["total_samples"] = total_samples
    if previous_total != total_samples:
        state["dataset_resplit"] = {
            "previous_total_samples": previous_total,
            "current_total_samples": total_samples,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        next_sample_index = int(state.get("next_sample_index", 0) or 0)
        if next_sample_index > total_samples:
            state["next_sample_index"] = 0
            state["epoch_loss_sum"] = 0.0
            state["epoch_batches"] = 0
    return state


def shuffled_epoch_indices(total_samples: int, epoch_index: int) -> list[int]:
    indices = list(range(total_samples))
    random.Random(20260823 + epoch_index).shuffle(indices)
    return indices


def save_torchvision_checkpoint(
    checkpoint_path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    classes: list[str],
    state: dict[str, Any]
) -> None:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": state["model_id"],
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "classes": classes,
        "training_state": state
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, checkpoint_path)


def save_torchvision_artifact(
    artifact_path: Path,
    model: Any,
    classes: list[str],
    state: dict[str, Any],
    metrics: dict[str, Any]
) -> None:
    import torch

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": state["model_id"],
            "state_dict": model.state_dict(),
            "classes": classes,
            "epochs": state.get("current_epoch", 0),
            "loss_history": state.get("loss_history", []),
            "validation_history": state.get("validation_history", []),
            "training_state": state,
            "metrics": metrics,
        },
        artifact_path
    )

def create_torchvision_optimizer(model: Any, args: argparse.Namespace) -> Any:
    import torch

    return torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def create_torchvision_scheduler(optimizer: Any, args: argparse.Namespace) -> Any | None:
    if args.lr_step_size <= 0:
        return None
    import torch

    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)


def current_learning_rate(optimizer: Any) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def set_backbone_trainable(model: Any, trainable: bool) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    for parameter in backbone.parameters():
        parameter.requires_grad = trainable


def resolve_validation_settings(model_id: str, args: argparse.Namespace) -> tuple[float, int]:
    confidence = args.validation_confidence
    max_detections = args.validation_max_detections
    if confidence >= 0 and max_detections > 0:
        return confidence, max_detections

    model_config = get_model_config(model_id)
    if confidence < 0:
        confidence = float(model_config.get("confidence_threshold") or 0.0)
    if max_detections <= 0:
        max_detections = int(model_config.get("max_detections") or model_config.get("detections_per_img") or 100)
    return confidence, max_detections


def get_model_config(model_id: str) -> dict[str, Any]:
    if not MODEL_REGISTRY_PATH.exists():
        return {}
    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    for model in registry.get("models", []):
        if model.get("id") == model_id:
            return model
    return {}


def build_validation_record(epoch: int, loss: float, summary: dict[str, Any], best_metric: str) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    metric_value = float(metrics.get(best_metric) or 0.0)
    return {
        "epoch": epoch,
        "loss": round(loss, 6),
        "best_metric": best_metric,
        "metric_value": round(metric_value, 6),
        "map50": metrics.get("map50"),
        "map50_95": metrics.get("map50_95"),
        "precision_at_50": metrics.get("precision_at_50"),
        "recall_at_50": metrics.get("recall_at_50"),
        "false_positives_at_50": metrics.get("false_positives_at_50"),
        "false_negatives_at_50": metrics.get("false_negatives_at_50"),
        "evaluated_images": summary.get("evaluated_images"),
        "total_predictions": summary.get("total_predictions"),
        "avg_processing_time_ms": summary.get("avg_processing_time_ms"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def bootstrap_best_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    classes: list[str],
    state: dict[str, Any],
    validation_samples: list[Any],
    validation_history: list[dict[str, Any]],
    validation_history_path: Path,
    best_checkpoint_path: Path,
    best_artifact_path: Path,
    model_id: str,
    args: argparse.Namespace,
    device: Any,
) -> None:
    if state.get("best_metric_value") is not None and best_checkpoint_path.exists() and best_artifact_path.exists():
        print(
            f"{model_id} best existente conservado "
            f"{state.get('best_metric')}={state.get('best_metric_value')}.",
            flush=True,
        )
        return

    from ml.training.torchvision_validation import evaluate_torchvision_model

    confidence_threshold, max_detections = resolve_validation_settings(model_id, args)
    baseline_epoch = int(state.get("current_epoch", 0))
    baseline_loss = float(state.get("last_epoch_loss") or state.get("last_loss") or 0.0)
    print(
        f"{model_id} validando checkpoint base antes de entrenar "
        f"conf={confidence_threshold} max_det={max_detections}...",
        flush=True,
    )
    model.eval()
    validation_summary = evaluate_torchvision_model(
        model=model,
        samples=validation_samples,
        classes=classes,
        device=device,
        confidence_threshold=confidence_threshold,
        max_detections=max_detections,
    )
    validation_record = build_validation_record(
        epoch=baseline_epoch,
        loss=baseline_loss,
        summary=validation_summary,
        best_metric=args.best_metric,
    )
    validation_record["baseline"] = True
    validation_history.append(validation_record)
    validation_history_path.write_text(json.dumps(validation_history, ensure_ascii=False, indent=2), encoding="utf-8")

    current_value = float(validation_record["metric_value"])
    state["validation_history"] = validation_history
    state["last_validation"] = validation_record
    state["best_metric"] = args.best_metric
    state["best_metric_value"] = current_value
    state["best_epoch"] = baseline_epoch
    state["best_checkpoint_path"] = relative_project_path(best_checkpoint_path)
    state["best_artifact_path"] = relative_project_path(best_artifact_path)
    state["epochs_without_improvement"] = 0

    save_torchvision_checkpoint(
        checkpoint_path=best_checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        classes=classes,
        state=state,
    )
    save_torchvision_artifact(
        artifact_path=best_artifact_path,
        model=model,
        classes=classes,
        state=state,
        metrics={"validation": validation_record, "validation_history": validation_history},
    )
    print(f"{model_id} checkpoint base guardado como best {args.best_metric}={current_value:.6f}.", flush=True)


def load_torch_checkpoint(checkpoint_path: Path, device: Any) -> dict[str, Any]:
    import torch

    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_resume_checkpoint(
    model_id: str,
    candidates: list[Path],
    device: Any,
) -> tuple[dict[str, Any] | None, Path | None]:
    for checkpoint_path in candidates:
        if not checkpoint_path.exists():
            continue

        checkpoint_size = checkpoint_path.stat().st_size
        if checkpoint_size < 1_000_000:
            print(
                f"{model_id} omitiendo checkpoint incompleto {checkpoint_path} "
                f"({checkpoint_size} bytes).",
                flush=True,
            )
            continue

        print(f"{model_id} cargando checkpoint {checkpoint_path}...", flush=True)
        try:
            return load_torch_checkpoint(checkpoint_path, device), checkpoint_path
        except Exception as exc:  # noqa: BLE001 - checkpoint corrupto: probar fallback.
            print(
                f"{model_id} no pudo cargar {checkpoint_path}: {exc}. "
                "Se intentara otro checkpoint si existe.",
                flush=True,
            )

    return None, None


def write_training_progress(
    artifact_dir: Path,
    run_root: Path,
    state: dict[str, Any],
    event: dict[str, Any]
) -> None:
    progress_path = artifact_dir / "training_progress.json"
    events_path = artifact_dir / "training_events.jsonl"
    run_progress_path = run_root / f"{state['model_id']}_progress.json"
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **event
    }

    progress_payload = {
        "state": state,
        "last_event": event
    }
    progress_path.write_text(json.dumps(progress_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_progress_path.write_text(json.dumps(progress_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with events_path.open("a", encoding="utf-8") as events_file:
        events_file.write(json.dumps(event, ensure_ascii=False) + "\n")


class TorchvisionYoloDataset:
    def __init__(self, samples: list[Any]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import torch
        from torchvision.transforms import functional as F

        sample = self.samples[index]
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"No se pudo abrir la imagen de entrenamiento: {sample.image_path}") from exc
        image_tensor = F.convert_image_dtype(F.pil_to_tensor(image), dtype=torch.float32)

        boxes = []
        labels = []
        for target in sample.targets:
            x_center = target.x_center * sample.width
            y_center = target.y_center * sample.height
            box_width = target.width * sample.width
            box_height = target.height * sample.height
            x_min = max(0.0, x_center - box_width / 2)
            y_min = max(0.0, y_center - box_height / 2)
            x_max = min(float(sample.width), x_center + box_width / 2)
            y_max = min(float(sample.height), y_center + box_height / 2)
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(target.class_id + 1)

        if not boxes:
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [1]

        return image_tensor, {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64)
        }


class EfficientDetYoloDataset:
    def __init__(self, samples: list[Any], image_size: int, training: bool):
        self.samples = samples
        self.image_size = image_size
        self.training = training
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.fill_color = tuple(int(round(255 * value)) for value in self.mean)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import numpy as np
        import torch

        sample = self.samples[index]
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"No se pudo abrir la imagen de entrenamiento: {sample.image_path}") from exc

        boxes = []
        labels = []
        for target in sample.targets:
            x_center = target.x_center * sample.width
            y_center = target.y_center * sample.height
            box_width = target.width * sample.width
            box_height = target.height * sample.height
            x_min = max(0.0, x_center - box_width / 2)
            y_min = max(0.0, y_center - box_height / 2)
            x_max = min(float(sample.width), x_center + box_width / 2)
            y_max = min(float(sample.height), y_center + box_height / 2)
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append([y_min, x_min, y_max, x_max])
            labels.append(target.class_id + 1)

        boxes_array = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        if self.training and boxes_array.size and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped = boxes_array.copy()
            flipped[:, 1] = sample.width - boxes_array[:, 3]
            flipped[:, 3] = sample.width - boxes_array[:, 1]
            boxes_array = flipped

        width, height = image.size
        scale = min(self.image_size / height, self.image_size / width)
        scaled_height = max(1, int(height * scale))
        scaled_width = max(1, int(width * scale))
        resized = image.resize((scaled_width, scaled_height), Image.BILINEAR)
        padded = Image.new("RGB", (self.image_size, self.image_size), color=self.fill_color)
        padded.paste(resized, (0, 0))

        if boxes_array.size:
            boxes_array[:, :4] *= scale
            boxes_array[:, [0, 2]] = boxes_array[:, [0, 2]].clip(0, scaled_height)
            boxes_array[:, [1, 3]] = boxes_array[:, [1, 3]].clip(0, scaled_width)
            valid = (boxes_array[:, 0] < boxes_array[:, 2]) & (boxes_array[:, 1] < boxes_array[:, 3])
            boxes_array = boxes_array[valid]
            labels_array = labels_array[valid]

        image_array = np.asarray(padded, dtype=np.float32) / 255.0
        image_array = (image_array - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        image_array = np.moveaxis(image_array, 2, 0)

        return torch.from_numpy(image_array), {
            "bbox": torch.tensor(boxes_array, dtype=torch.float32),
            "cls": torch.tensor(labels_array, dtype=torch.int64),
            "img_scale": torch.tensor(1.0 / scale, dtype=torch.float32),
            "img_size": torch.tensor((width, height), dtype=torch.float32),
        }


def efficientdet_collate(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[Any, dict[str, Any]]:
    import torch

    images = torch.stack([item[0] for item in batch])
    max_instances = min(100, max((len(item[1]["cls"]) for item in batch), default=0))
    max_instances = max(max_instances, 1)
    bbox = torch.zeros((len(batch), max_instances, 4), dtype=torch.float32)
    cls = torch.full((len(batch), max_instances), -1, dtype=torch.int64)
    img_scale = torch.stack([item[1]["img_scale"] for item in batch])
    img_size = torch.stack([item[1]["img_size"] for item in batch])

    for index, (_, target) in enumerate(batch):
        count = min(len(target["cls"]), max_instances)
        if count <= 0:
            continue
        bbox[index, :count] = target["bbox"][:count]
        cls[index, :count] = target["cls"][:count]

    return images, {
        "bbox": bbox,
        "cls": cls,
        "img_scale": img_scale,
        "img_size": img_size,
    }


def train_efficientdet_detector(
    args: argparse.Namespace,
    dataset: dict[str, Any],
    dataset_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    import torch
    from effdet import create_model
    from torch.utils.data import DataLoader

    classes = get_dataset_classes(dataset)
    train_split = dataset.get("splits", {}).get("train")
    if not train_split:
        raise ValueError("Dataset sin split train.")

    model_id = "custom-efficientdet-detector"
    print(f"efficientdet preparando indice del dataset {dataset['id']}...", flush=True)
    samples = build_yolo_index(dataset_root, train_split)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError("No hay muestras para entrenar.")

    validation_samples = []
    if args.validate_every_epoch:
        validation_split = dataset.get("splits", {}).get("val") or dataset.get("splits", {}).get("validation")
        if validation_split:
            from ml.training.torchvision_validation import select_validation_samples

            validation_samples = select_validation_samples(build_yolo_index(dataset_root, validation_split), args.validation_limit)
            print(f"efficientdet validacion configurada: {len(validation_samples)} muestras.", flush=True)
        else:
            print("efficientdet sin split de validacion; se omite validacion por epoca.", flush=True)

    device = torch.device("cuda:0" if resolve_device(args.device) != "cpu" and torch.cuda.is_available() else "cpu")
    artifact_dir = MODEL_ARTIFACTS_ROOT / model_id
    if args.max_samples > 0:
        artifact_dir = artifact_dir / "_debug"
    checkpoint_dir = artifact_dir / "checkpoints"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
    best_checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
    latest_artifact_path = artifact_dir / latest_artifact_filename(model_id)
    best_artifact_path = artifact_dir / final_artifact_filename(model_id)
    validation_history_path = artifact_dir / "validation_history.json"

    print(
        f"{model_id} construyendo {args.efficientdet_base_model} "
        f"pretrained_backbone={args.efficientdet_pretrained_backbone}...",
        flush=True,
    )
    model = create_model(
        args.efficientdet_base_model,
        bench_task="train",
        num_classes=len(classes),
        image_size=(args.imgsz, args.imgsz),
        pretrained=False,
        pretrained_backbone=bool(args.efficientdet_pretrained_backbone),
        bench_labeler=True,
        max_det_per_image=max(resolve_validation_settings(model_id, args)[1], 8),
    )
    model.to(device)

    state = initial_training_state(model_id=model_id, dataset_id=dataset["id"], total_samples=len(samples))
    checkpoint: dict[str, Any] | None = None
    if args.resume:
        checkpoint, loaded_checkpoint_path = load_resume_checkpoint(
            model_id=model_id,
            candidates=[checkpoint_path, best_checkpoint_path],
            device=device,
        )
        if checkpoint is None:
            raise RuntimeError(f"{model_id} no encontro un checkpoint valido para reanudar.")
    else:
        loaded_checkpoint_path = None

    if checkpoint:
        print(f"{model_id} aplicando pesos desde {loaded_checkpoint_path}...", flush=True)
        model.load_state_dict(checkpoint["state_dict"])
        state = checkpoint.get("training_state", state)
        state = sync_training_state_with_dataset(state, total_samples=len(samples))

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = create_torchvision_scheduler(optimizer, args)
    if checkpoint:
        try:
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and checkpoint.get("scheduler_state"):
                scheduler.load_state_dict(checkpoint["scheduler_state"])
        except ValueError as exc:
            state["optimizer_resume_warning"] = str(exc)
            print(f"{model_id} no pudo reusar el optimizador previo; se continua con uno nuevo: {exc}", flush=True)

    loss_history: list[float] = list(state.get("loss_history", []))
    validation_history: list[dict[str, Any]] = list(state.get("validation_history", []))

    if args.validate_before_training and args.validate_every_epoch and validation_samples:
        efficientdet_validate_and_update_best(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            classes=classes,
            state=state,
            validation_samples=validation_samples,
            validation_history=validation_history,
            validation_history_path=validation_history_path,
            best_checkpoint_path=best_checkpoint_path,
            best_artifact_path=best_artifact_path,
            model_id=model_id,
            args=args,
            device=device,
            epoch=int(state.get("current_epoch", 0)),
            loss=float(state.get("last_epoch_loss") or state.get("last_loss") or 0.0),
            run_root=run_root,
        )

    if int(state.get("current_epoch", 0)) >= args.epochs:
        return {
            "model": model_id,
            "status": "already_complete",
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
            "training_state": state,
        }

    session_count = max(int(args.session_count), 1)
    completed_training = False

    for session_number in range(1, session_count + 1):
        if int(state.get("current_epoch", 0)) >= args.epochs or state.get("early_stopped"):
            completed_training = int(state.get("current_epoch", 0)) >= args.epochs
            break

        model.train()
        epoch_index = int(state.get("current_epoch", 0))
        start_index = int(state.get("next_sample_index", 0))
        epoch_order = shuffled_epoch_indices(len(samples), epoch_index)
        session_size = len(samples) - start_index if args.session_samples <= 0 else args.session_samples
        end_index = min(start_index + session_size, len(samples))
        segment_indices = epoch_order[start_index:end_index]
        segment_samples = [samples[index] for index in segment_indices]
        if not segment_samples:
            raise ValueError("No hay muestras en la seccion solicitada.")

        data_loader = DataLoader(
            EfficientDetYoloDataset(segment_samples, image_size=args.imgsz, training=True),
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=efficientdet_collate,
        )

        print(
            f"{model_id} session {session_number}/{session_count} "
            f"epoch {epoch_index + 1}/{args.epochs} "
            f"items {start_index + 1}-{end_index}/{len(samples)} "
            f"batches={len(data_loader)} device={device} lr={current_learning_rate(optimizer):.6f}",
            flush=True,
        )

        epoch_loss_sum = float(state.get("epoch_loss_sum", 0.0))
        epoch_batches = int(state.get("epoch_batches", 0))
        processed_in_segment = 0
        start_time = time.time()
        write_training_progress(
            artifact_dir=artifact_dir,
            run_root=run_root,
            state=state,
            event={
                "event": "session_started",
                "model": model_id,
                "session": session_number,
                "session_count": session_count,
                "epoch": epoch_index + 1,
                "target_epochs": args.epochs,
                "start_index": start_index,
                "end_index": end_index,
                "total_samples": len(samples),
                "batch": args.batch,
                "image_size": args.imgsz,
                "device": str(device),
                "learning_rate": current_learning_rate(optimizer),
            },
        )

        try:
            for batch_number, (images, targets) in enumerate(data_loader, start=1):
                images = images.to(device)
                targets = {key: value.to(device) for key, value in targets.items()}

                output = model(images, targets)
                losses = output["loss"]
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                batch_loss = float(losses.detach().cpu())
                batch_size = len(images)
                processed_in_segment += batch_size
                epoch_loss_sum += batch_loss
                epoch_batches += 1
                state.update(
                    {
                        "status": "running",
                        "current_epoch": epoch_index,
                        "target_epochs": args.epochs,
                        "next_sample_index": min(start_index + processed_in_segment, len(samples)),
                        "total_samples": len(samples),
                        "epoch_loss_sum": epoch_loss_sum,
                        "epoch_batches": epoch_batches,
                        "last_loss": batch_loss,
                        "last_batch": batch_number,
                        "last_session": session_number,
                        "last_learning_rate": current_learning_rate(optimizer),
                        "last_update": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_seconds": round(time.time() - start_time, 2),
                    }
                )

                should_log = batch_number == 1 or batch_number == len(data_loader) or batch_number % max(args.log_every, 1) == 0
                should_checkpoint = (
                    batch_number == len(data_loader)
                    or (args.checkpoint_every > 0 and batch_number % args.checkpoint_every == 0)
                )
                if should_log:
                    progress_percent = (state["next_sample_index"] / max(len(samples), 1)) * 100
                    print(
                        f"{model_id} session {session_number}/{session_count} "
                        f"epoch {epoch_index + 1}/{args.epochs} "
                        f"batch {batch_number}/{len(data_loader)} "
                        f"items {state['next_sample_index']}/{len(samples)} "
                        f"({progress_percent:.2f}%) loss={batch_loss:.4f}",
                        flush=True,
                    )
                    write_training_progress(
                        artifact_dir=artifact_dir,
                        run_root=run_root,
                        state=state,
                        event={
                            "event": "batch_progress",
                            "model": model_id,
                            "session": session_number,
                            "session_count": session_count,
                            "epoch": epoch_index + 1,
                            "batch": batch_number,
                            "total_batches": len(data_loader),
                            "next_sample_index": state["next_sample_index"],
                            "total_samples": len(samples),
                            "loss": batch_loss,
                            "progress_percent": round(progress_percent, 4),
                            "learning_rate": current_learning_rate(optimizer),
                        },
                    )

                if should_checkpoint:
                    save_torchvision_checkpoint(
                        checkpoint_path=checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        classes=classes,
                        state=state,
                    )
        except Exception as exc:
            state.update(
                {
                    "status": "failed",
                    "error": str(exc) or repr(exc),
                    "last_update": datetime.now().isoformat(timespec="seconds"),
                }
            )
            write_training_progress(
                artifact_dir=artifact_dir,
                run_root=run_root,
                state=state,
                event={
                    "event": "failed",
                    "model": model_id,
                    "session": session_number,
                    "epoch": epoch_index + 1,
                    "next_sample_index": state.get("next_sample_index", start_index),
                    "error": str(exc) or repr(exc),
                },
            )
            save_torchvision_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                classes=classes,
                state=state,
            )
            raise

        completed_epoch = int(state["next_sample_index"]) >= len(samples)
        if completed_epoch:
            epoch_average_loss = epoch_loss_sum / max(epoch_batches, 1)
            loss_history.append(epoch_average_loss)
            epoch_index += 1
            state.update(
                {
                    "current_epoch": epoch_index,
                    "next_sample_index": 0,
                    "loss_history": loss_history,
                    "epoch_loss_sum": 0.0,
                    "epoch_batches": 0,
                    "last_epoch_loss": epoch_average_loss,
                    "last_update": datetime.now().isoformat(timespec="seconds"),
                }
            )
            print(f"{model_id} epoch {epoch_index}/{args.epochs} completada loss={epoch_average_loss:.4f}", flush=True)

            if args.validate_every_epoch and validation_samples:
                efficientdet_validate_and_update_best(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    classes=classes,
                    state=state,
                    validation_samples=validation_samples,
                    validation_history=validation_history,
                    validation_history_path=validation_history_path,
                    best_checkpoint_path=best_checkpoint_path,
                    best_artifact_path=best_artifact_path,
                    model_id=model_id,
                    args=args,
                    device=device,
                    epoch=epoch_index,
                    loss=epoch_average_loss,
                    run_root=run_root,
                )
                if args.early_stopping_patience > 0 and int(state.get("epochs_without_improvement", 0)) >= args.early_stopping_patience:
                    state["early_stopped"] = True
                    state["early_stopped_epoch"] = epoch_index
                    state["status"] = "early_stopped"

            if scheduler is not None:
                scheduler.step()
                state["last_learning_rate"] = current_learning_rate(optimizer)
        else:
            state["loss_history"] = loss_history

        completed_training = int(state["current_epoch"]) >= args.epochs
        if state.get("early_stopped"):
            state["status"] = "early_stopped"
        else:
            state["status"] = "completed" if completed_training else "partial"
        state["last_update"] = datetime.now().isoformat(timespec="seconds")

        save_torchvision_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            classes=classes,
            state=state,
        )
        write_training_progress(
            artifact_dir=artifact_dir,
            run_root=run_root,
            state=state,
            event={
                "event": "session_completed",
                "model": model_id,
                "session": session_number,
                "session_count": session_count,
                "status": state["status"],
                "next_sample_index": state["next_sample_index"],
                "total_samples": len(samples),
                "checkpoint_path": relative_project_path(checkpoint_path),
                "best_checkpoint_path": state.get("best_checkpoint_path"),
            },
        )

        if completed_training or state.get("early_stopped"):
            break

    metrics = {
        "loss_history": loss_history,
        "validation_history": validation_history,
        "training_state": state,
    }
    metrics_path = run_root / f"{model_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    update_model_registry_training_progress(
        model_id,
        {
            "status": state["status"],
            "dataset_id": dataset["id"],
            "target_epochs": args.epochs,
            "current_epoch": state["current_epoch"],
            "next_sample_index": state["next_sample_index"],
            "total_samples": len(samples),
            "checkpoint_path": relative_project_path(checkpoint_path),
            "best_checkpoint_path": state.get("best_checkpoint_path"),
            "progress_path": relative_project_path(artifact_dir / "training_progress.json"),
            "events_path": relative_project_path(artifact_dir / "training_events.jsonl"),
            "validation_history_path": relative_project_path(validation_history_path) if validation_history_path.exists() else None,
            "run_dir": relative_project_path(run_root),
            "last_update": state["last_update"],
        },
    )

    artifact_path = best_artifact_path if best_artifact_path.exists() else latest_artifact_path
    publish_training_artifact = args.max_samples <= 0 and (completed_training or args.publish_partial or state.get("early_stopped"))
    if publish_training_artifact:
        save_torchvision_artifact(
            artifact_path=latest_artifact_path,
            model=model,
            classes=classes,
            state=state,
            metrics=metrics,
        )
        update_model_registry(
            model_id,
            {
                "status": "trained" if (completed_training or state.get("early_stopped")) else state["status"],
                "artifact_path": relative_project_path(artifact_path),
                "latest_artifact_path": relative_project_path(latest_artifact_path),
                "best_artifact_path": relative_project_path(best_artifact_path) if best_artifact_path.exists() else None,
                "metrics": metrics,
                "labels": classes,
                "training": {
                    "dataset_id": dataset["id"],
                    "epochs": state["current_epoch"],
                    "target_epochs": args.epochs,
                    "batch": args.batch,
                    "image_size": args.imgsz,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "lr_step_size": args.lr_step_size,
                    "lr_gamma": args.lr_gamma,
                    "base_model": args.efficientdet_base_model,
                    "pretrained_backbone": bool(args.efficientdet_pretrained_backbone),
                    "best_metric": args.best_metric,
                    "best_metric_value": state.get("best_metric_value"),
                    "best_epoch": state.get("best_epoch"),
                    "checkpoint_path": relative_project_path(checkpoint_path),
                    "best_checkpoint_path": state.get("best_checkpoint_path"),
                    "validation_history_path": relative_project_path(validation_history_path) if validation_history_path.exists() else None,
                    "run_dir": relative_project_path(run_root),
                    "status": state["status"],
                },
            },
        )

    status = state["status"]
    if args.max_samples > 0:
        status = f"debug_{status}"

    return {
        "model": model_id,
        "status": status,
        "artifact_path": str(artifact_path) if artifact_path.exists() else None,
        "latest_artifact_path": str(latest_artifact_path) if latest_artifact_path.exists() else None,
        "best_artifact_path": str(best_artifact_path) if best_artifact_path.exists() else None,
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
        "metrics_path": str(metrics_path),
        "validation_history_path": str(validation_history_path) if validation_history_path.exists() else None,
        "progress_path": str(artifact_dir / "training_progress.json"),
        "events_path": str(artifact_dir / "training_events.jsonl"),
        "training_state": state,
    }


def efficientdet_validate_and_update_best(
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    classes: list[str],
    state: dict[str, Any],
    validation_samples: list[Any],
    validation_history: list[dict[str, Any]],
    validation_history_path: Path,
    best_checkpoint_path: Path,
    best_artifact_path: Path,
    model_id: str,
    args: argparse.Namespace,
    device: Any,
    epoch: int,
    loss: float,
    run_root: Path,
) -> None:
    confidence_threshold, max_detections = resolve_validation_settings(model_id, args)
    print(f"{model_id} validando epoca {epoch} conf={confidence_threshold} max_det={max_detections}...", flush=True)
    validation_summary = evaluate_efficientdet_model(
        model=model,
        samples=validation_samples,
        classes=classes,
        device=device,
        image_size=args.imgsz,
        confidence_threshold=confidence_threshold,
        max_detections=max_detections,
        visual_output_dir=(
            run_root / "visuals" / model_id / f"epoch_{epoch:03d}"
            if args.validation_visual_limit > 0
            else None
        ),
        visual_limit=args.validation_visual_limit,
    )
    validation_record = build_validation_record(
        epoch=epoch,
        loss=loss,
        summary=validation_summary,
        best_metric=args.best_metric,
    )
    validation_history.append(validation_record)
    validation_history_path.write_text(json.dumps(validation_history, ensure_ascii=False, indent=2), encoding="utf-8")
    state["validation_history"] = validation_history
    state["last_validation"] = validation_record
    state["best_metric"] = args.best_metric

    best_value = state.get("best_metric_value")
    current_value = float(validation_record["metric_value"])
    improved = best_value is None or current_value > float(best_value) + args.min_delta
    if improved:
        state["best_metric_value"] = current_value
        state["best_epoch"] = epoch
        state["epochs_without_improvement"] = 0
        if args.max_samples > 0:
            state["debug_best_not_published"] = True
            print(
                f"{model_id} debug best {args.best_metric}={current_value:.6f} en epoca {epoch}; "
                "no se publica por usar --max-samples.",
                flush=True,
            )
        else:
            state["best_checkpoint_path"] = relative_project_path(best_checkpoint_path)
            state["best_artifact_path"] = relative_project_path(best_artifact_path)
            save_torchvision_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                classes=classes,
                state=state,
            )
            save_torchvision_artifact(
                artifact_path=best_artifact_path,
                model=model,
                classes=classes,
                state=state,
                metrics={"validation": validation_record, "validation_history": validation_history},
            )
            print(f"{model_id} nuevo best {args.best_metric}={current_value:.6f} en epoca {epoch}.", flush=True)
    else:
        state["epochs_without_improvement"] = int(state.get("epochs_without_improvement", 0)) + 1
        print(f"{model_id} sin mejora en {args.best_metric}; racha={state['epochs_without_improvement']}", flush=True)


def evaluate_efficientdet_model(
    model: Any,
    samples: list[Any],
    classes: list[str],
    device: Any,
    image_size: int,
    confidence_threshold: float,
    max_detections: int,
    visual_output_dir: Path | None = None,
    visual_limit: int = 0,
) -> dict[str, Any]:
    import torch
    from ml.training.torchvision_validation import (
        compute_metrics,
        save_visual_evaluation_image,
        targets_to_ground_truth,
    )

    model.eval()
    predictions_by_image: dict[str, list[Any]] = {}
    ground_truth_by_image = {}
    timings: list[float] = []
    failures: list[dict[str, str]] = []
    validation_dataset = EfficientDetYoloDataset(samples, image_size=image_size, training=False)

    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            image_key = sample.image_path.name
            ground_truth = targets_to_ground_truth(sample, classes)
            ground_truth_by_image[image_key] = ground_truth
            try:
                image_tensor, target = validation_dataset[index - 1]
                images, targets = efficientdet_collate([(image_tensor, target)])
                images = images.to(device)
                targets = {key: value.to(device) for key, value in targets.items()}
                started = time.perf_counter()
                output = model(images, targets)
                timings.append((time.perf_counter() - started) * 1000)
                detections = output.get("detections")
                predictions = efficientdet_detections_to_predictions(
                    image_key=image_key,
                    detections=detections[0] if detections is not None else [],
                    classes=classes,
                    confidence_threshold=confidence_threshold,
                    max_detections=max_detections,
                )
                predictions_by_image[image_key] = predictions
                if visual_output_dir and index <= visual_limit:
                    image = Image.open(sample.image_path).convert("RGB")
                    save_visual_evaluation_image(
                        image=image,
                        output_path=visual_output_dir / f"{index:04d}_{sample.image_path.stem}.jpg",
                        predictions=predictions,
                        ground_truth=ground_truth,
                    )
            except Exception as exc:
                failures.append({"image": image_key, "error": str(exc)})
                predictions_by_image[image_key] = []

            if index % 25 == 0 or index == len(samples):
                print(f"validacion efficientdet: {index}/{len(samples)} imagenes", flush=True)

    metrics = compute_metrics(ground_truth_by_image, predictions_by_image, classes)
    model.train()
    return {
        "evaluated_images": len(samples),
        "ok": len(samples) - len(failures),
        "failures": failures,
        "images_with_detections": sum(1 for values in predictions_by_image.values() if values),
        "total_predictions": sum(len(values) for values in predictions_by_image.values()),
        "total_ground_truth": sum(len(values) for values in ground_truth_by_image.values()),
        "avg_predictions_per_image": round(
            sum(len(values) for values in predictions_by_image.values()) / max(1, len(samples)),
            4,
        ),
        "avg_processing_time_ms": round(sum(timings) / max(1, len(timings)), 2),
        "visual_output_dir": str(visual_output_dir) if visual_output_dir else None,
        "visual_outputs": min(max(visual_limit, 0), len(samples)) if visual_output_dir else 0,
        "metrics": metrics,
    }


def efficientdet_detections_to_predictions(
    image_key: str,
    detections: Any,
    classes: list[str],
    confidence_threshold: float,
    max_detections: int,
) -> list[Any]:
    from ml.training.torchvision_validation import Prediction

    predictions = []
    for detection in detections:
        values = detection.detach().cpu().tolist() if hasattr(detection, "detach") else list(detection)
        if len(values) < 6:
            continue
        x1, y1, x2, y2, score, class_id = values[:6]
        confidence = float(score)
        if confidence < confidence_threshold:
            continue
        label_index = int(class_id) - 1
        if not (0 <= label_index < len(classes)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        predictions.append(
            Prediction(
                image_key=image_key,
                label=classes[label_index],
                confidence=confidence,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
            )
        )

    predictions.sort(key=lambda item: item.confidence, reverse=True)
    if max_detections > 0:
        return predictions[:max_detections]
    return predictions


def update_model_registry(model_id: str, updates: dict[str, Any]) -> None:
    if not MODEL_REGISTRY_PATH.exists():
        return

    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    models = registry.get("models", [])
    for model in models:
        if model.get("id") != model_id:
            continue
        model.update(updates)
        MODEL_REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return


def update_model_registry_training_progress(model_id: str, training_updates: dict[str, Any]) -> None:
    if not MODEL_REGISTRY_PATH.exists():
        return

    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    models = registry.get("models", [])
    for model in models:
        if model.get("id") != model_id:
            continue
        training = model.get("training")
        if not isinstance(training, dict):
            training = {}
        training.update(training_updates)
        model["training"] = training
        MODEL_REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return


def relative_project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()

