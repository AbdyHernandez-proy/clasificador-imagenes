from __future__ import annotations

import argparse
import json
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
from ml.training.datasets.yolo_detection import build_yolo_index
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
    parser.add_argument("--publish-partial", action="store_true", help="Publica model.pt aunque el objetivo de epocas no haya terminado.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--allow-efficientdet", action="store_true", help="Reserva para trainer experimental futuro.")
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
                result = efficientdet_pending(args.allow_efficientdet)
            else:
                result = {"model": model_key, "status": "skipped", "reason": "Modelo no reconocido."}
        except Exception as exc:  # noqa: BLE001 - script local: registrar y seguir con el siguiente modelo.
            result = {"model": model_key, "status": "failed", "error": str(exc)}

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
    config = {
        "path": str(dataset_root),
        "train": dataset.get("splits", {}).get("train"),
        "val": dataset.get("splits", {}).get("val") or dataset.get("splits", {}).get("train"),
        "nc": len(classes),
        "names": classes
    }
    yaml_path = run_root / f"{dataset['id']}.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


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
        name="custom-yolo-v8-v11-detector",
        exist_ok=True
    )
    artifact_dir = MODEL_ARTIFACTS_ROOT / "custom-yolo-v8-v11-detector"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(result.save_dir)
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    artifact_best_path = artifact_dir / "best.pt"
    artifact_last_path = artifact_dir / "last.pt"

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
        "model": "custom-yolo-v8-v11-detector",
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

    model = RTDETR("rtdetr-l.pt")
    result = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=resolve_device(args.device),
        workers=args.workers,
        project=str(run_root),
        name="custom-detr-rtdetr-detector",
        exist_ok=True
    )
    return {
        "model": "custom-detr-rtdetr-detector",
        "status": "trained",
        "run_dir": str(result.save_dir),
        "expected_best": str(Path(result.save_dir) / "weights" / "best.pt")
    }


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

    print(f"{model_id} moviendo modelo a {device}...", flush=True)
    model.to(device)
    model.train()
    print(f"{model_id} preparando optimizador...", flush=True)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.0025,
        momentum=0.9,
        weight_decay=0.0005
    )

    artifact_dir = MODEL_ARTIFACTS_ROOT / model_id
    checkpoint_dir = artifact_dir / "checkpoints"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"

    state = initial_training_state(model_id=model_id, dataset_id=dataset["id"], total_samples=len(samples))
    if args.resume and checkpoint_path.exists():
        print(f"{model_id} cargando checkpoint {checkpoint_path}...", flush=True)
        checkpoint = load_torch_checkpoint(checkpoint_path, device)
        print(f"{model_id} aplicando pesos y estado del optimizador...", flush=True)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        state = checkpoint.get("training_state", state)
        print(f"{model_id} reanudado desde {checkpoint_path}", flush=True)

    if int(state.get("current_epoch", 0)) >= args.epochs:
        return {
            "model": model_id,
            "status": "already_complete",
            "checkpoint_path": str(checkpoint_path),
            "training_state": state
        }

    loss_history: list[float] = list(state.get("loss_history", []))
    session_count = max(int(args.session_count), 1)
    completed_training = False

    for session_number in range(1, session_count + 1):
        if int(state.get("current_epoch", 0)) >= args.epochs:
            completed_training = True
            break

        epoch_index = int(state.get("current_epoch", 0))
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
            f"batches={len(data_loader)} device={device}",
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
                "device": str(device)
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
                            "progress_percent": round(progress_percent, 4)
                        }
                    )

                if should_checkpoint:
                    save_torchvision_checkpoint(
                        checkpoint_path=checkpoint_path,
                        model=model,
                        optimizer=optimizer,
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
        else:
            state["loss_history"] = loss_history

        completed_training = int(state["current_epoch"]) >= args.epochs
        state["status"] = "completed" if completed_training else "partial"
        state["last_update"] = datetime.now().isoformat(timespec="seconds")

        save_torchvision_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
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
                "checkpoint_path": relative_project_path(checkpoint_path)
            }
        )

        if completed_training:
            break

    metrics = {
        "loss_history": loss_history,
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
            "progress_path": relative_project_path(artifact_dir / "training_progress.json"),
            "events_path": relative_project_path(artifact_dir / "training_events.jsonl"),
            "run_dir": relative_project_path(run_root),
            "last_update": state["last_update"]
        }
    )

    artifact_path = artifact_dir / "model.pt"
    if completed_training or args.publish_partial:
        torch.save(
            {
                "model_id": model_id,
                "state_dict": model.state_dict(),
                "classes": classes,
                "epochs": state["current_epoch"],
                "loss_history": loss_history,
                "training_state": state
            },
            artifact_path
        )
        update_model_registry(
            model_id,
            {
                "status": "trained" if completed_training else "partial",
                "artifact_path": relative_project_path(artifact_path),
                "metrics": metrics,
                "labels": classes,
                "training": {
                    "dataset_id": dataset["id"],
                    "epochs": state["current_epoch"],
                    "target_epochs": args.epochs,
                    "batch": args.batch,
                    "image_size": args.imgsz,
                    "checkpoint_path": relative_project_path(checkpoint_path),
                    "run_dir": relative_project_path(run_root),
                    "status": state["status"]
                }
            }
        )

    return {
        "model": model_id,
        "status": state["status"],
        "artifact_path": str(artifact_path) if artifact_path.exists() else None,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
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


def shuffled_epoch_indices(total_samples: int, epoch_index: int) -> list[int]:
    indices = list(range(total_samples))
    random.Random(20260823 + epoch_index).shuffle(indices)
    return indices


def save_torchvision_checkpoint(
    checkpoint_path: Path,
    model: Any,
    optimizer: Any,
    classes: list[str],
    state: dict[str, Any]
) -> None:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": state["model_id"],
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "classes": classes,
            "training_state": state
        },
        checkpoint_path
    )


def load_torch_checkpoint(checkpoint_path: Path, device: Any) -> dict[str, Any]:
    import torch

    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


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


def efficientdet_pending(allow_experimental: bool) -> dict[str, Any]:
    if allow_experimental:
        return {
            "model": "custom-efficientdet-detector",
            "status": "skipped",
            "reason": "El trainer EfficientDet aun no esta implementado. La configuracion y dependencias estan preparadas."
        }

    return {
        "model": "custom-efficientdet-detector",
        "status": "skipped",
        "reason": "EfficientDet queda pendiente para evitar un entrenamiento incorrecto. Usa YOLO/Faster R-CNN/RetinaNet/RT-DETR primero."
    }


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
