from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ml.training.dataset_catalog import resolve_project_path
from ml.training.local_train import (
    find_latest_ultralytics_epoch_checkpoint,
    final_artifact_filename,
    get_dataset_classes,
    get_required_dataset,
    latest_artifact_filename,
    prepare_ultralytics_resume_checkpoint,
    relative_project_path,
    resolve_device,
    summarize_ultralytics_results,
    update_model_registry,
    write_yolo_data_yaml,
)
from ml.training.paths import MODEL_ARTIFACTS_ROOT, RUNS_ROOT


MODEL_ID = "custom-detr-rtdetr-detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena RT-DETR localmente con dataset YOLO.")
    parser.add_argument("--dataset-id", default="voc-detect")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--base-weights", default="rtdetr-l.pt")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = get_required_dataset(args.dataset_id)
    dataset_root = resolve_project_path(dataset.get("dataset_path"))
    if not dataset_root or not dataset_root.exists():
        raise SystemExit(f"Dataset no disponible localmente: {dataset.get('dataset_path')}")

    from ultralytics import RTDETR

    artifact_dir = MODEL_ARTIFACTS_ROOT / MODEL_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_best_path = artifact_dir / final_artifact_filename(MODEL_ID)
    artifact_last_path = artifact_dir / latest_artifact_filename(MODEL_ID)

    run_root = RUNS_ROOT / f"rtdetr-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    data_yaml_path = write_yolo_data_yaml(dataset=dataset, dataset_root=dataset_root, run_root=run_root)

    resume_checkpoint = find_latest_ultralytics_epoch_checkpoint(MODEL_ID) or artifact_last_path
    if args.resume and resume_checkpoint.exists():
        resume_checkpoint = prepare_ultralytics_resume_checkpoint(
            checkpoint_path=resume_checkpoint,
            target_epochs=args.epochs,
            data_yaml_path=data_yaml_path,
            model_id=MODEL_ID
        )

    if args.resume and resume_checkpoint.exists():
        model = RTDETR(str(resume_checkpoint))
        resume_training = True
        print(f"{MODEL_ID} reanudando desde {resume_checkpoint}", flush=True)
    else:
        model = RTDETR(args.base_weights)
        resume_training = False
        if args.resume:
            print(f"{MODEL_ID} no encontro artefacto latest; iniciando desde {args.base_weights}", flush=True)

    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": resolve_device(args.device),
        "workers": args.workers,
        "project": str(run_root),
        "name": MODEL_ID,
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

    update_model_registry(
        MODEL_ID,
        {
            "status": "ready" if artifact_best_path.exists() else "planned",
            "artifact_path": relative_project_path(artifact_best_path),
            "latest_artifact_path": relative_project_path(artifact_last_path),
            "best_artifact_path": relative_project_path(artifact_best_path),
            "metrics": metrics_summary,
            "confidence_threshold": 0.45,
            "iou_threshold": 0.55,
            "max_detections": 8,
            "labels": get_dataset_classes(dataset),
            "training": {
                "dataset_id": args.dataset_id,
                "epochs": args.epochs,
                "target_epochs": args.epochs,
                "batch": args.batch,
                "image_size": args.imgsz,
                "base_weights": args.base_weights,
                "run_dir": relative_project_path(run_dir),
                "training_record": "docs/DETR_RTDETR.md",
                "confidence_threshold": 0.45,
                "iou_threshold": 0.55,
                "max_detections": 8,
                "status": "calibrated",
            },
        },
    )

    summary = {
        "model": MODEL_ID,
        "status": "trained",
        "run_dir": str(run_dir),
        "artifact_best": str(artifact_best_path),
        "artifact_last": str(artifact_last_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics_summary,
    }
    summary_path = artifact_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
