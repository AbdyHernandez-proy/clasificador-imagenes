from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for candidate in (PROJECT_ROOT, BACKEND_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.services.model_registry import ModelRegistry  # noqa: E402
from ml.training.dataset_catalog import load_dataset_registry  # noqa: E402
from ml.training.datasets.yolo_detection import DetectionSample, build_yolo_index  # noqa: E402
from ml.training.torchvision_validation import (  # noqa: E402
    Prediction,
    compute_metrics,
    save_visual_evaluation_image,
    targets_to_ground_truth,
)


@dataclass(frozen=True)
class ThresholdCombo:
    confidence: float
    iou: float
    max_detections: int
    label: str

    @property
    def key(self) -> str:
        conf_key = int(round(self.confidence * 100))
        iou_key = int(round(self.iou * 100))
        return f"{self.label}_conf_{conf_key:03d}_iou_{iou_key:03d}_max_{self.max_detections}"


DEFAULT_COMBOS = [
    ThresholdCombo(0.35, 0.55, 8, "actual"),
    ThresholdCombo(0.25, 0.55, 8, "sensible"),
    ThresholdCombo(0.30, 0.55, 8, "sensible"),
    ThresholdCombo(0.40, 0.55, 8, "balanceado"),
    ThresholdCombo(0.45, 0.55, 8, "conservador"),
    ThresholdCombo(0.50, 0.55, 8, "conservador"),
    ThresholdCombo(0.35, 0.50, 8, "iou_bajo"),
    ThresholdCombo(0.35, 0.60, 8, "iou_alto"),
    ThresholdCombo(0.40, 0.60, 6, "cajas_reducidas"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibra umbrales de modelos Ultralytics/RT-DETR.")
    parser.add_argument("--model", default="custom-detr-rtdetr-detector")
    parser.add_argument("--dataset", default="voc-detect")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--imgsz", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--visual-limit", type=int, default=12)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--combos",
        default="",
        help="Formato: conf,iou,max,label;conf,iou,max,label. Si se omite usa la matriz recomendada.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir) if args.output_dir else default_output_dir(args.model)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(args.dataset)
    dataset_root = resolve_project_path(dataset["dataset_path"])
    split_path = dataset["splits"][args.split]
    classes = list(dataset.get("classes") or [])
    all_samples = build_yolo_index(dataset_root=dataset_root, image_split=split_path)
    samples = select_samples(all_samples, args.limit)

    registry = ModelRegistry()
    model_config = registry.get_model(args.model)
    if not model_config:
        raise SystemExit(f"No existe el modelo en registro: {args.model}")

    artifact_path = resolve_project_path(str(model_config["artifact_path"]))
    if not artifact_path.exists():
        raise SystemExit(f"No existe el artefacto del modelo: {artifact_path}")

    image_size = args.imgsz or int(model_config.get("training", {}).get("image_size") or 512)
    baseline = ThresholdCombo(
        confidence=float(model_config.get("confidence_threshold") or 0.25),
        iou=float(model_config.get("iou_threshold") or model_config.get("nms_threshold") or 0.55),
        max_detections=int(model_config.get("max_detections") or 8),
        label="actual_registro",
    )
    combos = parse_combos(args.combos) if args.combos else [baseline, *DEFAULT_COMBOS[1:]]

    from ultralytics import RTDETR

    model = RTDETR(str(artifact_path))
    device = resolve_device(args.device)
    raw_results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for combo in combos:
        print(
            f"Calibrando {args.model}: conf={combo.confidence:.2f}, "
            f"iou={combo.iou:.2f}, max={combo.max_detections} ({combo.label})",
            flush=True,
        )
        combo_dir = output_root / combo.key
        combo_dir.mkdir(parents=True, exist_ok=True)
        result = evaluate_combo(
            model=model,
            combo=combo,
            samples=samples,
            classes=classes,
            image_size=image_size,
            device=device,
            output_dir=combo_dir,
            visual_limit=args.visual_limit,
        )
        raw_results.extend(result["raw_results"])
        summaries.append(result["summary"])

    summaries.sort(key=rank_summary, reverse=True)
    recommendation = summaries[0] if summaries else None
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "artifact_path": str(artifact_path),
        "dataset": args.dataset,
        "split": args.split,
        "total_split_images": len(all_samples),
        "evaluated_images": len(samples),
        "image_size": image_size,
        "current_registry_thresholds": {
            "confidence_threshold": baseline.confidence,
            "iou_threshold": baseline.iou,
            "max_detections": baseline.max_detections,
        },
        "recommendation": recommendation,
        "results": summaries,
    }
    (output_root / "threshold_calibration_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "threshold_calibration_raw_results.json").write_text(
        json.dumps(raw_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(output_root, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def evaluate_combo(
    model: Any,
    combo: ThresholdCombo,
    samples: list[DetectionSample],
    classes: list[str],
    image_size: int,
    device: str,
    output_dir: Path,
    visual_limit: int,
) -> dict[str, Any]:
    predictions_by_image: dict[str, list[Prediction]] = {}
    ground_truth_by_image = {}
    raw_results: list[dict[str, Any]] = []
    timings: list[float] = []
    failures: list[dict[str, str]] = []
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples, start=1):
        image_key = sample.image_path.name
        ground_truth_by_image[image_key] = targets_to_ground_truth(sample, classes)
        started = time.perf_counter()
        try:
            result = model.predict(
                source=str(sample.image_path),
                imgsz=image_size,
                conf=combo.confidence,
                iou=combo.iou,
                max_det=combo.max_detections,
                device=device,
                verbose=False,
            )[0]
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings.append(elapsed_ms)
            predictions = normalize_ultralytics_result(image_key, result)
            predictions_by_image[image_key] = predictions
            if index <= visual_limit:
                image = Image.open(sample.image_path).convert("RGB")
                save_visual_evaluation_image(
                    image=image,
                    output_path=annotated_dir / f"{index:03d}_{sample.image_path.stem}_{combo.key}.jpg",
                    predictions=predictions,
                    ground_truth=ground_truth_by_image[image_key],
                )
            raw_results.append({
                "combo": combo.key,
                "image": image_key,
                "ok": True,
                "processing_time_ms": round(elapsed_ms, 2),
                "prediction_count": len(predictions),
                "ground_truth_count": len(ground_truth_by_image[image_key]),
                "top_predictions": [prediction_to_dict(item) for item in predictions[:10]],
            })
        except Exception as exc:  # noqa: BLE001 - se registra y se continua calibrando.
            failures.append({"image": image_key, "error": str(exc)})
            predictions_by_image[image_key] = []
            raw_results.append({
                "combo": combo.key,
                "image": image_key,
                "ok": False,
                "error": str(exc),
                "prediction_count": 0,
                "ground_truth_count": len(ground_truth_by_image[image_key]),
            })

        if index % 25 == 0 or index == len(samples):
            print(f"{combo.key}: {index}/{len(samples)} imagenes", flush=True)

    metrics = compute_metrics(ground_truth_by_image, predictions_by_image, classes)
    total_predictions = sum(len(values) for values in predictions_by_image.values())
    summary = {
        "key": combo.key,
        "label": combo.label,
        "confidence_threshold": combo.confidence,
        "iou_threshold": combo.iou,
        "max_detections": combo.max_detections,
        "evaluated_images": len(samples),
        "ok": len(samples) - len(failures),
        "failures": failures,
        "images_with_detections": sum(1 for values in predictions_by_image.values() if values),
        "total_predictions": total_predictions,
        "avg_predictions_per_image": round(total_predictions / max(1, len(samples)), 4),
        "avg_processing_time_ms": round(sum(timings) / max(1, len(timings)), 2),
        "metrics": metrics,
        "score": score_metrics(metrics, total_predictions, len(samples)),
        "annotated_dir": str(annotated_dir),
    }
    return {"summary": summary, "raw_results": raw_results}


def normalize_ultralytics_result(image_key: str, result: Any) -> list[Prediction]:
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", None) or {}
    predictions: list[Prediction] = []
    if boxes is None:
        return predictions

    for box in boxes:
        xyxy = box.xyxy[0].detach().cpu().tolist()
        confidence = float(box.conf[0].detach().cpu())
        class_id = int(box.cls[0].detach().cpu())
        label = str(names.get(class_id, f"clase {class_id}"))
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        if x2 <= x1 or y2 <= y1:
            continue
        predictions.append(Prediction(image_key=image_key, label=label, confidence=confidence, bbox=(x1, y1, x2, y2)))

    return sorted(predictions, key=lambda item: item.confidence, reverse=True)


def get_dataset(dataset_id: str) -> dict[str, Any]:
    registry = load_dataset_registry()
    for dataset in registry.get("datasets", []):
        if dataset.get("id") == dataset_id:
            return dataset
    raise KeyError(f"No existe el dataset {dataset_id}")


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def select_samples(samples: list[DetectionSample], limit: int) -> list[DetectionSample]:
    if limit <= 0 or limit >= len(samples):
        return samples
    step = len(samples) / limit
    selected = [samples[min(len(samples) - 1, math.floor(index * step))] for index in range(limit)]
    deduped: list[DetectionSample] = []
    seen: set[Path] = set()
    for sample in selected:
        if sample.image_path in seen:
            continue
        seen.add(sample.image_path)
        deduped.append(sample)
    return deduped


def parse_combos(raw: str) -> list[ThresholdCombo]:
    combos: list[ThresholdCombo] = []
    for index, item in enumerate(raw.split(";"), start=1):
        if not item.strip():
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) not in {3, 4}:
            raise ValueError(f"Combo invalido: {item}")
        combos.append(
            ThresholdCombo(
                confidence=float(parts[0]),
                iou=float(parts[1]),
                max_detections=int(parts[2]),
                label=parts[3] if len(parts) == 4 else f"combo_{index}",
            )
        )
    return combos


def score_metrics(metrics: dict[str, Any], total_predictions: int, image_count: int) -> float:
    avg_predictions = total_predictions / max(1, image_count)
    penalty = max(0.0, avg_predictions - 6.0) * 0.015
    return round(
        (metrics["map50"] * 0.35)
        + (metrics["map50_95"] * 0.30)
        + (metrics["precision_at_50"] * 0.20)
        + (metrics["recall_at_50"] * 0.15)
        - penalty,
        6,
    )


def rank_summary(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = summary["metrics"]
    return (
        float(summary["score"]),
        float(metrics["map50_95"]),
        float(metrics["map50"]),
        float(metrics["precision_at_50"]),
    )


def prediction_to_dict(prediction: Prediction) -> dict[str, Any]:
    x1, y1, x2, y2 = prediction.bbox
    return {
        "label": prediction.label,
        "confidence": round(prediction.confidence, 4),
        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
    }


def default_output_dir(model_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "ml" / "evaluation" / f"threshold-calibration-{model_id}-{stamp}"


def write_markdown_report(output_root: Path, payload: dict[str, Any]) -> None:
    current = payload["current_registry_thresholds"]
    lines = [
        f"# Calibracion de umbrales - {payload['model']}",
        "",
        f"Creado: `{payload['created_at']}`",
        f"Dataset: `{payload['dataset']}`",
        f"Split: `{payload['split']}`",
        f"Imagenes evaluadas: `{payload['evaluated_images']}` de `{payload['total_split_images']}`",
        f"Image size: `{payload['image_size']}`",
        "",
        "## Ajuste actual",
        "",
        f"- Confidence: `{current['confidence_threshold']}`",
        f"- IoU: `{current['iou_threshold']}`",
        f"- Max cajas: `{current['max_detections']}`",
        "",
        "## Comparacion",
        "",
        "| Ajuste | Conf | IoU | Max | Score | mAP@50 | mAP@50:95 | Precision@50 | Recall@50 | Cajas/img | Predicciones | Tiempo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        metrics = result["metrics"]
        lines.append(
            f"| `{result['key']}` | {result['confidence_threshold']:.2f} | "
            f"{result['iou_threshold']:.2f} | {result['max_detections']} | "
            f"{result['score']:.4f} | {metrics['map50']:.4f} | {metrics['map50_95']:.4f} | "
            f"{metrics['precision_at_50']:.4f} | {metrics['recall_at_50']:.4f} | "
            f"{result['avg_predictions_per_image']:.2f} | {result['total_predictions']} | "
            f"{result['avg_processing_time_ms']:.2f} ms |"
        )

    recommendation = payload.get("recommendation")
    if recommendation:
        lines.extend([
            "",
            "## Recomendacion",
            "",
            f"- Ajuste sugerido: `{recommendation['key']}`",
            f"- Confidence: `{recommendation['confidence_threshold']}`",
            f"- IoU: `{recommendation['iou_threshold']}`",
            f"- Max cajas: `{recommendation['max_detections']}`",
            f"- Score: `{recommendation['score']}`",
            f"- Salidas visuales: `{recommendation['annotated_dir']}`",
        ])

    lines.extend([
        "",
        "## Archivos",
        "",
        f"- Resumen JSON: `{output_root / 'threshold_calibration_summary.json'}`",
        f"- Resultados crudos: `{output_root / 'threshold_calibration_raw_results.json'}`",
        f"- Reporte Markdown: `{output_root / 'threshold_calibration_report.md'}`",
    ])
    (output_root / "threshold_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
