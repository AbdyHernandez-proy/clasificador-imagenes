from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for candidate in (PROJECT_ROOT, BACKEND_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fastapi import UploadFile  # noqa: E402

from app.services.inference import InferenceService  # noqa: E402
from app.services.model_registry import ModelRegistry  # noqa: E402
from ml.training.dataset_catalog import load_dataset_registry  # noqa: E402
from ml.training.datasets.yolo_detection import DetectionSample, build_yolo_index  # noqa: E402

DEFAULT_MODELS = [
    "custom-faster-rcnn-detector",
    "custom-retinanet-detector",
]
DEFAULT_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
COLORS = [
    (0, 184, 148),
    (108, 92, 231),
    (253, 203, 110),
    (214, 48, 49),
    (9, 132, 227),
    (232, 67, 147),
    (0, 206, 201),
    (225, 112, 85),
]


@dataclass(frozen=True)
class GroundTruth:
    label: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    image_key: str
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida modelos TorchVision contra un split YOLO/VOC y genera imagenes anotadas."
    )
    parser.add_argument("--dataset", default="voc-detect")
    parser.add_argument("--split", default="val")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--visual-limit", type=int, default=40)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--min-visual-confidence", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


async def run(args: argparse.Namespace) -> None:
    dataset = get_dataset(args.dataset)
    dataset_root = resolve_project_path(dataset["dataset_path"])
    split_path = dataset["splits"][args.split]
    classes = list(dataset.get("classes") or [])
    all_samples = build_yolo_index(dataset_root=dataset_root, image_split=split_path)
    samples = select_samples(all_samples, args.limit)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    output_root = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)

    service = InferenceService(ModelRegistry())
    raw_results: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    try:
        for model_id in models:
            print(f"Validando {model_id} con {len(samples)} imagenes...", flush=True)
            model_results = await validate_model(
                service=service,
                model_id=model_id,
                samples=samples,
                classes=classes,
                output_root=output_root,
                visual_limit=args.visual_limit,
                min_visual_confidence=args.min_visual_confidence,
            )
            raw_results.extend(model_results["raw_results"])
            summaries[model_id] = model_results["summary"]
    finally:
        service.close()

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "split": args.split,
        "dataset_root": str(dataset_root),
        "total_split_images": len(all_samples),
        "evaluated_images": len(samples),
        "models": summaries,
    }
    (output_root / "validation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "validation_raw_results.json").write_text(
        json.dumps(raw_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(output_root, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


async def validate_model(
    service: InferenceService,
    model_id: str,
    samples: list[DetectionSample],
    classes: list[str],
    output_root: Path,
    visual_limit: int,
    min_visual_confidence: float,
) -> dict[str, Any]:
    predictions_by_image: dict[str, list[Prediction]] = {}
    ground_truth_by_image: dict[str, list[GroundTruth]] = {}
    raw_results: list[dict[str, Any]] = []
    timings: list[float] = []
    failures: list[dict[str, str]] = []
    annotated_dir = output_root / "annotated" / model_id
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples, start=1):
        image_key = sample.image_path.name
        ground_truth = targets_to_ground_truth(sample, classes)
        ground_truth_by_image[image_key] = ground_truth
        started = time.perf_counter()
        try:
            image_bytes = sample.image_path.read_bytes()
            upload = UploadFile(file=io.BytesIO(image_bytes), filename=sample.image_path.name)
            result = await service.predict(upload, model_id=model_id)
            elapsed_ms = float(result.get("processing_time_ms") or 0)
            timings.append(elapsed_ms)
            predictions = normalize_predictions(
                image_key=image_key,
                predictions=result.get("predictions", []),
            )
            predictions_by_image[image_key] = predictions

            if index <= visual_limit:
                save_annotated_image(
                    source_path=sample.image_path,
                    output_path=annotated_dir / f"{index:03d}_{sample.image_path.stem}_{model_id}.jpg",
                    predictions=[
                        prediction for prediction in predictions
                        if prediction.confidence >= min_visual_confidence
                    ],
                    title=f"{model_id} - {sample.image_path.name}",
                )

            raw_results.append({
                "model_id": model_id,
                "image": image_key,
                "ok": True,
                "processing_time_ms": elapsed_ms,
                "wall_time_ms": round((time.perf_counter() - started) * 1000, 2),
                "prediction_count": len(predictions),
                "ground_truth_count": len(ground_truth),
                "top_predictions": [
                    prediction_to_dict(prediction)
                    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True)[:10]
                ],
            })
        except Exception as exc:  # noqa: BLE001 - validacion: registrar y continuar.
            failures.append({"image": image_key, "error": str(getattr(exc, "detail", exc))})
            predictions_by_image[image_key] = []
            raw_results.append({
                "model_id": model_id,
                "image": image_key,
                "ok": False,
                "error": failures[-1]["error"],
                "wall_time_ms": round((time.perf_counter() - started) * 1000, 2),
                "prediction_count": 0,
                "ground_truth_count": len(ground_truth),
                "top_predictions": [],
            })

        if index % 25 == 0 or index == len(samples):
            print(f"{model_id}: {index}/{len(samples)} imagenes", flush=True)

    metrics = compute_metrics(ground_truth_by_image, predictions_by_image, classes)
    summary = {
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
        "metrics": metrics,
        "annotated_dir": str(annotated_dir),
    }
    return {"summary": summary, "raw_results": raw_results}


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


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "ml" / "evaluation" / f"voc-validation-{stamp}"


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


def targets_to_ground_truth(sample: DetectionSample, classes: list[str]) -> list[GroundTruth]:
    ground_truth: list[GroundTruth] = []
    for target in sample.targets:
        if not (0 <= target.class_id < len(classes)):
            continue
        box_width = target.width * sample.width
        box_height = target.height * sample.height
        x_min = target.x_center * sample.width - box_width / 2
        y_min = target.y_center * sample.height - box_height / 2
        ground_truth.append(
            GroundTruth(
                label=classes[target.class_id],
                bbox=(
                    max(0.0, x_min),
                    max(0.0, y_min),
                    min(float(sample.width), x_min + box_width),
                    min(float(sample.height), y_min + box_height),
                ),
            )
        )
    return ground_truth


def normalize_predictions(image_key: str, predictions: list[dict[str, Any]]) -> list[Prediction]:
    normalized: list[Prediction] = []
    for prediction in predictions:
        bbox = prediction.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x_min, y_min, width, height = [float(value) for value in bbox]
        if width <= 0 or height <= 0:
            continue
        label = str(prediction.get("label") or prediction.get("className") or prediction.get("class") or "")
        confidence = float(prediction.get("confidence") or prediction.get("score") or 0)
        normalized.append(
            Prediction(
                image_key=image_key,
                label=label,
                confidence=confidence,
                bbox=(x_min, y_min, x_min + width, y_min + height),
            )
        )
    return sorted(normalized, key=lambda item: item.confidence, reverse=True)


def compute_metrics(
    ground_truth_by_image: dict[str, list[GroundTruth]],
    predictions_by_image: dict[str, list[Prediction]],
    classes: list[str],
) -> dict[str, Any]:
    ap_by_threshold: dict[str, float] = {}
    class_metrics_at_50: dict[str, Any] = {}
    for threshold in DEFAULT_IOU_THRESHOLDS:
        threshold_metrics = compute_ap_at_threshold(
            ground_truth_by_image=ground_truth_by_image,
            predictions_by_image=predictions_by_image,
            classes=classes,
            iou_threshold=threshold,
        )
        ap_by_threshold[f"{threshold:.2f}"] = threshold_metrics["map"]
        if abs(threshold - 0.5) < 1e-9:
            class_metrics_at_50 = threshold_metrics["classes"]

    map50 = ap_by_threshold["0.50"]
    map50_95 = round(sum(ap_by_threshold.values()) / max(1, len(ap_by_threshold)), 4)
    totals = aggregate_precision_recall(class_metrics_at_50)
    return {
        "map50": map50,
        "map50_95": map50_95,
        "precision_at_50": totals["precision"],
        "recall_at_50": totals["recall"],
        "true_positives_at_50": totals["tp"],
        "false_positives_at_50": totals["fp"],
        "false_negatives_at_50": totals["fn"],
        "ap_by_iou": ap_by_threshold,
        "per_class_at_50": class_metrics_at_50,
    }


def compute_ap_at_threshold(
    ground_truth_by_image: dict[str, list[GroundTruth]],
    predictions_by_image: dict[str, list[Prediction]],
    classes: list[str],
    iou_threshold: float,
) -> dict[str, Any]:
    class_metrics: dict[str, Any] = {}
    ap_values: list[float] = []

    for label in classes:
        gt_for_label: dict[str, list[GroundTruth]] = {
            image_key: [gt for gt in values if gt.label == label]
            for image_key, values in ground_truth_by_image.items()
        }
        gt_count = sum(len(values) for values in gt_for_label.values())
        if gt_count == 0:
            continue

        predictions = [
            prediction
            for values in predictions_by_image.values()
            for prediction in values
            if prediction.label == label
        ]
        predictions.sort(key=lambda item: item.confidence, reverse=True)
        matched: dict[str, set[int]] = defaultdict(set)
        true_positive: list[int] = []
        false_positive: list[int] = []

        for prediction in predictions:
            candidates = gt_for_label.get(prediction.image_key, [])
            best_iou = 0.0
            best_index = -1
            for gt_index, gt in enumerate(candidates):
                if gt_index in matched[prediction.image_key]:
                    continue
                current_iou = iou_xyxy(prediction.bbox, gt.bbox)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_index = gt_index

            if best_iou >= iou_threshold and best_index >= 0:
                matched[prediction.image_key].add(best_index)
                true_positive.append(1)
                false_positive.append(0)
            else:
                true_positive.append(0)
                false_positive.append(1)

        cumulative_tp = cumulative_sum(true_positive)
        cumulative_fp = cumulative_sum(false_positive)
        recalls = [value / gt_count for value in cumulative_tp]
        precisions = [
            tp / max(1, tp + fp)
            for tp, fp in zip(cumulative_tp, cumulative_fp)
        ]
        ap = average_precision(recalls, precisions)
        tp_total = cumulative_tp[-1] if cumulative_tp else 0
        fp_total = cumulative_fp[-1] if cumulative_fp else 0
        fn_total = gt_count - tp_total
        precision = tp_total / max(1, tp_total + fp_total)
        recall = tp_total / max(1, gt_count)
        class_metrics[label] = {
            "gt": gt_count,
            "predictions": len(predictions),
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "ap": round(ap, 4),
        }
        ap_values.append(ap)

    return {
        "map": round(sum(ap_values) / max(1, len(ap_values)), 4),
        "classes": class_metrics,
    }


def cumulative_sum(values: list[int]) -> list[int]:
    total = 0
    result: list[int] = []
    for value in values:
        total += value
        result.append(total)
    return result


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls or not precisions:
        return 0.0
    ap = 0.0
    for recall_level in [index / 100 for index in range(101)]:
        precision_at_recall = [
            precision
            for recall, precision in zip(recalls, precisions)
            if recall >= recall_level
        ]
        ap += max(precision_at_recall) if precision_at_recall else 0.0
    return ap / 101


def aggregate_precision_recall(class_metrics: dict[str, Any]) -> dict[str, Any]:
    tp = sum(int(item["tp"]) for item in class_metrics.values())
    fp = sum(int(item["fp"]) for item in class_metrics.values())
    fn = sum(int(item["fn"]) for item in class_metrics.values())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(tp / max(1, tp + fp), 4),
        "recall": round(tp / max(1, tp + fn), 4),
    }


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def prediction_to_dict(prediction: Prediction) -> dict[str, Any]:
    x1, y1, x2, y2 = prediction.bbox
    return {
        "label": prediction.label,
        "confidence": round(prediction.confidence, 4),
        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
    }


def save_annotated_image(
    source_path: Path,
    output_path: Path,
    predictions: list[Prediction],
    title: str,
) -> None:
    image = Image.open(source_path).convert("RGB")
    font = ImageFont.load_default()
    panel_width = 380
    canvas = Image.new("RGB", (image.width + panel_width, image.height), (18, 28, 46))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([image.width, 0, image.width + panel_width, image.height], fill=(18, 28, 46))
    draw.text((image.width + 14, 14), title[:55], fill=(230, 239, 255), font=font)
    draw.text((image.width + 14, 34), "Cajas generadas por el modelo", fill=(171, 184, 208), font=font)

    for index, prediction in enumerate(predictions, start=1):
        color = COLORS[(index - 1) % len(COLORS)]
        x1, y1, x2, y2 = prediction.bbox
        x1 = max(0, min(image.width - 1, int(round(x1))))
        y1 = max(0, min(image.height - 1, int(round(y1))))
        x2 = max(0, min(image.width - 1, int(round(x2))))
        y2 = max(0, min(image.height - 1, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"#{index} {prediction.label} {prediction.confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_height = text_bbox[3] - text_bbox[1] + 6
        text_width = text_bbox[2] - text_bbox[0] + 8
        text_y = max(0, y1 - text_height)
        draw.rectangle([x1, text_y, x1 + text_width, text_y + text_height], fill=color)
        draw.text((x1 + 4, text_y + 3), label, fill=(0, 0, 0), font=font)

        legend_y = 62 + (index - 1) * 20
        if legend_y < image.height - 20:
            draw.rectangle([image.width + 14, legend_y + 3, image.width + 24, legend_y + 13], fill=color)
            draw.text(
                (image.width + 32, legend_y),
                label[:48],
                fill=(230, 239, 255),
                font=font,
            )

    if not predictions:
        draw.text(
            (image.width + 14, 62),
            "Sin detecciones sobre el umbral configurado.",
            fill=(230, 239, 255),
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def write_markdown_report(output_root: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Validacion VOC de Faster R-CNN y RetinaNet",
        "",
        f"Dataset: `{payload['dataset']}`",
        f"Split: `{payload['split']}`",
        f"Imagenes evaluadas: `{payload['evaluated_images']}` de `{payload['total_split_images']}`",
        "",
        "| Modelo | mAP@50 | mAP@50:95 | Precision@50 | Recall@50 | Predicciones | Tiempo medio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id, summary in payload["models"].items():
        metrics = summary["metrics"]
        lines.append(
            f"| {model_id} | {metrics['map50']} | {metrics['map50_95']} | "
            f"{metrics['precision_at_50']} | {metrics['recall_at_50']} | "
            f"{summary['total_predictions']} | {summary['avg_processing_time_ms']} ms |"
        )

    lines.extend(["", "## Salidas visuales", ""])
    for model_id, summary in payload["models"].items():
        lines.append(f"- `{model_id}`: `{summary['annotated_dir']}`")

    lines.extend(["", "## Archivos", ""])
    lines.append(f"- Resumen JSON: `{output_root / 'validation_summary.json'}`")
    lines.append(f"- Resultados crudos: `{output_root / 'validation_raw_results.json'}`")
    (output_root / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
