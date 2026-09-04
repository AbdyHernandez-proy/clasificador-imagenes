from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ml.training.datasets.yolo_detection import DetectionSample

DEFAULT_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


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


def select_validation_samples(samples: list[DetectionSample], limit: int) -> list[DetectionSample]:
    if limit <= 0 or limit >= len(samples):
        return samples

    step = len(samples) / limit
    selected = [samples[min(len(samples) - 1, int(index * step))] for index in range(limit)]
    deduped: list[DetectionSample] = []
    seen: set[Path] = set()
    for sample in selected:
        if sample.image_path in seen:
            continue
        seen.add(sample.image_path)
        deduped.append(sample)
    return deduped


def evaluate_torchvision_model(
    model: Any,
    samples: list[DetectionSample],
    classes: list[str],
    device: Any,
    confidence_threshold: float,
    max_detections: int,
    visual_output_dir: Path | None = None,
    visual_limit: int = 0,
) -> dict[str, Any]:
    import torch
    from torchvision.transforms import functional as F

    model.eval()
    predictions_by_image: dict[str, list[Prediction]] = {}
    ground_truth_by_image: dict[str, list[GroundTruth]] = {}
    timings: list[float] = []
    failures: list[dict[str, str]] = []

    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            image_key = sample.image_path.name
            ground_truth = targets_to_ground_truth(sample, classes)
            ground_truth_by_image[image_key] = ground_truth
            started = time.perf_counter()
            try:
                image = Image.open(sample.image_path).convert("RGB")
                tensor = F.convert_image_dtype(F.pil_to_tensor(image), dtype=torch.float32).to(device)
                output = model([tensor])[0]
                timings.append((time.perf_counter() - started) * 1000)
                predictions = output_to_predictions(
                    image_key=image_key,
                    output=output,
                    classes=classes,
                    confidence_threshold=confidence_threshold,
                    max_detections=max_detections,
                )
                predictions_by_image[image_key] = predictions
                if visual_output_dir and index <= visual_limit:
                    save_visual_evaluation_image(
                        image=image,
                        output_path=visual_output_dir / f"{index:04d}_{sample.image_path.stem}.jpg",
                        predictions=predictions,
                        ground_truth=ground_truth,
                    )
            except Exception as exc:  # noqa: BLE001 - validacion: registrar y continuar.
                failures.append({"image": image_key, "error": str(exc)})
                predictions_by_image[image_key] = []

            if index % 25 == 0 or index == len(samples):
                print(f"validacion: {index}/{len(samples)} imagenes", flush=True)

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


def save_visual_evaluation_image(
    image: Image.Image,
    output_path: Path,
    predictions: list[Prediction],
    ground_truth: list[GroundTruth],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, gt in enumerate(ground_truth, start=1):
        draw_box_with_label(
            draw=draw,
            bbox=gt.bbox,
            label=f"GT {index} {gt.label}",
            color=(255, 196, 0),
            font=font,
        )

    for index, prediction in enumerate(predictions, start=1):
        draw_box_with_label(
            draw=draw,
            bbox=prediction.bbox,
            label=f"#{index} {prediction.label} {prediction.confidence:.2f}",
            color=(0, 190, 140),
            font=font,
        )

    canvas.save(output_path, quality=92)


def draw_box_with_label(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    label_bbox = draw.textbbox((x1, y1), label, font=font)
    label_width = label_bbox[2] - label_bbox[0]
    label_height = label_bbox[3] - label_bbox[1]
    label_y = max(0, y1 - label_height - 4)
    draw.rectangle((x1, label_y, x1 + label_width + 6, label_y + label_height + 4), fill=color)
    draw.text((x1 + 3, label_y + 2), label, fill=(0, 0, 0), font=font)


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


def output_to_predictions(
    image_key: str,
    output: dict[str, Any],
    classes: list[str],
    confidence_threshold: float,
    max_detections: int,
) -> list[Prediction]:
    boxes = output.get("boxes", [])
    scores = output.get("scores", [])
    labels = output.get("labels", [])
    predictions: list[Prediction] = []

    for box, score, label in zip(boxes, scores, labels):
        confidence = float(score.detach().cpu())
        if confidence < confidence_threshold:
            continue
        label_index = int(label.detach().cpu()) - 1
        if not (0 <= label_index < len(classes)):
            continue
        x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        predictions.append(
            Prediction(
                image_key=image_key,
                label=classes[label_index],
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
            )
        )

    predictions.sort(key=lambda item: item.confidence, reverse=True)
    if max_detections > 0:
        return predictions[:max_detections]
    return predictions


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
    result = []
    for value in values:
        total += value
        result.append(total)
    return result


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
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
