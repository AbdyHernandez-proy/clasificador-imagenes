from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for candidate in (PROJECT_ROOT, BACKEND_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.services.model_registry import ModelRegistry  # noqa: E402
from ml.training.dataset_catalog import get_dataset, resolve_project_path  # noqa: E402
from ml.training.datasets.yolo_detection import DetectionSample, build_yolo_index  # noqa: E402
from ml.training.local_train import EfficientDetYoloDataset, efficientdet_collate  # noqa: E402
from ml.training.torchvision_validation import (  # noqa: E402
    Prediction,
    compute_metrics,
    save_visual_evaluation_image,
    select_validation_samples,
    targets_to_ground_truth,
)


@dataclass(frozen=True)
class CalibrationCombo:
    confidence: float
    nms: float
    max_detections: int
    label: str

    @property
    def key(self) -> str:
        confidence_key = int(round(self.confidence * 100))
        nms_key = int(round(self.nms * 100))
        return f"{self.label}_conf_{confidence_key:03d}_nms_{nms_key:03d}_max_{self.max_detections}"


DEFAULT_COMBOS = [
    CalibrationCombo(0.30, 0.50, 8, "sensible"),
    CalibrationCombo(0.35, 0.50, 8, "actual"),
    CalibrationCombo(0.40, 0.50, 8, "balanceado"),
    CalibrationCombo(0.45, 0.50, 8, "conservador"),
    CalibrationCombo(0.50, 0.50, 7, "conservador"),
    CalibrationCombo(0.35, 0.45, 8, "nms_bajo"),
    CalibrationCombo(0.35, 0.55, 8, "nms_alto"),
    CalibrationCombo(0.40, 0.45, 6, "cajas_reducidas"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibra umbrales de EfficientDet sobre el dataset registrado.")
    parser.add_argument("--dataset", default="voc-detect")
    parser.add_argument("--split", default="val")
    parser.add_argument("--model", default="custom-efficientdet-detector")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--visual-limit", type=int, default=20)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=0)
    parser.add_argument("--artifact", default="")
    parser.add_argument("--base-model", default="")
    parser.add_argument(
        "--combos",
        default="",
        help="Formato: conf,nms,max,label;conf,nms,max,label. Si se omite usa la matriz recomendada.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = get_dataset(args.dataset)
    if not dataset:
        raise SystemExit(f"Dataset no registrado: {args.dataset}")

    dataset_root = resolve_project_path(dataset["dataset_path"])
    split_path = dataset["splits"][args.split]
    dataset_classes = list(dataset.get("classes") or [])
    all_samples = build_yolo_index(dataset_root=dataset_root, image_split=split_path)
    samples = select_validation_samples(all_samples, args.limit)

    registry = ModelRegistry()
    model_config = registry.get_model(args.model)
    if not model_config:
        raise SystemExit(f"No existe el modelo en registro: {args.model}")

    baseline = CalibrationCombo(
        confidence=float(model_config.get("confidence_threshold") or 0.35),
        nms=float(model_config.get("nms_threshold") or 0.5),
        max_detections=int(model_config.get("max_detections") or model_config.get("detections_per_img") or 8),
        label="actual_registro",
    )
    combos = parse_combos(args.combos) if args.combos else [baseline, *DEFAULT_COMBOS]
    output_root = Path(args.output_dir) if args.output_dir else default_output_dir(args.model)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    device, torch = resolve_torch_device(args.device)
    artifact_path = resolve_project_path(args.artifact or str(model_config.get("artifact_path") or ""))
    if not artifact_path.exists():
        raise SystemExit(f"No existe el artefacto EfficientDet: {artifact_path}")

    checkpoint = load_checkpoint(artifact_path=artifact_path, device=device, torch=torch)
    classes = [str(value) for value in checkpoint.get("classes") or model_config.get("labels") or dataset_classes]
    if not classes:
        raise SystemExit("No se encontraron clases en el artefacto, registro o dataset.")

    image_size = args.imgsz or int(model_config.get("training", {}).get("image_size") or 512)
    base_model = args.base_model or str(model_config.get("training", {}).get("base_model") or "tf_efficientdet_d0")
    internal_max_detections = max(100, *(combo.max_detections for combo in combos))
    model = load_efficientdet_model(
        checkpoint=checkpoint,
        classes=classes,
        device=device,
        image_size=image_size,
        base_model=base_model,
        max_detections=internal_max_detections,
    )

    raw_results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for combo in combos:
        print(
            f"Calibrando {args.model}: conf={combo.confidence:.2f}, "
            f"nms={combo.nms:.2f}, max={combo.max_detections} ({combo.label})",
            flush=True,
        )
        combo_dir = output_root / combo.key
        result = evaluate_combo(
            model=model,
            combo=combo,
            samples=samples,
            classes=classes,
            device=device,
            image_size=image_size,
            output_dir=combo_dir,
            visual_limit=args.visual_limit,
        )
        raw_results.extend(result["raw_results"])
        summaries.append(result["summary"])
        print(format_result_line(result["summary"]), flush=True)

    summaries.sort(key=rank_summary, reverse=True)
    recommendation = summaries[0] if summaries else None
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "artifact_path": str(artifact_path),
        "dataset": args.dataset,
        "split": args.split,
        "device": str(device),
        "total_split_images": len(all_samples),
        "evaluated_images": len(samples),
        "image_size": image_size,
        "base_model": base_model,
        "current_registry_thresholds": {
            "confidence_threshold": baseline.confidence,
            "nms_threshold": baseline.nms,
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
    combo: CalibrationCombo,
    samples: list[DetectionSample],
    classes: list[str],
    device: Any,
    image_size: int,
    output_dir: Path,
    visual_limit: int,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    model.eval()
    predictions_by_image: dict[str, list[Prediction]] = {}
    ground_truth_by_image = {}
    raw_results: list[dict[str, Any]] = []
    timings: list[float] = []
    failures: list[dict[str, str]] = []
    validation_dataset = EfficientDetYoloDataset(samples, image_size=image_size, training=False)
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            image_key = sample.image_path.name
            ground_truth = targets_to_ground_truth(sample, classes)
            ground_truth_by_image[image_key] = ground_truth
            started = time.perf_counter()
            try:
                image_tensor, target = validation_dataset[index - 1]
                images, targets = efficientdet_collate([(image_tensor, target)])
                images = images.to(device)
                targets = {key: value.to(device) for key, value in targets.items()}
                output = model(images, targets)
                elapsed_ms = (time.perf_counter() - started) * 1000
                timings.append(elapsed_ms)
                detections = output.get("detections")
                predictions = detections_to_predictions(
                    image_key=image_key,
                    detections=detections[0] if detections is not None else [],
                    classes=classes,
                    confidence_threshold=combo.confidence,
                    nms_threshold=combo.nms,
                    max_detections=combo.max_detections,
                )
                predictions_by_image[image_key] = predictions
                if index <= visual_limit:
                    image = Image.open(sample.image_path).convert("RGB")
                    save_visual_evaluation_image(
                        image=image,
                        output_path=annotated_dir / f"{index:04d}_{sample.image_path.stem}_{combo.key}.jpg",
                        predictions=predictions,
                        ground_truth=ground_truth,
                    )
                raw_results.append({
                    "combo": combo.key,
                    "image": image_key,
                    "ok": True,
                    "processing_time_ms": round(elapsed_ms, 2),
                    "prediction_count": len(predictions),
                    "ground_truth_count": len(ground_truth),
                    "top_predictions": [prediction_to_dict(item) for item in predictions[:10]],
                })
            except Exception as exc:  # noqa: BLE001 - calibracion: registrar y continuar.
                failures.append({"image": image_key, "error": str(exc)})
                predictions_by_image[image_key] = []
                raw_results.append({
                    "combo": combo.key,
                    "image": image_key,
                    "ok": False,
                    "error": str(exc),
                    "prediction_count": 0,
                    "ground_truth_count": len(ground_truth),
                })

            if index % 25 == 0 or index == len(samples):
                print(f"{combo.key}: {index}/{len(samples)} imagenes", flush=True)

    metrics = compute_metrics(ground_truth_by_image, predictions_by_image, classes)
    total_predictions = sum(len(values) for values in predictions_by_image.values())
    summary = {
        "key": combo.key,
        "label": combo.label,
        "confidence_threshold": combo.confidence,
        "nms_threshold": combo.nms,
        "max_detections": combo.max_detections,
        "evaluated_images": len(samples),
        "ok": len(samples) - len(failures),
        "failures": failures,
        "images_with_detections": sum(1 for values in predictions_by_image.values() if values),
        "total_predictions": total_predictions,
        "total_ground_truth": sum(len(values) for values in ground_truth_by_image.values()),
        "avg_predictions_per_image": round(total_predictions / max(1, len(samples)), 4),
        "avg_processing_time_ms": round(sum(timings) / max(1, len(timings)), 2),
        "metrics": metrics,
        "score": score_metrics(metrics, total_predictions, len(samples)),
        "annotated_dir": str(annotated_dir),
    }
    return {"summary": summary, "raw_results": raw_results}


def load_efficientdet_model(
    checkpoint: dict[str, Any],
    classes: list[str],
    device: Any,
    image_size: int,
    base_model: str,
    max_detections: int,
) -> Any:
    from effdet import create_model

    model = create_model(
        base_model,
        bench_task="train",
        num_classes=len(classes),
        image_size=(image_size, image_size),
        pretrained=False,
        pretrained_backbone=False,
        bench_labeler=True,
        max_det_per_image=max(max_detections, 8),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def detections_to_predictions(
    image_key: str,
    detections: Any,
    classes: list[str],
    confidence_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> list[Prediction]:
    predictions: list[Prediction] = []
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

    return apply_classwise_nms(predictions, iou_threshold=nms_threshold, max_detections=max_detections)


def apply_classwise_nms(
    predictions: list[Prediction],
    iou_threshold: float,
    max_detections: int,
) -> list[Prediction]:
    predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    kept: list[Prediction] = []
    for prediction in predictions:
        overlaps_same_class = any(
            prediction.label == candidate.label and iou_xyxy(prediction.bbox, candidate.bbox) >= iou_threshold
            for candidate in kept
        )
        if overlaps_same_class:
            continue
        kept.append(prediction)
        if max_detections > 0 and len(kept) >= max_detections:
            break
    return kept


def parse_combos(value: str) -> list[CalibrationCombo]:
    combos: list[CalibrationCombo] = []
    for index, raw_combo in enumerate(value.split(";"), start=1):
        raw_combo = raw_combo.strip()
        if not raw_combo:
            continue
        parts = [part.strip() for part in raw_combo.split(",")]
        if len(parts) not in {3, 4}:
            raise ValueError(f"Combo invalido: {raw_combo}")
        combos.append(
            CalibrationCombo(
                confidence=float(parts[0]),
                nms=float(parts[1]),
                max_detections=int(parts[2]),
                label=parts[3] if len(parts) == 4 else f"combo_{index}",
            )
        )
    if not combos:
        raise ValueError("No se indicaron combos de calibracion.")
    return combos


def resolve_torch_device(value: str) -> tuple[Any, Any]:
    import torch

    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), torch
    if value.isdigit():
        return torch.device(f"cuda:{value}"), torch
    return torch.device(value), torch


def load_checkpoint(artifact_path: Path, device: Any, torch: Any) -> dict[str, Any]:
    try:
        return torch.load(artifact_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(artifact_path, map_location=device)


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


def format_result_line(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return (
        f"{result['key']}: score={result['score']:.4f}, "
        f"precision={metrics['precision_at_50']:.4f}, recall={metrics['recall_at_50']:.4f}, "
        f"mAP50={metrics['map50']:.4f}, mAP50-95={metrics['map50_95']:.4f}, "
        f"fp={metrics['false_positives_at_50']}, fn={metrics['false_negatives_at_50']}, "
        f"preds={result['total_predictions']}"
    )


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
        f"Device: `{payload['device']}`",
        f"Imagenes evaluadas: `{payload['evaluated_images']}` de `{payload['total_split_images']}`",
        f"Image size: `{payload['image_size']}`",
        f"Arquitectura base: `{payload['base_model']}`",
        "",
        "## Ajuste actual",
        "",
        f"- Confidence: `{current['confidence_threshold']}`",
        f"- NMS post-proceso: `{current['nms_threshold']}`",
        f"- Max cajas: `{current['max_detections']}`",
        "",
        "## Comparacion",
        "",
        "| Ajuste | Conf | NMS | Max | Score | mAP@50 | mAP@50:95 | Precision@50 | Recall@50 | Cajas/img | Predicciones | Tiempo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        metrics = result["metrics"]
        lines.append(
            f"| `{result['key']}` | {result['confidence_threshold']:.2f} | "
            f"{result['nms_threshold']:.2f} | {result['max_detections']} | "
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
            f"- NMS post-proceso: `{recommendation['nms_threshold']}`",
            f"- Max cajas: `{recommendation['max_detections']}`",
            f"- Score: `{recommendation['score']}`",
            f"- Salidas visuales: `{recommendation['annotated_dir']}`",
        ])

    lines.extend([
        "",
        "## Nota tecnica",
        "",
        "EfficientDet aplica su propio filtrado interno. Esta calibracion agrega un NMS por clase en post-proceso para que el ajuste sea comparable y controlable desde el proyecto.",
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
