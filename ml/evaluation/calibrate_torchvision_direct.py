from __future__ import annotations

import argparse
import json
import sys
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
from ml.training.datasets.yolo_detection import build_yolo_index  # noqa: E402
from ml.training.torchvision_validation import evaluate_torchvision_model, select_validation_samples  # noqa: E402


@dataclass(frozen=True)
class CalibrationCombo:
    confidence: float
    nms: float
    max_detections: int

    @property
    def key(self) -> str:
        confidence_key = int(round(self.confidence * 100))
        nms_key = int(round(self.nms * 100))
        return f"conf_{confidence_key:03d}_nms_{nms_key:03d}_max_{self.max_detections}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibra confidence/NMS/max_detections cargando el modelo TorchVision directamente."
    )
    parser.add_argument("--dataset", default="voc-detect")
    parser.add_argument("--split", default="val")
    parser.add_argument("--model", default="custom-retinanet-detector")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--visual-limit", type=int, default=12)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--combos",
        default="0.45,0.35,8;0.50,0.35,8;0.55,0.35,8;0.60,0.35,8;0.65,0.35,8;0.70,0.35,7;0.75,0.35,6;0.80,0.35,5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = get_dataset(args.dataset)
    if not dataset:
        raise SystemExit(f"Dataset no registrado: {args.dataset}")

    dataset_root = resolve_project_path(dataset["dataset_path"])
    split_path = dataset["splits"][args.split]
    classes = list(dataset.get("classes") or [])
    all_samples = build_yolo_index(dataset_root=dataset_root, image_split=split_path)
    samples = select_validation_samples(all_samples, args.limit)
    combos = parse_combos(args.combos)

    output_root = Path(args.output_dir) if args.output_dir else default_output_dir(args.model)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    registry = ModelRegistry()
    model_config = registry.get_model(args.model)
    if not model_config:
        raise SystemExit(f"Modelo no registrado: {args.model}")

    results: list[dict[str, Any]] = []
    for combo in combos:
        print(
            f"Calibrando {args.model}: confidence={combo.confidence:.2f}, "
            f"nms={combo.nms:.2f}, max={combo.max_detections}",
            flush=True,
        )
        model, loaded_classes = load_torchvision_model(
            model_id=args.model,
            model_config=model_config,
            device=device,
            nms_threshold=combo.nms,
            detections_per_img=combo.max_detections,
            image_size=int(model_config.get("training", {}).get("image_size") or 512),
        )
        if loaded_classes:
            classes = loaded_classes

        combo_output = output_root / combo.key
        summary = evaluate_torchvision_model(
            model=model,
            samples=samples,
            classes=classes,
            device=device,
            confidence_threshold=combo.confidence,
            max_detections=combo.max_detections,
            visual_output_dir=combo_output / "annotated",
            visual_limit=args.visual_limit,
        )
        metrics = summary["metrics"]
        result = {
            "key": combo.key,
            "confidence_threshold": combo.confidence,
            "nms_threshold": combo.nms,
            "max_detections": combo.max_detections,
            "evaluated_images": summary["evaluated_images"],
            "images_with_detections": summary["images_with_detections"],
            "total_predictions": summary["total_predictions"],
            "total_ground_truth": summary["total_ground_truth"],
            "avg_predictions_per_image": summary["avg_predictions_per_image"],
            "avg_processing_time_ms": summary["avg_processing_time_ms"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
            "precision_at_50": metrics["precision_at_50"],
            "recall_at_50": metrics["recall_at_50"],
            "true_positives_at_50": metrics["true_positives_at_50"],
            "false_positives_at_50": metrics["false_positives_at_50"],
            "false_negatives_at_50": metrics["false_negatives_at_50"],
            "annotated_dir": str(combo_output / "annotated"),
        }
        results.append(result)
        print(format_result_line(result), flush=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "device": str(device),
        "total_split_images": len(all_samples),
        "evaluated_images": len(samples),
        "results": results,
        "recommendation": recommend(results),
    }
    (output_root / "threshold_calibration_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(output_root, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def resolve_torch_device(value: str) -> Any:
    import torch

    if value != "auto":
        return torch.device(value)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_torchvision_model(
    model_id: str,
    model_config: dict[str, Any],
    device: Any,
    nms_threshold: float,
    detections_per_img: int,
    image_size: int,
) -> tuple[Any, list[str]]:
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn

    artifact_path = resolve_project_path(model_config["artifact_path"])
    checkpoint = torch.load(artifact_path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes") or list(model_config.get("labels") or [])
    num_classes = len(classes) + 1

    if model_id == "custom-faster-rcnn-detector":
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            min_size=image_size,
            max_size=image_size,
            box_score_thresh=0.0,
            box_nms_thresh=nms_threshold,
            box_detections_per_img=detections_per_img,
        )
    elif model_id == "custom-retinanet-detector":
        model = retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            min_size=image_size,
            max_size=image_size,
            score_thresh=0.0,
            nms_thresh=nms_threshold,
            detections_per_img=detections_per_img,
        )
    else:
        raise ValueError(f"Modelo TorchVision no soportado: {model_id}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def parse_combos(value: str) -> list[CalibrationCombo]:
    combos: list[CalibrationCombo] = []
    for raw_combo in value.split(";"):
        raw_combo = raw_combo.strip()
        if not raw_combo:
            continue
        parts = [part.strip() for part in raw_combo.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Combo invalido: {raw_combo}")
        combos.append(
            CalibrationCombo(
                confidence=float(parts[0]),
                nms=float(parts[1]),
                max_detections=int(parts[2]),
            )
        )
    if not combos:
        raise ValueError("No se indicaron combos de calibracion.")
    return combos


def recommend(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}

    viable = [
        item for item in results
        if item["precision_at_50"] >= 0.85 and item["recall_at_50"] >= 0.45
    ]
    if viable:
        best = max(
            viable,
            key=lambda item: (
                item["map50_95"],
                item["recall_at_50"],
                item["precision_at_50"],
                -item["avg_predictions_per_image"],
            ),
        )
        reason = "Mejor balance con precision alta y recall util."
    else:
        best = max(
            results,
            key=lambda item: (
                item["precision_at_50"] * 0.45
                + item["recall_at_50"] * 0.35
                + item["map50_95"] * 0.20
                - max(0.0, item["avg_predictions_per_image"] - 5.0) * 0.03
            ),
        )
        reason = "Mejor balance encontrado dentro de los umbrales evaluados."

    return {
        "key": best["key"],
        "confidence_threshold": best["confidence_threshold"],
        "nms_threshold": best["nms_threshold"],
        "max_detections": best["max_detections"],
        "precision_at_50": best["precision_at_50"],
        "recall_at_50": best["recall_at_50"],
        "map50": best["map50"],
        "map50_95": best["map50_95"],
        "false_positives_at_50": best["false_positives_at_50"],
        "false_negatives_at_50": best["false_negatives_at_50"],
        "reason": reason,
    }


def format_result_line(result: dict[str, Any]) -> str:
    return (
        f"{result['key']}: precision={result['precision_at_50']:.4f}, "
        f"recall={result['recall_at_50']:.4f}, mAP50={result['map50']:.4f}, "
        f"mAP50-95={result['map50_95']:.4f}, fp={result['false_positives_at_50']}, "
        f"fn={result['false_negatives_at_50']}, preds={result['total_predictions']}"
    )


def default_output_dir(model_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "ml" / "evaluation" / f"threshold-calibration-{model_id}-{stamp}"


def write_markdown_report(output_root: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Calibracion directa de umbrales TorchVision",
        "",
        f"Modelo: `{payload['model']}`",
        f"Dataset: `{payload['dataset']}`",
        f"Split: `{payload['split']}`",
        f"Device: `{payload['device']}`",
        f"Imagenes evaluadas: `{payload['evaluated_images']}` de `{payload['total_split_images']}`",
        "",
        "| Ajuste | Confidence | NMS | Max cajas | Precision@50 | Recall@50 | mAP@50 | mAP@50:95 | FP | FN | Predicciones |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in sorted(payload["results"], key=lambda item: item["precision_at_50"], reverse=True):
        lines.append(
            f"| `{result['key']}` | {result['confidence_threshold']:.2f} | "
            f"{result['nms_threshold']:.2f} | {result['max_detections']} | "
            f"{result['precision_at_50']:.4f} | {result['recall_at_50']:.4f} | "
            f"{result['map50']:.4f} | {result['map50_95']:.4f} | "
            f"{result['false_positives_at_50']} | {result['false_negatives_at_50']} | "
            f"{result['total_predictions']} |"
        )

    recommendation = payload["recommendation"]
    lines.extend(
        [
            "",
            "## Recomendacion",
            "",
            f"- Ajuste sugerido: `{recommendation.get('key')}`",
            f"- Confidence: `{recommendation.get('confidence_threshold')}`",
            f"- NMS: `{recommendation.get('nms_threshold')}`",
            f"- Max cajas: `{recommendation.get('max_detections')}`",
            f"- Motivo: {recommendation.get('reason')}",
            "",
            "## Salidas visuales",
            "",
        ]
    )
    for result in payload["results"]:
        lines.append(f"- `{result['key']}`: `{result['annotated_dir']}`")

    lines.extend(
        [
            "",
            "## Archivos",
            "",
            f"- Resumen JSON: `{output_root / 'threshold_calibration_summary.json'}`",
            f"- Reporte Markdown: `{output_root / 'threshold_calibration_report.md'}`",
        ]
    )
    (output_root / "threshold_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
