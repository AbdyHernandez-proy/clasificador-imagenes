from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_REGISTRY_PATH = PROJECT_ROOT / "ml" / "registry.json"
DEFAULT_MODEL_ID = "custom-detr-rtdetr-detector"
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
class Prediction:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    class_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta un modelo Ultralytics sobre una imagen local y guarda una salida visual anotada."
    )
    parser.add_argument("--image", required=True, help="Ruta de la imagen a evaluar.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="ID del modelo en ml/registry.json.")
    parser.add_argument("--output-dir", default="", help="Carpeta donde guardar la imagen anotada y el JSON.")
    parser.add_argument("--conf", type=float, default=-1, help="Sobrescribe confidence_threshold del registro.")
    parser.add_argument("--iou", type=float, default=-1, help="Sobrescribe iou_threshold del registro.")
    parser.add_argument("--max-det", type=int, default=0, help="Sobrescribe max_detections del registro.")
    parser.add_argument("--imgsz", type=int, default=0, help="Sobrescribe image_size del registro.")
    parser.add_argument("--device", default="auto", help="auto, cpu o indice CUDA como 0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = resolve_project_path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"La ruta no es un archivo: {image_path}")

    model_config = load_model_config(args.model)
    artifact_path = resolve_project_path(str(model_config.get("artifact_path") or ""))
    if not artifact_path.exists():
        raise FileNotFoundError(f"No existe el artefacto del modelo: {artifact_path}")

    confidence = args.conf if args.conf >= 0 else float(model_config.get("confidence_threshold") or 0.25)
    iou = args.iou if args.iou >= 0 else float(model_config.get("iou_threshold") or model_config.get("nms_threshold") or 0.7)
    max_det = args.max_det if args.max_det > 0 else int(model_config.get("max_detections") or model_config.get("detections_per_img") or 300)
    image_size = args.imgsz if args.imgsz > 0 else int(model_config.get("training", {}).get("image_size") or 640)
    device = resolve_device(args.device)

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO, RTDETR

    model_class = RTDETR if "rtdetr" in args.model.lower() or "detr" in args.model.lower() else YOLO
    model = model_class(str(artifact_path))

    start = time.perf_counter()
    result = model.predict(
        source=str(image_path),
        imgsz=image_size,
        conf=confidence,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )[0]
    processing_time_ms = round((time.perf_counter() - start) * 1000, 2)

    predictions = normalize_result(result)
    annotated_path = output_dir / f"{image_path.stem}_{args.model}_annotated.jpg"
    json_path = output_dir / f"{image_path.stem}_{args.model}_predictions.json"
    save_annotated_image(
        source_path=image_path,
        output_path=annotated_path,
        predictions=predictions,
        title=f"{model_config.get('name', args.model)} - {image_path.name}",
    )

    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image": str(image_path),
        "model_id": args.model,
        "model_name": model_config.get("name", args.model),
        "artifact_path": str(artifact_path),
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "max_detections": max_det,
        "image_size": image_size,
        "device": device,
        "processing_time_ms": processing_time_ms,
        "prediction_count": len(predictions),
        "annotated_image": str(annotated_path),
        "predictions_json": str(json_path),
        "predictions": [
            {
                "rank": index,
                "label": prediction.label,
                "class_id": prediction.class_id,
                "confidence": round(prediction.confidence, 4),
                "bbox_xyxy": [round(value, 2) for value in prediction.bbox],
            }
            for index, prediction in enumerate(predictions, start=1)
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_output_dir(raw_path: str) -> Path:
    if raw_path:
        return resolve_project_path(raw_path)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "ml" / "evaluation" / "manual-predictions" / stamp


def load_model_config(model_id: str) -> dict[str, Any]:
    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    for model in registry.get("models", []):
        if model.get("id") == model_id:
            if model.get("runtime") != "ultralytics":
                raise ValueError(f"El modelo '{model_id}' no usa runtime ultralytics.")
            return model
    raise ValueError(f"No existe el modelo '{model_id}' en ml/registry.json.")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "0"
    return device


def normalize_result(result: Any) -> list[Prediction]:
    predictions: list[Prediction] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return predictions

    names = getattr(result, "names", None) or {}
    for box in boxes:
        xyxy = box.xyxy[0].detach().cpu().tolist()
        confidence = float(box.conf[0].detach().cpu())
        class_id = int(box.cls[0].detach().cpu())
        label = str(names.get(class_id, f"clase {class_id}"))
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        if x2 <= x1 or y2 <= y1:
            continue
        predictions.append(
            Prediction(
                label=label,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
            )
        )

    return sorted(predictions, key=lambda item: item.confidence, reverse=True)


def save_annotated_image(source_path: Path, output_path: Path, predictions: list[Prediction], title: str) -> None:
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

        label = f"#{index} {prediction.label} {prediction.confidence:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_height = text_bbox[3] - text_bbox[1] + 6
        text_width = text_bbox[2] - text_bbox[0] + 8
        text_y = max(0, y1 - text_height)
        draw.rectangle([x1, text_y, x1 + text_width, text_y + text_height], fill=color)
        draw.text((x1 + 4, text_y + 3), label, fill=(0, 0, 0), font=font)

        legend_y = 62 + (index - 1) * 20
        if legend_y < image.height - 20:
            draw.rectangle([image.width + 14, legend_y + 3, image.width + 24, legend_y + 13], fill=color)
            draw.text((image.width + 32, legend_y), label[:48], fill=(230, 239, 255), font=font)

    if not predictions:
        draw.text(
            (image.width + 14, 62),
            "Sin detecciones sobre el umbral configurado.",
            fill=(230, 239, 255),
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


if __name__ == "__main__":
    main()
