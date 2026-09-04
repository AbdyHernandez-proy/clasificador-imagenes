from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_ID = "custom-efficientdet-detector"
MODEL_REGISTRY_PATH = PROJECT_ROOT / "ml" / "registry.json"
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
        description="Ejecuta EfficientDet sobre una imagen local y guarda una salida visual anotada."
    )
    parser.add_argument("--image", required=True, help="Ruta de la imagen a evaluar.")
    parser.add_argument("--output-dir", default="", help="Carpeta donde guardar la imagen anotada y el JSON.")
    parser.add_argument("--conf", type=float, default=-1, help="Sobrescribe confidence_threshold del registro.")
    parser.add_argument("--max-det", type=int, default=0, help="Sobrescribe max_detections del registro.")
    parser.add_argument("--imgsz", type=int, default=0, help="Sobrescribe image_size del registro.")
    parser.add_argument("--artifact", default="", help="Sobrescribe artifact_path del registro.")
    parser.add_argument("--device", default="auto", help="auto, cpu o indice CUDA como 0.")
    parser.add_argument("--base-model", default="", help="Sobrescribe la arquitectura EfficientDet base.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = resolve_project_path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"La ruta no es un archivo: {image_path}")

    model_config = load_model_config()
    artifact_path = resolve_project_path(args.artifact or str(model_config.get("artifact_path") or ""))
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"No existe el artefacto EfficientDet: {artifact_path}. "
            "Entrena el modelo antes de ejecutar prediccion."
        )

    confidence = args.conf if args.conf >= 0 else float(model_config.get("confidence_threshold") or 0.5)
    max_detections = args.max_det if args.max_det > 0 else int(model_config.get("max_detections") or 8)
    image_size = args.imgsz if args.imgsz > 0 else int(model_config.get("training", {}).get("image_size") or 512)
    base_model = args.base_model or str(model_config.get("training", {}).get("base_model") or "tf_efficientdet_d0")

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from effdet import create_model

    device = torch.device(resolve_device(args.device, torch))
    checkpoint = torch.load(artifact_path, map_location=device)
    classes = [str(value) for value in checkpoint.get("classes") or model_config.get("labels") or []]
    if not classes:
        raise ValueError("El artefacto EfficientDet no contiene clases y el registro no tiene labels.")

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

    image = Image.open(image_path).convert("RGB")
    tensor, target = preprocess_image_for_efficientdet(image=image, image_size=image_size, device=device)

    start = time.perf_counter()
    with torch.inference_mode():
        output = model(tensor, target)
    processing_time_ms = round((time.perf_counter() - start) * 1000, 2)

    detections = output.get("detections")
    predictions = normalize_detections(
        image_key=image_path.name,
        detections=detections[0] if detections is not None else [],
        classes=classes,
        confidence_threshold=confidence,
        max_detections=max_detections,
    )

    annotated_path = output_dir / f"{image_path.stem}_{MODEL_ID}_annotated.jpg"
    json_path = output_dir / f"{image_path.stem}_{MODEL_ID}_predictions.json"
    save_annotated_image(
        source_path=image_path,
        output_path=annotated_path,
        predictions=predictions,
        title=f"{model_config.get('name', MODEL_ID)} - {image_path.name}",
    )

    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image": str(image_path),
        "model_id": MODEL_ID,
        "model_name": model_config.get("name", MODEL_ID),
        "artifact_path": str(artifact_path),
        "confidence_threshold": confidence,
        "max_detections": max_detections,
        "image_size": image_size,
        "base_model": base_model,
        "device": str(device),
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


def preprocess_image_for_efficientdet(image: Image.Image, image_size: int, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32)
    fill_color = tuple(int(round(255 * value)) for value in mean)

    width, height = image.size
    scale = min(image_size / height, image_size / width)
    scaled_height = max(1, int(height * scale))
    scaled_width = max(1, int(width * scale))
    resized = image.resize((scaled_width, scaled_height), Image.BILINEAR)
    padded = Image.new("RGB", (image_size, image_size), color=fill_color)
    padded.paste(resized, (0, 0))

    image_array = np.asarray(padded, dtype=np.float32) / 255.0
    image_array = (image_array - mean) / std
    image_array = np.moveaxis(image_array, 2, 0)

    tensor = torch.from_numpy(image_array).unsqueeze(0).to(device)
    target = {
        "bbox": torch.zeros((1, 1, 4), dtype=torch.float32, device=device),
        "cls": torch.full((1, 1), -1, dtype=torch.int64, device=device),
        "img_scale": torch.tensor([1.0 / scale], dtype=torch.float32, device=device),
        "img_size": torch.tensor([[width, height]], dtype=torch.float32, device=device),
    }
    return tensor, target


def normalize_detections(
    image_key: str,
    detections: Any,
    classes: list[str],
    confidence_threshold: float,
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
                label=classes[label_index],
                confidence=confidence,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                class_id=int(class_id),
            )
        )

    predictions.sort(key=lambda item: item.confidence, reverse=True)
    return predictions[:max_detections] if max_detections > 0 else predictions


def save_annotated_image(source_path: Path, output_path: Path, predictions: list[Prediction], title: str) -> None:
    from PIL import ImageDraw, ImageFont

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


def load_model_config() -> dict[str, Any]:
    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    for model in registry.get("models", []):
        if model.get("id") == MODEL_ID:
            return model
    raise ValueError(f"No existe el modelo '{MODEL_ID}' en ml/registry.json.")


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


def resolve_device(device: str, torch: Any) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.isdigit():
        return f"cuda:{device}"
    return device


if __name__ == "__main__":
    main()
