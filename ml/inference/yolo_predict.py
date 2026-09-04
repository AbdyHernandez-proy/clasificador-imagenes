from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta inferencia Ultralytics y devuelve JSON.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise SystemExit(f"No existe el modelo: {model_path}")
    if not image_path.exists():
        raise SystemExit(f"No existe la imagen: {image_path}")

    model = load_ultralytics_model(str(model_path), args.model_id)

    result = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=False
    )[0]

    payload = {
        "predictions": normalize_result(result)
    }
    print(json.dumps(payload, ensure_ascii=False))


def load_ultralytics_model(model_path: str, model_id: str = "") -> Any:
    from ultralytics import RTDETR, YOLO

    if "rtdetr" in model_id.lower() or "rtdetr" in Path(model_path).name.lower():
        return RTDETR(model_path)

    return YOLO(model_path)


def normalize_result(result: Any) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return predictions

    names = getattr(result, "names", None) or {}

    for index, box in enumerate(boxes):
        xyxy = box.xyxy[0].detach().cpu().tolist()
        confidence = float(box.conf[0].detach().cpu())
        class_id = int(box.cls[0].detach().cpu())
        label = names.get(class_id, f"clase {class_id}")
        x_min, y_min, x_max, y_max = xyxy
        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)

        if width <= 0 or height <= 0:
            continue

        predictions.append({
            "label": label,
            "class": label,
            "className": label,
            "confidence": round(confidence, 4),
            "score": round(confidence, 4),
            "probability": round(confidence, 4),
            "type": "object_detection",
            "class_id": class_id,
            "rank": index + 1,
            "bbox": [
                round(float(x_min), 2),
                round(float(y_min), 2),
                round(width, 2),
                round(height, 2)
            ],
            "details": f"Clase #{class_id}"
        })

    return sorted(predictions, key=lambda item: item["confidence"], reverse=True)


if __name__ == "__main__":
    main()
