from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for candidate in (PROJECT_ROOT, BACKEND_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fastapi import UploadFile  # noqa: E402

from app.services.inference import InferenceService  # noqa: E402
from app.services.model_registry import ModelRegistry  # noqa: E402
from ml.inference.visual_output import (  # noqa: E402
    normalize_predictions,
    save_annotated_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta un modelo TorchVision sobre una imagen local y guarda una salida anotada."
    )
    parser.add_argument("--image", required=True, help="Ruta de la imagen a evaluar.")
    parser.add_argument("--model", default="custom-faster-rcnn-detector")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


async def run(args: argparse.Namespace) -> None:
    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"La ruta no es un archivo: {image_path}")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry()
    service = InferenceService(registry)
    try:
        image_bytes = image_path.read_bytes()
        upload = UploadFile(file=io.BytesIO(image_bytes), filename=image_path.name)
        result = await service.predict(upload, model_id=args.model)
    finally:
        service.close()

    predictions = normalize_predictions(
        image_key=image_path.name,
        predictions=result.get("predictions", []),
    )
    model_config = registry.get_model(args.model) or {}
    confidence = float(model_config.get("confidence_threshold") or 0.0)

    annotated_path = output_dir / f"{image_path.stem}_{args.model}_annotated.jpg"
    save_annotated_image(
        source_path=image_path,
        output_path=annotated_path,
        predictions=[prediction for prediction in predictions if prediction.confidence >= confidence],
        title=f"{args.model} - {image_path.name}",
    )

    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image": str(image_path),
        "model_id": result.get("model_id"),
        "model_name": result.get("model_name"),
        "runtime": result.get("runtime"),
        "processing_time_ms": result.get("processing_time_ms"),
        "confidence_threshold": confidence,
        "nms_threshold": model_config.get("nms_threshold"),
        "max_detections": model_config.get("max_detections"),
        "prediction_count": len(predictions),
        "annotated_image": str(annotated_path),
        "predictions": [
            {
                "label": prediction.label,
                "confidence": round(prediction.confidence, 4),
                "bbox_xyxy": [round(value, 2) for value in prediction.bbox],
            }
            for prediction in predictions
        ],
    }
    json_path = output_dir / f"{image_path.stem}_{args.model}_predictions.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "ml" / "evaluation" / "manual-predictions" / stamp


if __name__ == "__main__":
    main()
