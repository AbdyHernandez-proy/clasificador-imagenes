from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predict_efficientdet_image import (  # noqa: E402
    normalize_detections,
    preprocess_image_for_efficientdet,
)

PROTOCOL_STDOUT = sys.stdout


def main() -> None:
    models: dict[str, tuple[Any, list[str], Any]] = {}

    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from PIL import Image

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            model_path = Path(request["model"])
            image_path = Path(request["image"])
            model_id = str(request.get("model_id") or "")
            confidence = float(request.get("conf", 0.35))
            max_detections = int(request.get("max_detections", 8))
            image_size = int(request.get("image_size", 512))
            base_model = str(request.get("base_model") or "tf_efficientdet_d0")
            device = resolve_device(str(request.get("device") or "cpu"), torch)

            if not model_path.exists():
                raise FileNotFoundError(f"No existe el modelo: {model_path}")
            if not image_path.exists():
                raise FileNotFoundError(f"No existe la imagen: {image_path}")

            cache_key = f"{model_path.resolve()}|{model_id}|{base_model}|size={image_size}|device={device}"
            if cache_key not in models:
                models[cache_key] = load_model(
                    model_path=model_path,
                    device=device,
                    image_size=image_size,
                    base_model=base_model,
                )

            model, classes, device = models[cache_key]
            image = Image.open(image_path).convert("RGB")
            tensor, target = preprocess_image_for_efficientdet(image=image, image_size=image_size, device=device)

            with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
                output = model(tensor, target)

            detections = output.get("detections")
            predictions = normalize_detections(
                image_key=image_path.name,
                detections=detections[0] if detections is not None else [],
                classes=classes,
                confidence_threshold=confidence,
                max_detections=max_detections,
            )
            write_response({
                "ok": True,
                "predictions": [
                    prediction_to_payload(prediction=prediction, model_id=model_id, rank=index)
                    for index, prediction in enumerate(predictions, start=1)
                ],
            })
        except Exception as exc:  # noqa: BLE001 - proceso worker: reportar error al backend.
            write_response({"ok": False, "error": str(exc)})


def load_model(model_path: Path, device: Any, image_size: int, base_model: str) -> tuple[Any, list[str], Any]:
    import torch
    from effdet import create_model

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    classes = [str(value) for value in checkpoint.get("classes") or []]
    if not classes:
        raise ValueError("El artefacto EfficientDet no contiene clases.")

    model = create_model(
        base_model,
        bench_task="train",
        num_classes=len(classes),
        image_size=(image_size, image_size),
        pretrained=False,
        pretrained_backbone=False,
        bench_labeler=True,
        max_det_per_image=100,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, classes, device


def resolve_device(value: str, torch: Any) -> Any:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def prediction_to_payload(prediction: Any, model_id: str, rank: int) -> dict[str, Any]:
    x_min, y_min, x_max, y_max = prediction.bbox
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    class_id = int(prediction.class_id)
    return {
        "label": prediction.label,
        "class": prediction.label,
        "className": prediction.label,
        "confidence": round(float(prediction.confidence), 4),
        "score": round(float(prediction.confidence), 4),
        "probability": round(float(prediction.confidence), 4),
        "type": "object_detection",
        "model": model_id,
        "class_id": class_id,
        "rank": rank,
        "bbox": [
            round(float(x_min), 2),
            round(float(y_min), 2),
            round(width, 2),
            round(height, 2),
        ],
        "details": f"Clase VOC #{max(class_id - 1, 0)}",
    }


def write_response(payload: dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTOCOL_STDOUT.flush()


if __name__ == "__main__":
    main()
