from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROTOCOL_STDOUT = sys.stdout


def main() -> None:
    models: dict[str, Any] = {}

    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from PIL import Image
        from torchvision.transforms import functional as F

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            model_path = Path(request["model"])
            image_path = Path(request["image"])
            model_id = request["model_id"]
            confidence = float(request.get("conf", 0.25))
            max_detections = int(request.get("max_detections", 50))
            nms_threshold = float(request.get("nms_threshold", 0.5))
            detections_per_img = int(request.get("detections_per_img", max_detections))

            if not model_path.exists():
                raise FileNotFoundError(f"No existe el modelo: {model_path}")
            if not image_path.exists():
                raise FileNotFoundError(f"No existe la imagen: {image_path}")

            cache_key = f"{model_path.resolve()}|nms={nms_threshold}|detections={detections_per_img}"
            if cache_key not in models:
                models[cache_key] = load_model(
                    model_id=model_id,
                    model_path=model_path,
                    device=device,
                    nms_threshold=nms_threshold,
                    detections_per_img=detections_per_img
                )

            model, classes = models[cache_key]
            image = Image.open(image_path).convert("RGB")
            image_tensor = F.convert_image_dtype(F.pil_to_tensor(image), dtype=torch.float32).to(device)

            with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
                result = model([image_tensor])[0]

            write_response({
                "ok": True,
                "predictions": normalize_result(
                    result=result,
                    classes=classes,
                    confidence=confidence,
                    max_detections=max_detections
                )
            })
        except Exception as exc:  # noqa: BLE001 - proceso worker: reportar error al backend.
            write_response({"ok": False, "error": str(exc)})


def load_model(
    model_id: str,
    model_path: Path,
    device: Any,
    nms_threshold: float,
    detections_per_img: int
) -> tuple[Any, list[str]]:
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes") or []
    num_classes = len(classes) + 1

    if model_id == "custom-faster-rcnn-detector":
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            box_score_thresh=0.0,
            box_nms_thresh=nms_threshold,
            box_detections_per_img=detections_per_img
        )
    elif model_id == "custom-retinanet-detector":
        model = retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            score_thresh=0.0,
            nms_thresh=nms_threshold,
            detections_per_img=detections_per_img
        )
    else:
        raise ValueError(f"Modelo torchvision no soportado: {model_id}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def normalize_result(
    result: dict[str, Any],
    classes: list[str],
    confidence: float,
    max_detections: int
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", [])

    for index, (box, score, label_id) in enumerate(zip(boxes, scores, labels), start=1):
        score_value = float(score.detach().cpu())
        if score_value < confidence:
            continue

        class_id = int(label_id.detach().cpu())
        label = label_from_classes(classes, class_id)
        x_min, y_min, x_max, y_max = [float(value) for value in box.detach().cpu().tolist()]
        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)

        if width <= 0 or height <= 0:
            continue

        predictions.append({
            "label": label,
            "class": label,
            "className": label,
            "confidence": round(score_value, 4),
            "score": round(score_value, 4),
            "probability": round(score_value, 4),
            "type": "object_detection",
            "class_id": class_id,
            "rank": len(predictions) + 1,
            "bbox": [
                round(x_min, 2),
                round(y_min, 2),
                round(width, 2),
                round(height, 2)
            ],
            "details": f"Clase COCO #{max(class_id - 1, 0)}"
        })

        if len(predictions) >= max_detections:
            break

    return predictions


def label_from_classes(classes: list[str], class_id: int) -> str:
    class_index = class_id - 1
    if 0 <= class_index < len(classes):
        return str(classes[class_index])
    return f"clase {class_id}"


def write_response(payload: dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTOCOL_STDOUT.flush()


if __name__ == "__main__":
    main()
