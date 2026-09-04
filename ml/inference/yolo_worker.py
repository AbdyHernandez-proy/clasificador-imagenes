from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.yolo_predict import load_ultralytics_model, normalize_result

PROTOCOL_STDOUT = sys.stdout


def main() -> None:
    models: dict[str, Any] = {}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            model_path = Path(request["model"])
            image_path = Path(request["image"])
            model_id = str(request.get("model_id") or "")
            confidence = float(request.get("conf", 0.25))
            iou = float(request.get("iou", 0.7))
            max_det = int(request.get("max_det", 300))
            image_size = int(request.get("imgsz", 640))

            if not model_path.exists():
                raise FileNotFoundError(f"No existe el modelo: {model_path}")
            if not image_path.exists():
                raise FileNotFoundError(f"No existe la imagen: {image_path}")

            cache_key = f"{model_path.resolve()}|{model_id}"
            if cache_key not in models:
                with contextlib.redirect_stdout(sys.stderr):
                    models[cache_key] = load_ultralytics_model(str(model_path), model_id)

            with contextlib.redirect_stdout(sys.stderr):
                result = models[cache_key].predict(
                    source=str(image_path),
                    imgsz=image_size,
                    conf=confidence,
                    iou=iou,
                    max_det=max_det,
                    verbose=False
                )[0]

            write_response({"ok": True, "predictions": normalize_result(result)})
        except Exception as exc:  # noqa: BLE001 - proceso worker: reportar error al backend.
            write_response({"ok": False, "error": str(exc)})


def write_response(payload: dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTOCOL_STDOUT.flush()


if __name__ == "__main__":
    main()
