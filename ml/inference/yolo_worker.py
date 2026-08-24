from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.yolo_predict import normalize_result

PROTOCOL_STDOUT = sys.stdout


def main() -> None:
    models: dict[str, Any] = {}

    with contextlib.redirect_stdout(sys.stderr):
        from ultralytics import YOLO

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            model_path = Path(request["model"])
            image_path = Path(request["image"])
            confidence = float(request.get("conf", 0.25))
            image_size = int(request.get("imgsz", 640))

            if not model_path.exists():
                raise FileNotFoundError(f"No existe el modelo: {model_path}")
            if not image_path.exists():
                raise FileNotFoundError(f"No existe la imagen: {image_path}")

            cache_key = str(model_path.resolve())
            if cache_key not in models:
                with contextlib.redirect_stdout(sys.stderr):
                    models[cache_key] = YOLO(cache_key)

            with contextlib.redirect_stdout(sys.stderr):
                result = models[cache_key].predict(
                    source=str(image_path),
                    imgsz=image_size,
                    conf=confidence,
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
