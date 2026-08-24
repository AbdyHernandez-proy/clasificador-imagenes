from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageStat, UnidentifiedImageError

from app.core.paths import PROJECT_ROOT
from app.services.model_registry import ModelRegistry

MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
YOLO_INFERENCE_TIMEOUT_SECONDS = 120


class InferenceService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._ultralytics_models: dict[str, Any] = {}
        self._yolo_worker: subprocess.Popen[str] | None = None
        self._yolo_worker_lock = threading.Lock()
        self._torchvision_worker: subprocess.Popen[str] | None = None
        self._torchvision_worker_lock = threading.Lock()

    def close(self) -> None:
        self._stop_yolo_worker()
        self._stop_torchvision_worker()

    async def predict(self, image: UploadFile, model_id: str | None = None) -> dict[str, Any]:
        selected_model_id = model_id or self.registry.get_default_model_id()
        if not selected_model_id:
            raise HTTPException(status_code=400, detail="No hay un modelo predeterminado configurado.")

        model = self.registry.get_model(selected_model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"El modelo '{selected_model_id}' no existe en ml/registry.json.")

        image_bytes = await image.read()
        pil_image = self._load_image(image, image_bytes)
        start_time = time.perf_counter()

        if model.get("runtime") == "backend_builtin":
            predictions = self._predict_visual_profile(pil_image)
        elif model.get("runtime") == "ultralytics":
            predictions = self._predict_ultralytics(model, pil_image, image_bytes)
        elif model.get("runtime") == "torchvision":
            predictions = self._predict_torchvision(model, image_bytes)
        else:
            raise HTTPException(
                status_code=501,
                detail=f"El runtime '{model.get('runtime')}' aun no esta implementado."
            )

        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_id": selected_model_id,
            "model_name": model.get("name", selected_model_id),
            "runtime": model.get("runtime"),
            "filename": image.filename,
            "image": {
                "width": pil_image.width,
                "height": pil_image.height,
                "mode": pil_image.mode
            },
            "predictions": predictions,
            "processing_time_ms": processing_time_ms
        }

    def _load_image(self, image: UploadFile, image_bytes: bytes) -> Image.Image:
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen valida.")

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="La imagen supera el tamano maximo permitido de 10 MB.")

        try:
            loaded_image = Image.open(io.BytesIO(image_bytes))
            loaded_image.verify()
            loaded_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada.") from exc

        return loaded_image

    def _predict_visual_profile(self, image: Image.Image) -> list[dict[str, Any]]:
        stat = ImageStat.Stat(image)
        red, green, blue = stat.mean[:3]
        brightness = (red + green + blue) / (3 * 255)
        channel_values = {"rojo": red, "verde": green, "azul": blue}
        dominant_channel = max(channel_values, key=channel_values.get)
        channel_spread = max(channel_values.values()) - min(channel_values.values())
        aspect_ratio = image.width / image.height if image.height else 1

        if brightness >= 0.66:
            light_label = "imagen clara"
            light_confidence = min(brightness, 0.99)
        elif brightness <= 0.34:
            light_label = "imagen oscura"
            light_confidence = min(1 - brightness, 0.99)
        else:
            light_label = "iluminacion media"
            light_confidence = 1 - abs(brightness - 0.5)

        if channel_spread < 18:
            color_label = "tono neutro"
            color_confidence = 0.72
        else:
            color_label = f"predominio {dominant_channel}"
            color_confidence = min(0.55 + (channel_spread / 255), 0.98)

        if aspect_ratio > 1.15:
            shape_label = "formato horizontal"
            shape_confidence = min(aspect_ratio / 2, 0.98)
        elif aspect_ratio < 0.87:
            shape_label = "formato vertical"
            shape_confidence = min((1 / aspect_ratio) / 2, 0.98)
        else:
            shape_label = "formato cuadrado"
            shape_confidence = 0.86

        predictions = [
            {"label": light_label, "confidence": round(light_confidence, 4), "type": "brightness"},
            {"label": color_label, "confidence": round(color_confidence, 4), "type": "dominant_color"},
            {"label": shape_label, "confidence": round(shape_confidence, 4), "type": "orientation"}
        ]

        return sorted(predictions, key=lambda item: item["confidence"], reverse=True)

    def _predict_ultralytics(
        self,
        model_config: dict[str, Any],
        pil_image: Image.Image,
        image_bytes: bytes
    ) -> list[dict[str, Any]]:
        artifact_path = self._resolve_artifact_path(model_config)
        confidence_threshold = float(model_config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        image_size = int(model_config.get("training", {}).get("image_size") or 640)

        try:
            return self._predict_ultralytics_in_process(
                model_config=model_config,
                artifact_path=artifact_path,
                pil_image=pil_image,
                confidence_threshold=confidence_threshold,
                image_size=image_size
            )
        except ImportError:
            return self._predict_ultralytics_subprocess(
                model_config=model_config,
                artifact_path=artifact_path,
                image_bytes=image_bytes,
                confidence_threshold=confidence_threshold,
                image_size=image_size
            )

    def _predict_ultralytics_in_process(
        self,
        model_config: dict[str, Any],
        artifact_path: Path,
        pil_image: Image.Image,
        confidence_threshold: float,
        image_size: int
    ) -> list[dict[str, Any]]:
        from ultralytics import YOLO

        cache_key = str(artifact_path)
        if cache_key not in self._ultralytics_models:
            self._ultralytics_models[cache_key] = YOLO(cache_key)

        model = self._ultralytics_models[cache_key]
        result = model.predict(
            source=pil_image,
            imgsz=image_size,
            conf=confidence_threshold,
            verbose=False
        )[0]
        return self._normalize_ultralytics_result(result, model_config)

    def _predict_ultralytics_subprocess(
        self,
        model_config: dict[str, Any],
        artifact_path: Path,
        image_bytes: bytes,
        confidence_threshold: float,
        image_size: int
    ) -> list[dict[str, Any]]:
        runner_path = PROJECT_ROOT / "ml" / "inference" / "yolo_predict.py"

        if not runner_path.exists():
            raise HTTPException(status_code=500, detail="No existe el runner de inferencia YOLO.")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as image_file:
            image_file.write(image_bytes)
            image_path = Path(image_file.name)

        try:
            return self._predict_ultralytics_worker(
                artifact_path=artifact_path,
                image_path=image_path,
                confidence_threshold=confidence_threshold,
                image_size=image_size
            )
        except HTTPException:
            raise
        except Exception:
            return self._predict_ultralytics_oneshot(
                runner_path=runner_path,
                artifact_path=artifact_path,
                image_path=image_path,
                confidence_threshold=confidence_threshold,
                image_size=image_size
            )
        finally:
            image_path.unlink(missing_ok=True)

    def _predict_ultralytics_worker(
        self,
        artifact_path: Path,
        image_path: Path,
        confidence_threshold: float,
        image_size: int
    ) -> list[dict[str, Any]]:
        worker = self._get_yolo_worker()
        request = {
            "model": str(artifact_path),
            "image": str(image_path),
            "conf": confidence_threshold,
            "imgsz": image_size
        }

        with self._yolo_worker_lock:
            if worker.stdin is None or worker.stdout is None:
                self._stop_yolo_worker()
                raise RuntimeError("El worker YOLO no tiene canales de comunicacion disponibles.")

            worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            worker.stdin.flush()
            response_line = worker.stdout.readline()

        if not response_line:
            self._stop_yolo_worker()
            raise RuntimeError("El worker YOLO no devolvio respuesta.")

        response = json.loads(response_line)
        if not response.get("ok"):
            raise HTTPException(status_code=500, detail=f"YOLO worker fallo: {response.get('error', 'error desconocido')}")

        predictions = response.get("predictions", [])
        if not isinstance(predictions, list):
            raise HTTPException(status_code=500, detail="YOLO worker devolvio predicciones con formato invalido.")

        return predictions

    def _predict_ultralytics_oneshot(
        self,
        runner_path: Path,
        artifact_path: Path,
        image_path: Path,
        confidence_threshold: float,
        image_size: int
    ) -> list[dict[str, Any]]:
        python_path = self._resolve_ml_python()

        try:
            completed = subprocess.run(
                [
                    str(python_path),
                    str(runner_path),
                    "--model",
                    str(artifact_path),
                    "--image",
                    str(image_path),
                    "--conf",
                    str(confidence_threshold),
                    "--imgsz",
                    str(image_size)
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=YOLO_INFERENCE_TIMEOUT_SECONDS,
                check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="La inferencia YOLO excedio el tiempo maximo.") from exc

        if completed.returncode != 0:
            error_detail = completed.stderr.strip() or completed.stdout.strip() or "Error desconocido ejecutando YOLO."
            raise HTTPException(status_code=500, detail=f"No se pudo ejecutar YOLO: {error_detail}")

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail="YOLO devolvio una respuesta no valida.") from exc

        predictions = payload.get("predictions", [])
        if not isinstance(predictions, list):
            raise HTTPException(status_code=500, detail="YOLO devolvio predicciones con formato invalido.")

        return predictions

    def _get_yolo_worker(self) -> subprocess.Popen[str]:
        if self._yolo_worker and self._yolo_worker.poll() is None:
            return self._yolo_worker

        worker_path = PROJECT_ROOT / "ml" / "inference" / "yolo_worker.py"
        if not worker_path.exists():
            raise HTTPException(status_code=500, detail="No existe el worker de inferencia YOLO.")

        self._yolo_worker = subprocess.Popen(
            [str(self._resolve_ml_python()), "-u", str(worker_path)],
            cwd=PROJECT_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        return self._yolo_worker

    def _stop_yolo_worker(self) -> None:
        if not self._yolo_worker:
            return

        if self._yolo_worker.poll() is None:
            self._yolo_worker.terminate()

        self._yolo_worker = None

    def _predict_torchvision(self, model_config: dict[str, Any], image_bytes: bytes) -> list[dict[str, Any]]:
        artifact_path = self._resolve_artifact_path(model_config)
        confidence_threshold = float(model_config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        max_detections = int(model_config.get("max_detections", 50))
        nms_threshold = float(model_config.get("nms_threshold", 0.5))
        detections_per_img = int(model_config.get("detections_per_img", max_detections))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as image_file:
            image_file.write(image_bytes)
            image_path = Path(image_file.name)

        try:
            worker = self._get_torchvision_worker()
            request = {
                "model_id": model_config.get("id"),
                "model": str(artifact_path),
                "image": str(image_path),
                "conf": confidence_threshold,
                "max_detections": max_detections,
                "nms_threshold": nms_threshold,
                "detections_per_img": detections_per_img
            }

            with self._torchvision_worker_lock:
                if worker.stdin is None or worker.stdout is None:
                    self._stop_torchvision_worker()
                    raise RuntimeError("El worker TorchVision no tiene canales de comunicacion disponibles.")

                worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                worker.stdin.flush()
                response_line = worker.stdout.readline()

            if not response_line:
                self._stop_torchvision_worker()
                raise RuntimeError("El worker TorchVision no devolvio respuesta.")

            response = json.loads(response_line)
            if not response.get("ok"):
                raise HTTPException(status_code=500, detail=f"TorchVision worker fallo: {response.get('error', 'error desconocido')}")

            predictions = response.get("predictions", [])
            if not isinstance(predictions, list):
                raise HTTPException(status_code=500, detail="TorchVision worker devolvio predicciones con formato invalido.")

            return predictions
        finally:
            image_path.unlink(missing_ok=True)

    def _get_torchvision_worker(self) -> subprocess.Popen[str]:
        if self._torchvision_worker and self._torchvision_worker.poll() is None:
            return self._torchvision_worker

        worker_path = PROJECT_ROOT / "ml" / "inference" / "torchvision_worker.py"
        if not worker_path.exists():
            raise HTTPException(status_code=500, detail="No existe el worker de inferencia TorchVision.")

        self._torchvision_worker = subprocess.Popen(
            [str(self._resolve_ml_python()), "-u", str(worker_path)],
            cwd=PROJECT_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        return self._torchvision_worker

    def _stop_torchvision_worker(self) -> None:
        if not self._torchvision_worker:
            return

        if self._torchvision_worker.poll() is None:
            self._torchvision_worker.terminate()

        self._torchvision_worker = None

    def _normalize_ultralytics_result(self, result: Any, model_config: dict[str, Any]) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return predictions

        names = getattr(result, "names", None) or {}

        for index, box in enumerate(boxes):
            xyxy = box.xyxy[0].detach().cpu().tolist()
            confidence = float(box.conf[0].detach().cpu())
            class_id = int(box.cls[0].detach().cpu())
            label = names.get(class_id) or self._label_from_config(model_config, class_id)
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
                "model": model_config.get("id"),
                "class_id": class_id,
                "rank": index + 1,
                "bbox": [
                    round(float(x_min), 2),
                    round(float(y_min), 2),
                    round(width, 2),
                    round(height, 2)
                ],
                "details": f"Clase COCO #{class_id}"
            })

        return sorted(predictions, key=lambda item: item["confidence"], reverse=True)

    def _resolve_artifact_path(self, model_config: dict[str, Any]) -> Path:
        artifact_path = model_config.get("artifact_path")
        if not artifact_path:
            raise HTTPException(status_code=500, detail=f"El modelo '{model_config.get('id')}' no tiene artifact_path.")

        path = Path(artifact_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            raise HTTPException(status_code=500, detail=f"No existe el artefacto del modelo: {path}")

        return path

    def _resolve_ml_python(self) -> Path:
        configured_python = os.environ.get("CIML_PYTHON")
        candidates = [
            Path(configured_python) if configured_python else None,
            Path.home() / ".ciml" / "venv" / "Scripts" / "python.exe",
            Path.home() / ".ciml" / "venv" / "bin" / "python",
            Path(sys.executable)
        ]

        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate

        raise HTTPException(
            status_code=500,
            detail="No se encontro un Python con dependencias ML. Configura CIML_PYTHON o ejecuta scripts/train-local-models.ps1 -Install."
        )

    def _label_from_config(self, model_config: dict[str, Any], class_id: int) -> str:
        labels = model_config.get("labels") or []
        if 0 <= class_id < len(labels):
            return str(labels[class_id])
        return f"clase {class_id}"
