from __future__ import annotations

import io
import time
from typing import Any

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageStat, UnidentifiedImageError

from app.services.model_registry import ModelRegistry

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class InferenceService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

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
