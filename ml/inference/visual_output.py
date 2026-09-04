from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

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
    image_key: str
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


def normalize_predictions(image_key: str, predictions: list[dict[str, Any]]) -> list[Prediction]:
    normalized: list[Prediction] = []
    for prediction in predictions:
        bbox = prediction.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x_min, y_min, width, height = [float(value) for value in bbox]
        if width <= 0 or height <= 0:
            continue
        label = str(prediction.get("label") or prediction.get("className") or prediction.get("class") or "")
        confidence = float(prediction.get("confidence") or prediction.get("score") or 0)
        normalized.append(
            Prediction(
                image_key=image_key,
                label=label,
                confidence=confidence,
                bbox=(x_min, y_min, x_min + width, y_min + height),
            )
        )
    return sorted(normalized, key=lambda item: item.confidence, reverse=True)


def save_annotated_image(
    source_path: Path,
    output_path: Path,
    predictions: list[Prediction],
    title: str,
) -> None:
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
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"#{index} {prediction.label} {prediction.confidence:.2f}"
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
