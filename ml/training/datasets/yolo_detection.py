from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class DetectionTarget:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    label_path: Path
    width: int
    height: int
    targets: list[DetectionTarget]


def build_yolo_index(dataset_root: Path, image_split: str) -> list[DetectionSample]:
    cache_path = yolo_index_cache_path(dataset_root, image_split)
    if cache_path.exists():
        try:
            return read_yolo_index_cache(cache_path)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            cache_path.unlink(missing_ok=True)

    split_path = dataset_root / image_split

    if split_path.is_file():
        image_paths = read_image_list(dataset_root, split_path)
        image_dir = None
        label_dir = None
    else:
        image_dir = split_path
        label_dir = dataset_root / image_split.replace("images", "labels", 1)

        if not image_dir.exists():
            raise FileNotFoundError(f"No existe el directorio de imagenes: {image_dir}")

        if not label_dir.exists():
            raise FileNotFoundError(f"No existe el directorio de etiquetas: {label_dir}")

        image_paths = sorted(
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    samples: list[DetectionSample] = []
    total_images = len(image_paths)
    print(f"Indexando dataset YOLO: 0/{total_images}", flush=True)
    for index, image_path in enumerate(image_paths, start=1):
        label_path = resolve_label_path(
            dataset_root=dataset_root,
            image_path=image_path,
            image_dir=image_dir,
            label_dir=label_dir
        )
        with Image.open(image_path) as image:
            width, height = image.size
        samples.append(
            DetectionSample(
                image_path=image_path,
                label_path=label_path,
                width=width,
                height=height,
                targets=read_yolo_labels(label_path)
            )
        )
        if index % 1000 == 0 or index == total_images:
            print(f"Indexando dataset YOLO: {index}/{total_images}", flush=True)

    write_yolo_index_cache(cache_path, samples)
    return samples


def yolo_index_cache_path(dataset_root: Path, image_split: str) -> Path:
    split_path = dataset_root / image_split
    cache_source = image_split.encode("utf-8")
    if split_path.is_file():
        cache_source += b"\0" + split_path.read_bytes()
    cache_key = hashlib.sha1(cache_source).hexdigest()[:12]
    return dataset_root / ".cache" / f"yolo_index_{cache_key}.json"


def read_yolo_index_cache(cache_path: Path) -> list[DetectionSample]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Version de cache no soportada.")

    samples = []
    for raw_sample in payload["samples"]:
        samples.append(
            DetectionSample(
                image_path=Path(raw_sample["image_path"]),
                label_path=Path(raw_sample["label_path"]),
                width=int(raw_sample["width"]),
                height=int(raw_sample["height"]),
                targets=[
                    DetectionTarget(
                        class_id=int(raw_target["class_id"]),
                        x_center=float(raw_target["x_center"]),
                        y_center=float(raw_target["y_center"]),
                        width=float(raw_target["width"]),
                        height=float(raw_target["height"])
                    )
                    for raw_target in raw_sample["targets"]
                ]
            )
        )
    print(f"Indice YOLO cargado desde cache: {len(samples)} muestras.", flush=True)
    return samples


def write_yolo_index_cache(cache_path: Path, samples: list[DetectionSample]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "samples": [
            {
                "image_path": str(sample.image_path),
                "label_path": str(sample.label_path),
                "width": sample.width,
                "height": sample.height,
                "targets": [
                    {
                        "class_id": target.class_id,
                        "x_center": target.x_center,
                        "y_center": target.y_center,
                        "width": target.width,
                        "height": target.height
                    }
                    for target in sample.targets
                ]
            }
            for sample in samples
        ]
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read_image_list(dataset_root: Path, split_path: Path) -> list[Path]:
    image_paths: list[Path] = []
    for raw_line in split_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        image_path = Path(line)
        if not image_path.is_absolute():
            image_path = dataset_root / image_path

        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths.append(image_path)

    return sorted(image_paths)


def resolve_label_path(
    dataset_root: Path,
    image_path: Path,
    image_dir: Path | None,
    label_dir: Path | None
) -> Path:
    if image_dir and label_dir:
        return label_dir / image_path.relative_to(image_dir).with_suffix(".txt")

    try:
        relative_path = image_path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        relative_path = image_path

    parts = list(relative_path.parts)
    if "JPEGImages" in parts:
        parts[parts.index("JPEGImages")] = "labels"
        return dataset_root / Path(*parts).with_suffix(".txt")

    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return dataset_root / Path(*parts).with_suffix(".txt")

    return image_path.with_suffix(".txt")


def read_yolo_labels(label_path: Path) -> list[DetectionTarget]:
    if not label_path.exists():
        return []

    targets: list[DetectionTarget] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Etiqueta YOLO invalida en {label_path}:{line_number}")

        class_id = int(float(parts[0]))
        x_center, y_center, width, height = (float(value) for value in parts[1:])
        targets.append(
            DetectionTarget(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height
            )
        )

    return targets


def summarize_yolo_dataset(dataset_root: Path, image_split: str) -> dict[str, int]:
    samples = build_yolo_index(dataset_root=dataset_root, image_split=image_split)
    return {
        "images": len(samples),
        "labels": sum(1 for sample in samples if sample.label_path.exists()),
        "boxes": sum(len(sample.targets) for sample in samples)
    }
