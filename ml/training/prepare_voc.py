from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from ml.training.paths import DATASET_REGISTRY_PATH, PROJECT_ROOT

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_ARCHIVES = [
    {
        "filename": "VOCtrainval_06-Nov-2007.zip",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_06-Nov-2007.zip",
    },
    {
        "filename": "VOCtest_06-Nov-2007.zip",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtest_06-Nov-2007.zip",
    },
    {
        "filename": "VOCtrainval_11-May-2012.zip",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_11-May-2012.zip",
    },
]

SPLITS = [
    ("2007", "trainval", "train"),
    ("2012", "trainval", "train"),
    ("2007", "test", "val"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga y prepara PASCAL VOC en formato YOLO.")
    parser.add_argument("--dataset-id", default="voc-detect")
    parser.add_argument("--force", action="store_true", help="Recrea conversion/splits aunque ya existan.")
    parser.add_argument("--skip-download", action="store_true", help="Usa archivos ya descargados en ml/datasets/downloads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = PROJECT_ROOT / "ml" / "datasets" / "raw" / args.dataset_id / "VOC"
    downloads_root = PROJECT_ROOT / "ml" / "datasets" / "downloads"
    downloads_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    archives = download_archives(downloads_root, skip_download=args.skip_download)
    extract_archives(archives, dataset_root)
    summary = convert_voc_to_yolo(dataset_root, force=args.force)
    update_dataset_registry(args.dataset_id, dataset_root, archives, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def download_archives(downloads_root: Path, skip_download: bool) -> list[Path]:
    archive_paths: list[Path] = []
    for archive in VOC_ARCHIVES:
        archive_path = downloads_root / archive["filename"]
        archive_paths.append(archive_path)

        if archive_path.exists():
            continue

        if skip_download:
            raise FileNotFoundError(f"No existe {archive_path} y se indico --skip-download.")

        print(f"Descargando {archive['filename']}...")
        urllib.request.urlretrieve(archive["url"], archive_path)

    return archive_paths


def extract_archives(archive_paths: list[Path], dataset_root: Path) -> None:
    for archive_path in archive_paths:
        marker_path = dataset_root / f".extracted-{archive_path.stem}"
        if marker_path.exists():
            continue

        print(f"Extrayendo {archive_path.name}...")
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract_zip(archive, dataset_root)
        marker_path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


def safe_extract_zip(archive: zipfile.ZipFile, target_path: Path) -> None:
    target_root = target_path.resolve()
    for member in archive.infolist():
        member_path = (target_path / member.filename).resolve()
        if not is_relative_to(member_path, target_root):
            raise ValueError(f"Ruta insegura dentro del ZIP: {member.filename}")
    archive.extractall(target_path)


def convert_voc_to_yolo(dataset_root: Path, force: bool) -> dict[str, Any]:
    split_files = {
        "train": dataset_root / "splits" / "train.txt",
        "val": dataset_root / "splits" / "val.txt",
    }
    if not force and all(path.exists() for path in split_files.values()):
        return summarize_prepared_dataset(dataset_root)

    for path in split_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    split_images: dict[str, list[str]] = {"train": [], "val": []}
    total_labels = 0
    total_boxes = 0

    for year, voc_split, output_split in SPLITS:
        voc_root = dataset_root / "VOCdevkit" / f"VOC{year}"
        image_set_path = voc_root / "ImageSets" / "Main" / f"{voc_split}.txt"
        if not image_set_path.exists():
            raise FileNotFoundError(f"No existe split VOC: {image_set_path}")

        for image_id in read_image_ids(image_set_path):
            image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
            annotation_path = voc_root / "Annotations" / f"{image_id}.xml"
            label_path = voc_root / "labels" / f"{image_id}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)

            box_count = convert_annotation(annotation_path, label_path)
            total_labels += 1
            total_boxes += box_count
            split_images[output_split].append(relative_project_path(image_path, dataset_root))

    for split_name, images in split_images.items():
        split_files[split_name].write_text("\n".join(sorted(images)) + "\n", encoding="utf-8")

    return {
        "dataset_id": "voc-detect",
        "classes": len(VOC_CLASSES),
        "train_images": len(split_images["train"]),
        "val_images": len(split_images["val"]),
        "labels": total_labels,
        "boxes": total_boxes,
        "dataset_path": relative_project_path(dataset_root, PROJECT_ROOT),
    }


def convert_annotation(annotation_path: Path, label_path: Path) -> int:
    tree = ElementTree.parse(annotation_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Anotacion sin size: {annotation_path}")

    image_width = int(size.findtext("width", "0"))
    image_height = int(size.findtext("height", "0"))
    lines: list[str] = []

    for obj in root.iter("object"):
        class_name = obj.findtext("name")
        difficult = int(obj.findtext("difficult", "0"))
        if difficult == 1 or class_name not in VOC_CLASSES:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))
        yolo_box = voc_box_to_yolo(image_width, image_height, xmin, ymin, xmax, ymax)
        if not yolo_box:
            continue

        class_id = VOC_CLASSES.index(class_name)
        lines.append(" ".join([str(class_id), *(f"{value:.6f}" for value in yolo_box)]))

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def voc_box_to_yolo(
    image_width: int,
    image_height: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[float, float, float, float] | None:
    if image_width <= 0 or image_height <= 0:
        return None

    xmin = max(0.0, min(xmin, float(image_width)))
    ymin = max(0.0, min(ymin, float(image_height)))
    xmax = max(0.0, min(xmax, float(image_width)))
    ymax = max(0.0, min(ymax, float(image_height)))
    box_width = xmax - xmin
    box_height = ymax - ymin
    if box_width <= 0 or box_height <= 0:
        return None

    x_center = (xmin + xmax) / 2 / image_width
    y_center = (ymin + ymax) / 2 / image_height
    return (
        x_center,
        y_center,
        box_width / image_width,
        box_height / image_height,
    )


def summarize_prepared_dataset(dataset_root: Path) -> dict[str, Any]:
    train_path = dataset_root / "splits" / "train.txt"
    val_path = dataset_root / "splits" / "val.txt"
    return {
        "dataset_id": "voc-detect",
        "classes": len(VOC_CLASSES),
        "train_images": count_lines(train_path),
        "val_images": count_lines(val_path),
        "dataset_path": relative_project_path(dataset_root, PROJECT_ROOT),
        "status": "already_prepared",
    }


def update_dataset_registry(
    dataset_id: str,
    dataset_root: Path,
    archives: list[Path],
    summary: dict[str, Any],
) -> None:
    registry = json.loads(DATASET_REGISTRY_PATH.read_text(encoding="utf-8"))
    datasets = registry.setdefault("datasets", [])
    now = datetime.now(timezone.utc).isoformat()
    dataset = next((item for item in datasets if item.get("id") == dataset_id), None)

    next_dataset = {
        "id": dataset_id,
        "name": "PASCAL VOC",
        "task": "object_detection",
        "format": "yolo",
        "status": "downloaded",
        "size_mb": 2800,
        "class_count": len(VOC_CLASSES),
        "classes": VOC_CLASSES,
        "source": {
            "type": "public_dataset",
            "homepage": "https://docs.ultralytics.com/datasets/detect/voc/",
            "license": "PASCAL VOC terms; conversion/download metadata follows Ultralytics assets/config.",
        },
        "download": {
            "urls": [archive["url"] for archive in VOC_ARCHIVES],
            "archive_type": "zip",
        },
        "splits": {
            "train": "splits/train.txt",
            "val": "splits/val.txt",
        },
        "compatible_model_ids": [
            "custom-yolo-v8-v11-detector",
            "custom-faster-rcnn-detector",
            "custom-retinanet-detector",
        ],
        "description": "Dataset PASCAL VOC convertido a formato YOLO para deteccion general de 20 clases comunes.",
        "recommended_use": "Siguiente fase de entrenamiento para Faster R-CNN y RetinaNet antes de datasets mas grandes como COCO completo.",
        "local_path": relative_project_path(dataset_root.parent, PROJECT_ROOT),
        "dataset_path": relative_project_path(dataset_root, PROJECT_ROOT),
        "archive_paths": [relative_project_path(path, PROJECT_ROOT) for path in archives],
        "downloaded_at": now,
        "summary": summary,
    }

    if dataset is None:
        datasets.append(next_dataset)
    else:
        dataset.clear()
        dataset.update(next_dataset)

    DATASET_REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_image_ids(path: Path) -> list[str]:
    return [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def relative_project_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
