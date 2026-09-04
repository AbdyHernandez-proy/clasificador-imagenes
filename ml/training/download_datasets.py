from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ml.training.dataset_catalog import load_dataset_registry, resolve_project_path
from ml.training.paths import DATASET_REGISTRY_PATH, PROJECT_ROOT
from ml.training.prepare_voc import (
    VOC_ARCHIVES,
    convert_voc_to_yolo,
    download_archives as download_voc_archives,
    extract_archives as extract_voc_archives,
    update_dataset_registry as update_voc_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga datasets registrados para entrenamiento/evaluacion.")
    parser.add_argument("--dataset-id", default="voc-detect", help="Dataset a descargar. Usa 'all' para todo el catalogo.")
    parser.add_argument("--force", action="store_true", help="Recrea el dataset local si ya existe.")
    parser.add_argument("--skip-existing", action="store_true", help="No descarga ni extrae archivos ya presentes.")
    parser.add_argument("--keep-archives", action="store_true", help="Conserva los ZIP/TAR descargados despues de extraer.")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Split train usado por VOC.")
    parser.add_argument("--split-seed", type=int, default=20260827, help="Semilla del split train/val usado por VOC.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_dataset_registry()
    datasets = registry.get("datasets", [])
    selected = datasets if args.dataset_id == "all" else [find_dataset(datasets, args.dataset_id)]

    results = []
    for dataset in selected:
        if dataset["id"] == "voc-detect":
            results.append(prepare_voc(args))
        else:
            results.append(download_generic_dataset(dataset=dataset, args=args))

    print(json.dumps({"status": "ok", "datasets": results}, ensure_ascii=False, indent=2), flush=True)


def prepare_voc(args: argparse.Namespace) -> dict[str, Any]:
    dataset_id = "voc-detect"
    dataset_root = PROJECT_ROOT / "ml" / "datasets" / "raw" / dataset_id / "VOC"
    downloads_root = PROJECT_ROOT / "ml" / "datasets" / "downloads"

    if args.force and dataset_root.parent.exists():
        remove_inside_datasets(dataset_root.parent)

    downloads_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    archives = download_voc_archives(downloads_root, skip_download=False)
    extract_voc_archives(archives, dataset_root)
    summary = convert_voc_to_yolo(
        dataset_root=dataset_root,
        force=args.force,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
    )
    update_voc_registry(dataset_id, dataset_root, archives if args.keep_archives else [], summary)

    if not args.keep_archives:
        remove_archives(archives)

    return {
        "id": dataset_id,
        "status": "downloaded",
        "dataset_path": relative_project_path(dataset_root),
        "archives_kept": bool(args.keep_archives),
        "summary": summary,
    }


def download_generic_dataset(dataset: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset_id = str(dataset["id"])
    dataset_root = resolve_project_path(str(dataset.get("local_path") or f"ml/datasets/raw/{dataset_id}"))
    if dataset_root is None:
        raise ValueError(f"No se pudo resolver local_path para {dataset_id}.")

    if args.force and dataset_root.exists():
        remove_inside_datasets(dataset_root)

    dataset_root.mkdir(parents=True, exist_ok=True)
    downloads_root = PROJECT_ROOT / "ml" / "datasets" / "downloads"
    downloads_root.mkdir(parents=True, exist_ok=True)

    archive_paths = []
    for url in dataset_download_urls(dataset):
        archive_path = downloads_root / filename_from_url(url)
        archive_paths.append(archive_path)
        if archive_path.exists() and args.skip_existing:
            print(f"Usando descarga existente: {archive_path.name}", flush=True)
        elif not archive_path.exists():
            print(f"Descargando {archive_path.name}...", flush=True)
            urllib.request.urlretrieve(url, archive_path)

        marker_path = dataset_root / f".extracted-{archive_path.stem}"
        if marker_path.exists() and args.skip_existing:
            continue
        print(f"Extrayendo {archive_path.name}...", flush=True)
        extract_archive_safely(archive_path, dataset_root)
        marker_path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    if not args.keep_archives:
        remove_archives(archive_paths)

    update_generic_dataset_registry(dataset_id=dataset_id, dataset_root=dataset_root, archive_paths=archive_paths, args=args)
    return {
        "id": dataset_id,
        "status": "downloaded",
        "local_path": relative_project_path(dataset_root),
        "archives_kept": bool(args.keep_archives),
    }


def dataset_download_urls(dataset: dict[str, Any]) -> list[str]:
    download = dataset.get("download") or {}
    urls = []
    if download.get("url"):
        urls.append(str(download["url"]))
    if download.get("annotations_url"):
        urls.append(str(download["annotations_url"]))
    urls.extend(str(url) for url in download.get("urls") or [])
    if not urls:
        raise ValueError(f"El dataset {dataset.get('id')} no tiene URLs de descarga registradas.")
    return urls


def extract_archive_safely(archive_path: Path, target_path: Path) -> None:
    suffixes = "".join(archive_path.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract_zip(archive, target_path)
        return
    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or suffixes.endswith(".tar"):
        with tarfile.open(archive_path) as archive:
            safe_extract_tar(archive, target_path)
        return
    raise ValueError(f"Tipo de archivo no soportado: {archive_path}")


def safe_extract_zip(archive: zipfile.ZipFile, target_path: Path) -> None:
    target_root = target_path.resolve()
    for member in archive.infolist():
        member_path = (target_path / member.filename).resolve()
        if not is_relative_to(member_path, target_root):
            raise ValueError(f"Ruta insegura dentro del ZIP: {member.filename}")
    archive.extractall(target_path)


def safe_extract_tar(archive: tarfile.TarFile, target_path: Path) -> None:
    target_root = target_path.resolve()
    for member in archive.getmembers():
        member_path = (target_path / member.name).resolve()
        if not is_relative_to(member_path, target_root):
            raise ValueError(f"Ruta insegura dentro del TAR: {member.name}")
    archive.extractall(target_path)


def update_generic_dataset_registry(
    dataset_id: str,
    dataset_root: Path,
    archive_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    registry = json.loads(DATASET_REGISTRY_PATH.read_text(encoding="utf-8"))
    dataset = find_dataset(registry.get("datasets", []), dataset_id)
    dataset["status"] = "downloaded"
    dataset["local_path"] = relative_project_path(dataset_root)
    dataset["downloaded_at"] = datetime.now(timezone.utc).isoformat()
    if args.keep_archives:
        if len(archive_paths) == 1:
            dataset["archive_path"] = relative_project_path(archive_paths[0])
        else:
            dataset["archive_paths"] = [relative_project_path(path) for path in archive_paths]
    else:
        dataset.pop("archive_path", None)
        dataset.pop("archive_paths", None)
    DATASET_REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remove_archives(archive_paths: list[Path]) -> None:
    for archive_path in archive_paths:
        if archive_path.exists():
            remove_inside_datasets(archive_path)


def remove_inside_datasets(path: Path) -> None:
    datasets_root = (PROJECT_ROOT / "ml" / "datasets").resolve()
    target = path.resolve()
    if not is_relative_to(target, datasets_root):
        raise ValueError(f"Se intento eliminar una ruta fuera de ml/datasets: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def find_dataset(datasets: list[dict[str, Any]], dataset_id: str) -> dict[str, Any]:
    for dataset in datasets:
        if dataset.get("id") == dataset_id:
            return dataset
    raise KeyError(f"No existe el dataset registrado: {dataset_id}")


def filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"No se pudo resolver nombre de archivo desde URL: {url}")
    return name


def relative_project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
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
