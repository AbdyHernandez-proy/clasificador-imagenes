from __future__ import annotations

import json
import shutil
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.paths import DATASET_DOWNLOADS_ROOT, DATASET_RAW_ROOT, DATASET_REGISTRY_PATH, PROJECT_ROOT


class DatasetRegistry:
    def __init__(self, registry_path: Path = DATASET_REGISTRY_PATH):
        self.registry_path = registry_path

    def load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"schema_version": 1, "datasets": []}

        with self.registry_path.open("r", encoding="utf-8-sig") as registry_file:
            return json.load(registry_file)

    def save(self, registry: dict[str, Any]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("w", encoding="utf-8") as registry_file:
            json.dump(registry, registry_file, ensure_ascii=False, indent=2)
            registry_file.write("\n")

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.load().get("datasets", [])

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        for dataset in self.list_datasets():
            if dataset.get("id") == dataset_id:
                return dataset
        return None

    def ensure_storage(self) -> None:
        DATASET_DOWNLOADS_ROOT.mkdir(parents=True, exist_ok=True)
        DATASET_RAW_ROOT.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_id: str) -> dict[str, Any]:
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset no registrado: {dataset_id}")

        download = dataset.get("download") or {}
        url = download.get("url")
        if not url:
            raise ValueError(f"El dataset '{dataset_id}' no tiene URL de descarga automatica.")

        self.ensure_storage()
        archive_path = DATASET_DOWNLOADS_ROOT / f"{dataset_id}{self._archive_suffix(url)}"
        target_path = DATASET_RAW_ROOT / dataset_id

        if not archive_path.exists():
            urllib.request.urlretrieve(url, archive_path)

        if target_path.exists():
            shutil.rmtree(target_path)
        target_path.mkdir(parents=True, exist_ok=True)

        self._extract_archive(archive_path, target_path, download.get("archive_type"))

        return self.mark_downloaded(
            dataset_id=dataset_id,
            archive_path=archive_path,
            local_path=target_path
        )

    def mark_downloaded(self, dataset_id: str, archive_path: Path, local_path: Path) -> dict[str, Any]:
        registry = self.load()
        now = datetime.now(timezone.utc).isoformat()
        updated_dataset: dict[str, Any] | None = None

        for dataset in registry.get("datasets", []):
            if dataset.get("id") != dataset_id:
                continue

            dataset["status"] = "downloaded"
            dataset["local_path"] = self._relative_path(local_path)
            dataset["dataset_path"] = self._relative_path(self._find_dataset_root(local_path))
            dataset["archive_path"] = self._relative_path(archive_path)
            dataset["downloaded_at"] = now
            updated_dataset = dataset
            break

        if not updated_dataset:
            raise ValueError(f"Dataset no registrado: {dataset_id}")

        self.save(registry)
        return updated_dataset

    def _extract_archive(self, archive_path: Path, target_path: Path, archive_type: str | None = None) -> None:
        normalized_type = (archive_type or archive_path.suffix.lstrip(".")).lower()

        if normalized_type == "zip":
            with zipfile.ZipFile(archive_path) as archive:
                self._safe_extract_zip(archive, target_path)
            return

        if normalized_type in {"tar", "gz", "tgz", "tar.gz"}:
            with tarfile.open(archive_path) as archive:
                self._safe_extract_tar(archive, target_path)
            return

        raise ValueError(f"Tipo de archivo no soportado: {normalized_type}")

    def _safe_extract_zip(self, archive: zipfile.ZipFile, target_path: Path) -> None:
        target_root = target_path.resolve()
        for member in archive.infolist():
            member_path = (target_path / member.filename).resolve()
            if not self._is_relative_to(member_path, target_root):
                raise ValueError(f"Ruta insegura dentro del ZIP: {member.filename}")
        archive.extractall(target_path)

    def _safe_extract_tar(self, archive: tarfile.TarFile, target_path: Path) -> None:
        target_root = target_path.resolve()
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Enlace no permitido dentro del TAR: {member.name}")
            member_path = (target_path / member.name).resolve()
            if not self._is_relative_to(member_path, target_root):
                raise ValueError(f"Ruta insegura dentro del TAR: {member.name}")
        archive.extractall(target_path)

    def _is_relative_to(self, path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def _archive_suffix(self, url: str) -> str:
        filename = url.rstrip("/").split("/")[-1]
        if "." not in filename:
            return ".download"
        return "." + filename.split(".", 1)[1]

    def _find_dataset_root(self, local_path: Path) -> Path:
        children = [child for child in local_path.iterdir() if child.is_dir()]
        if len(children) == 1:
            return children[0]
        return local_path

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
        except ValueError:
            return str(path)
