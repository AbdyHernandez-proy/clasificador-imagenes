from __future__ import annotations

import argparse
import json

from app.services.dataset_registry import DatasetRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gestion interna de datasets del proyecto.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="Lista los datasets registrados.")

    show_parser = subparsers.add_parser("show", help="Muestra el detalle de un dataset.")
    show_parser.add_argument("dataset_id")

    download_parser = subparsers.add_parser("download", help="Descarga y extrae un dataset registrado.")
    download_parser.add_argument("dataset_id")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry = DatasetRegistry()

    if args.command == "list":
        datasets = registry.list_datasets()
        rows = [
            {
                "id": dataset.get("id"),
                "task": dataset.get("task"),
                "format": dataset.get("format"),
                "status": dataset.get("status")
            }
            for dataset in datasets
        ]
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if args.command == "show":
        dataset = registry.get_dataset(args.dataset_id)
        if not dataset:
            raise SystemExit(f"Dataset no encontrado: {args.dataset_id}")
        print(json.dumps(dataset, ensure_ascii=False, indent=2))
        return

    if args.command == "download":
        dataset = registry.download_dataset(args.dataset_id)
        print(json.dumps(dataset, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
