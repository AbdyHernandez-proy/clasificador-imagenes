from __future__ import annotations

import argparse
import json

from ml.training.dataset_catalog import get_dataset, list_datasets, resolve_project_path
from ml.training.datasets.yolo_detection import summarize_yolo_dataset
from ml.training.detectors import build_detector_plan
from ml.training.model_zoo import compatible_detectors_for_dataset, get_detector_config, load_detector_configs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI base para modelos propios.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models", help="Lista detectores propios disponibles como scaffold.")
    subparsers.add_parser("list-datasets", help="Lista datasets registrados.")

    show_model = subparsers.add_parser("show-model", help="Muestra la configuracion de un detector.")
    show_model.add_argument("model_id")

    plan = subparsers.add_parser("plan", help="Genera un plan de entrenamiento sin entrenar.")
    plan.add_argument("model_id")
    plan.add_argument("--dataset-id", default=None)

    validate = subparsers.add_parser("validate-dataset", help="Valida lectura basica de un dataset.")
    validate.add_argument("dataset_id")
    validate.add_argument("--split", default="train")

    compatible = subparsers.add_parser("compatible-models", help="Lista modelos compatibles con un dataset.")
    compatible.add_argument("dataset_id")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "list-models":
        rows = [
            {
                "id": config.get("id"),
                "name": config.get("name"),
                "framework": config.get("framework"),
                "status": config.get("status"),
                "default_dataset_id": config.get("default_dataset_id")
            }
            for config in load_detector_configs()
        ]
        print_json(rows)
        return

    if args.command == "list-datasets":
        rows = [
            {
                "id": dataset.get("id"),
                "task": dataset.get("task"),
                "format": dataset.get("format"),
                "status": dataset.get("status"),
                "dataset_path": dataset.get("dataset_path")
            }
            for dataset in list_datasets()
        ]
        print_json(rows)
        return

    if args.command == "show-model":
        config = get_detector_config(args.model_id)
        if not config:
            raise SystemExit(f"Modelo no encontrado: {args.model_id}")
        print_json(config)
        return

    if args.command == "plan":
        print_json(build_detector_plan(args.model_id, args.dataset_id))
        return

    if args.command == "validate-dataset":
        dataset = get_dataset(args.dataset_id)
        if not dataset:
            raise SystemExit(f"Dataset no encontrado: {args.dataset_id}")

        if dataset.get("format") != "yolo":
            raise SystemExit(f"Validador no implementado para formato: {dataset.get('format')}")

        dataset_root = resolve_project_path(dataset.get("dataset_path"))
        split_path = dataset.get("splits", {}).get(args.split)
        if not dataset_root or not split_path:
            raise SystemExit("El dataset no tiene dataset_path o split configurado.")

        print_json(summarize_yolo_dataset(dataset_root, split_path))
        return

    if args.command == "compatible-models":
        dataset = get_dataset(args.dataset_id)
        if not dataset:
            raise SystemExit(f"Dataset no encontrado: {args.dataset_id}")
        print_json(compatible_detectors_for_dataset(dataset))
        return


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
