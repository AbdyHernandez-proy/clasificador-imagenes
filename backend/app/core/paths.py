from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_ROOT = PROJECT_ROOT / "ml"
REGISTRY_PATH = ML_ROOT / "registry.json"
MODEL_ARTIFACTS_ROOT = ML_ROOT / "models"
DATASETS_ROOT = ML_ROOT / "datasets"
DATASET_DOWNLOADS_ROOT = DATASETS_ROOT / "downloads"
DATASET_RAW_ROOT = DATASETS_ROOT / "raw"
DATASET_PROCESSED_ROOT = DATASETS_ROOT / "processed"
DATASET_REGISTRY_PATH = DATASETS_ROOT / "registry.json"
