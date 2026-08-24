from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ml"
MODEL_REGISTRY_PATH = ML_ROOT / "registry.json"
DATASET_REGISTRY_PATH = ML_ROOT / "datasets" / "registry.json"
TRAINING_ROOT = ML_ROOT / "training"
MODEL_CONFIGS_ROOT = TRAINING_ROOT / "configs"
MODEL_ARTIFACTS_ROOT = ML_ROOT / "models"
RUNS_ROOT = ML_ROOT / "runs"
