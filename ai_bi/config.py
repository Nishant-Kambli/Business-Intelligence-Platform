"""Shared paths and dataset configuration contracts."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "IBM_HR_Cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "attrition_model.pkl"
EVALUATION_PATH = PROJECT_ROOT / "artifacts" / "model_evaluation.json"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "artifacts" / "feature_importance.csv"
LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"
STYLES_PATH = PROJECT_ROOT / "assets" / "styles.css"

POSITIVE_LABEL = "Yes"

APP_TITLE = "People Intelligence"
APP_SUBTITLE = "Enterprise workforce intelligence"

DATASET_CONFIG_DIR = PROJECT_ROOT / "ai_bi" / "datasets"
MODEL_REGISTRY_DIR = PROJECT_ROOT / "ai_bi" / "models"
TRAINED_MODELS_DIR = PROJECT_ROOT / "models"
MODULE_REGISTRY_PATH = PROJECT_ROOT / "ai_bi" / "modules.json"


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-specific schema and module metadata loaded from JSON."""

    dataset_id: str
    dataset_name: str
    target_column: str | None
    employee_id_column: str | None
    department_column: str | None
    role_column: str | None
    salary_column: str | None
    satisfaction_columns: tuple[str, ...]
    experience_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    supported_modules: tuple[str, ...]
    detection_fields: tuple[str, ...]
    aliases: dict[str, tuple[str, ...]]
    module_requirements: dict[str, tuple[str, ...]]
    source_path: Path | None = None
    minimum_detection_fields: int = 0
    module_any_requirements: dict[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    date_columns: tuple[str, ...] = ()
    target_positive_values: tuple[str, ...] = ("yes", "1", "true")
    target_negative_values: tuple[str, ...] = ("no", "0", "false")
