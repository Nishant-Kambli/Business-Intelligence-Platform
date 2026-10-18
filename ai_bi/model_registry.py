"""Configuration-driven registry for independent prediction models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import MODEL_REGISTRY_DIR, PROJECT_ROOT, TRAINED_MODELS_DIR


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    task: str
    target: str
    artifact: str
    active: bool
    required_fields: tuple[str, ...]
    required_source_columns: tuple[str, ...]
    positive_label: str | None = None
    input_groups: dict[str, tuple[str, ...]] | None = None
    field_labels: dict[str, str] | None = None
    field_choices: dict[str, dict[str, str]] | None = None
    dataset: str | None = None
    metrics: dict[str, float] | None = None
    source_path: Path | None = None


def load_model_registry(directory: str | Path = MODEL_REGISTRY_DIR) -> dict[str, ModelSpec]:
    """Load model declarations from JSON files; inactive entries stay registered."""
    specs = {}
    directories = [Path(directory)]
    if Path(directory).resolve() == MODEL_REGISTRY_DIR.resolve():
        directories.append(TRAINED_MODELS_DIR)
    for registry_path in directories:
        for path in sorted(registry_path.glob("*.json")):
            with path.open(encoding="utf-8") as stream:
                data: dict[str, Any] = json.load(stream)
            spec = ModelSpec(
                model_id=data["model_id"],
                display_name=data["display_name"],
                task=data["task"],
                target=data["target"],
                artifact=data["artifact"],
                active=bool(data.get("active", False)),
                required_fields=tuple(data.get("required_fields", [])),
                required_source_columns=tuple(data.get("required_source_columns", [])),
                positive_label=data.get("positive_label"),
                input_groups={name: tuple(fields) for name, fields in data.get("input_groups", {}).items()},
                field_labels=dict(data.get("field_labels", {})),
                field_choices={field: dict(choices) for field, choices in data.get("field_choices", {}).items()},
                dataset=data.get("dataset"),
                metrics={key: float(value) for key, value in data.get("metrics", {}).items() if isinstance(value, (int, float))},
                source_path=path,
            )
            if spec.model_id in specs:
                raise ValueError(f"Duplicate model_id in registry: {spec.model_id}")
            specs[spec.model_id] = spec
    return specs


def active_models(directory: str | Path = MODEL_REGISTRY_DIR) -> list[ModelSpec]:
    """Return only model modules explicitly enabled in registry configuration."""
    return sorted(
        (spec for spec in load_model_registry(directory).values() if spec.active),
        key=lambda spec: (
            (spec.metrics or {}).get("cv_f1", (spec.metrics or {}).get("f1", 0.0)),
            (spec.metrics or {}).get("roc_auc", 0.0),
            (spec.metrics or {}).get("f1", 0.0),
            (spec.metrics or {}).get("recall", 0.0),
            (spec.metrics or {}).get("accuracy", 0.0),
        ),
        reverse=True,
    )


def model_compatibility(
    spec: ModelSpec,
    available_fields: set[str] | list[str] | tuple[str, ...],
    source_columns: set[str] | list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Check semantic feature compatibility, with optional source-specific constraints."""
    fields = set(available_fields)
    source = {str(value).casefold() for value in source_columns}
    missing = [field for field in spec.required_fields if field not in fields]
    source_missing = [column for column in spec.required_source_columns if column.casefold() not in source]
    compatible = spec.active and not missing and not source_missing and (PROJECT_ROOT / spec.artifact).exists()
    return {
        "compatible": compatible,
        "missing_fields": missing,
        "missing_source_columns": source_missing,
        "reason": None if compatible else "Required model features or a compatible source schema are not available.",
    }
