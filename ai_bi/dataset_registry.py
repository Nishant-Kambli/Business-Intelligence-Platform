"""Load dataset definitions and map source headers to canonical fields."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .config import DATASET_CONFIG_DIR, DatasetConfig
from .module_registry import module_health as evaluate_module_health


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def normalize_column_name(value: str) -> str:
    """Normalize unknown header text to stable lowercase snake_case."""
    name = str(value).strip().casefold()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_") or "unnamed_column"


def config_from_dict(data: dict[str, Any], source_path: Path | None = None) -> DatasetConfig:
    """Convert a JSON object into an immutable dataset configuration."""
    return DatasetConfig(
        dataset_id=data["dataset_id"],
        dataset_name=data["dataset_name"],
        target_column=data.get("target_column"),
        employee_id_column=data.get("employee_id_column"),
        department_column=data.get("department_column"),
        role_column=data.get("role_column"),
        salary_column=data.get("salary_column"),
        satisfaction_columns=tuple(data.get("satisfaction_columns", [])),
        experience_columns=tuple(data.get("experience_columns", [])),
        categorical_columns=tuple(data.get("categorical_columns", [])),
        numeric_columns=tuple(data.get("numeric_columns", [])),
        supported_modules=tuple(data.get("supported_modules", [])),
        detection_fields=tuple(data.get("detection_fields", [])),
        aliases={key: tuple(values) for key, values in data.get("aliases", {}).items()},
        module_requirements={key: tuple(values) for key, values in data.get("module_requirements", {}).items()},
        source_path=source_path,
        minimum_detection_fields=int(data.get("minimum_detection_fields", 0)),
        module_any_requirements={
            key: tuple(tuple(group) for group in groups)
            for key, groups in data.get("module_any_requirements", {}).items()
        },
        date_columns=tuple(data.get("date_columns", [])),
        target_positive_values=tuple(str(value).casefold() for value in data.get("target_positive_values", ["yes", "1", "true"])),
        target_negative_values=tuple(str(value).casefold() for value in data.get("target_negative_values", ["no", "0", "false"])),
    )


def load_dataset_configs(directory: str | Path = DATASET_CONFIG_DIR) -> dict[str, DatasetConfig]:
    """Discover JSON dataset definitions without editing the application registry."""
    configs: dict[str, DatasetConfig] = {}
    for path in sorted(Path(directory).glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            config = config_from_dict(json.load(handle), path)
        if config.dataset_id in configs:
            raise ValueError(f"Duplicate dataset_id in configuration files: {config.dataset_id}")
        configs[config.dataset_id] = config
    return configs


def map_headers(headers: Iterable[str], config: DatasetConfig) -> dict[str, Any]:
    """Map known aliases and retain a report of extras and ambiguous mappings."""
    alias_to_field: dict[str, str] = {}
    for field, aliases in config.aliases.items():
        for alias in (field, *aliases):
            key = _column_key(alias)
            previous = alias_to_field.get(key)
            if previous is not None and previous != field:
                raise ValueError(f"Alias '{alias}' is configured for both {previous} and {field}.")
            alias_to_field[key] = field

    raw_to_canonical: dict[str, str] = {}
    canonical_to_raw: dict[str, list[str]] = {}
    unsupported: list[str] = []
    collisions: dict[str, list[str]] = {}
    for raw in headers:
        field = alias_to_field.get(_column_key(raw))
        if field is None:
            unsupported.append(str(raw))
            field = normalize_column_name(raw)
        if field in canonical_to_raw:
            collisions.setdefault(field, list(canonical_to_raw[field])).append(str(raw))
            field = normalize_column_name(raw)
            base = field
            suffix = 2
            while field in canonical_to_raw:
                field = f"{base}_{suffix}"
                suffix += 1
        raw_to_canonical[str(raw)] = field
        canonical_to_raw.setdefault(field, []).append(str(raw))
    return {
        "raw_to_canonical": raw_to_canonical,
        "canonical_to_raw": canonical_to_raw,
        "unsupported_columns": unsupported,
        "collisions": collisions,
    }


def detect_dataset(headers: Iterable[str], configs: dict[str, DatasetConfig]) -> tuple[DatasetConfig | None, dict[str, Any] | None]:
    """Choose the closest alias registry while retaining unknown columns generically."""
    header_list = list(headers)
    candidates = []
    for config in configs.values():
        mapping = map_headers(header_list, config)
        present = set(mapping["canonical_to_raw"])
        score = sum(len(values) for values in mapping["canonical_to_raw"].values() if values[0] not in mapping["unsupported_columns"])
        candidates.append((score, config, mapping))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    # Ambiguous schema selection must not discard a readable dataset. Choose the
    # strongest configured alias set deterministically; modules are checked later.
    return candidates[0][1], candidates[0][2]


def module_health(config: DatasetConfig, available_fields: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Report available modules using the independent module registry."""
    return evaluate_module_health(available_fields)
