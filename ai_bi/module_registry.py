"""Schema-independent registry of analytics capabilities and their field needs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import MODULE_REGISTRY_PATH


def load_modules(path: str | Path = MODULE_REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def module_health(available_fields, modules: dict[str, dict[str, Any]] | None = None):
    """Evaluate each module independently against required and optional fields."""
    available = set(available_fields)
    modules = modules or load_modules()
    health = {}
    for module_id, spec in modules.items():
        required = list(spec.get("required_fields", []))
        missing = [field for field in required if field not in available]
        groups = spec.get("required_any", [])
        missing_any = [list(group) for group in groups if not set(group).intersection(available)]
        enabled = not missing and not missing_any
        health[module_id] = {
            "supported": enabled,
            "missing_fields": missing,
            "missing_any_of": missing_any,
            "required_fields": required,
            "optional_fields": list(spec.get("optional_fields", [])),
            "reason": None if enabled else "Missing required fields: " + "; ".join(
                [*missing, *("one of " + ", ".join(group) for group in missing_any)]
            ),
        }
    return health
