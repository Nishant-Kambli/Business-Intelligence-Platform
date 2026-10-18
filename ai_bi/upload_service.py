"""CSV upload parsing and configurable schema detection."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import pandas as pd

from .config import DatasetConfig
from .dataset_registry import detect_dataset


@dataclass
class UploadResult:
    """Result of parsing a CSV and matching it to a configured schema."""

    frame: pd.DataFrame | None
    config: DatasetConfig | None
    mapping: dict[str, Any] | None
    error_code: str | None = None


def read_csv_upload(upload: Any) -> tuple[pd.DataFrame | None, str | None]:
    """Read a browser upload without exposing low-level parser errors to the UI."""
    try:
        if isinstance(upload, (bytes, bytearray)):
            frame = pd.read_csv(BytesIO(upload))
        elif hasattr(upload, "getvalue"):
            content = upload.getvalue()
            frame = pd.read_csv(BytesIO(content))
        else:
            frame = pd.read_csv(upload)
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError, OSError):
        return None, "invalid_csv"
    if frame.empty or len(frame.columns) == 0:
        return None, "empty_csv"
    return frame, None


def detect_upload_schema(frame: pd.DataFrame, configs: dict[str, DatasetConfig]) -> tuple[DatasetConfig | None, dict[str, Any] | None]:
    """Detect a configured dataset from its aliases and detection threshold."""
    try:
        return detect_dataset(frame.columns, configs)
    except ValueError:
        return None, {"error_code": "ambiguous_columns"}


def process_upload(upload: Any, configs: dict[str, DatasetConfig]) -> UploadResult:
    """Parse and identify an upload; cleaning is intentionally delegated elsewhere."""
    frame, error_code = read_csv_upload(upload)
    if error_code:
        return UploadResult(None, None, None, error_code)
    config, mapping = detect_upload_schema(frame, configs)
    if config is None or mapping is None:
        return UploadResult(frame, None, mapping, "unsupported_schema")
    return UploadResult(frame, config, mapping)
