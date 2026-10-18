"""Dataset loading, lightweight repairs, and form-schema helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .config import DATA_PATH, DatasetConfig
from .cleaning_service import clean_dataset


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the configured source without applying dataset-specific repairs."""
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading workforce data…")
def cached_dataset(path: str = str(DATA_PATH)) -> pd.DataFrame:
    """Cache the repaired dataset for Streamlit reruns."""
    return load_dataset(path)


def missing_value_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing and whitespace-only values by column."""
    missing = frame.isna().sum()
    blank = frame.select_dtypes(include=["object", "string"]).apply(
        lambda column: column.astype("string").str.strip().eq("").fillna(False).sum()
    )
    result = pd.DataFrame({"Missing values": missing, "Blank strings": blank}).fillna(0)
    result = result.astype(int)
    return result.loc[result.sum(axis=1).gt(0)].sort_values(
        ["Missing values", "Blank strings"], ascending=False
    )


def feature_defaults(frame: pd.DataFrame, fields: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Create numeric medians and categorical modes for requested model inputs."""
    features = frame[[field for field in fields if field in frame]].copy()
    defaults: dict[str, Any] = {}
    for column in features.columns:
        series = features[column]
        if pd.api.types.is_numeric_dtype(series):
            value = series.median()
            defaults[column] = 0 if pd.isna(value) else value.item() if hasattr(value, "item") else value
        else:
            modes = series.dropna().mode()
            defaults[column] = str(modes.iloc[0]) if not modes.empty else "Unknown"
    return defaults


def feature_options(frame: pd.DataFrame, fields: tuple[str, ...] | list[str]) -> dict[str, list[str]]:
    """Return observed string options for the requested input fields."""
    features = frame[[field for field in fields if field in frame]].copy()
    options: dict[str, list[str]] = {}
    for column in features.select_dtypes(include=["object", "string", "category"]).columns:
        values = features[column].dropna().astype(str).unique().tolist()
        options[column] = sorted(values)
    return options


def normalize_dataset(frame: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the configured cleaning engine and return canonical fields."""
    result = clean_dataset(frame, config)
    result.report["dataset_name"] = config.dataset_name
    return result.frame, result.report


def read_uploaded_csv(uploaded_file: Any, configs: dict[str, DatasetConfig]) -> tuple[pd.DataFrame | None, DatasetConfig | None, dict[str, Any]]:
    """Compatibility wrapper around the upload and cleaning services."""
    from .upload_service import process_upload

    result = process_upload(uploaded_file, configs)
    if result.error_code:
        return result.frame, None, {"error_code": result.error_code}
    cleaned = clean_dataset(result.frame, result.config)
    cleaned.report["dataset_name"] = result.config.dataset_name
    return cleaned.frame, result.config, cleaned.report
