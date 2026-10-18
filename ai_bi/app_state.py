"""Shared Streamlit state for the configured dataset and optional upload."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .config import DATA_PATH, DatasetConfig
from .data import cached_dataset, load_dataset, normalize_dataset
from .dataset_registry import load_dataset_configs


def dataset_configs() -> dict[str, DatasetConfig]:
    return load_dataset_configs()


def active_dataset() -> tuple[pd.DataFrame, DatasetConfig, dict[str, Any], str]:
    """Return the current session dataset, falling back to the configured default."""
    configs = dataset_configs()
    uploaded = st.session_state.get("uploaded_dataset")
    if uploaded is not None:
        config = configs[st.session_state["uploaded_dataset_id"]]
        frame = uploaded.copy()
        report = st.session_state.get("uploaded_dataset_report", {})
        return frame, config, report, report.get("source_name", "Uploaded HR Dataset")
    config = configs["workforce_v1"]
    raw = cached_dataset(str(DATA_PATH))
    frame, report = normalize_dataset(raw, config)
    return frame, config, report, config.dataset_name


def active_raw_dataset() -> tuple[pd.DataFrame, DatasetConfig, str]:
    """Return the un-imputed source used for leakage-safe model training."""
    configs = dataset_configs()
    uploaded = st.session_state.get("uploaded_raw_dataset", st.session_state.get("uploaded_dataset"))
    if uploaded is not None:
        config = configs[st.session_state["uploaded_dataset_id"]]
        report = st.session_state.get("uploaded_dataset_report", {})
        return uploaded.copy(), config, report.get("source_name", "Uploaded HR Dataset")
    config = configs["workforce_v1"]
    return load_dataset(DATA_PATH), config, config.dataset_name


def set_uploaded_dataset(
    frame: pd.DataFrame,
    config: DatasetConfig,
    report: dict[str, Any] | None = None,
    raw_frame: pd.DataFrame | None = None,
) -> None:
    st.session_state["uploaded_dataset"] = frame.copy()
    st.session_state["uploaded_raw_dataset"] = (raw_frame if raw_frame is not None else frame).copy()
    st.session_state["uploaded_dataset_id"] = config.dataset_id
    st.session_state["uploaded_dataset_report"] = report or {}


def clear_uploaded_dataset() -> None:
    st.session_state.pop("uploaded_dataset", None)
    st.session_state.pop("uploaded_raw_dataset", None)
    st.session_state.pop("uploaded_dataset_id", None)
    st.session_state.pop("uploaded_dataset_report", None)
