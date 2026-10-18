"""Schema-aware cleaning for uploaded and configured workforce datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import DatasetConfig
from .dataset_registry import map_headers, module_health


NULL_TEXT = {"", "nan", "none", "null", "na", "n/a", "<na>"}


@dataclass
class CleaningResult:
    """Cleaned canonical data and an internal audit summary."""

    frame: pd.DataFrame
    report: dict[str, Any]


def _clean_text(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    nulls = values.str.casefold().isin(NULL_TEXT).fillna(False)
    return values.mask(nulls, pd.NA)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = _clean_text(series).str.replace(r"[,$€£₹%]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def clean_dataset(source: pd.DataFrame, config: DatasetConfig) -> CleaningResult:
    """Map aliases, remove duplicates, normalize values, parse types, and impute."""
    if source.empty:
        raise ValueError("The dataset has no rows.")
    mapping = map_headers(source.columns, config)
    frame = source.rename(columns=mapping["raw_to_canonical"]).copy()
    rows_before = len(frame)
    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    imputed: dict[str, int] = {}
    converted_numeric: list[str] = []
    preserved_categorical: list[str] = []
    parsed_dates: list[str] = []

    for column in frame.select_dtypes(include=["object", "string", "category"]).columns:
        frame[column] = _clean_text(frame[column])

    date_fields = set(config.date_columns)
    for column in frame.columns:
        if column in date_fields or ("date" in str(column).casefold() and column not in config.numeric_columns):
            parsed = pd.to_datetime(frame[column], format="mixed", dayfirst=True, errors="coerce")
            if parsed.notna().sum() > 0:
                frame[column] = parsed
                parsed_dates.append(str(column))

    categorical_fields = set(config.categorical_columns)
    numeric_fields = set(config.numeric_columns)
    # Salary can be numeric pay or a dataset's categorical salary band.
    for column in frame.columns:
        if column in numeric_fields or column == config.salary_column:
            converted = _coerce_numeric(frame[column])
            nonmissing = frame[column].notna().sum()
            numeric_share = converted.notna().sum() / nonmissing if nonmissing else 0
            if numeric_share >= 0.9:
                frame[column] = converted
                converted_numeric.append(str(column))
            elif nonmissing:
                # Ordinal fields may be labels (e.g. High/Medium/Low) in one
                # source and integers in another. Preserve labels when numeric
                # parsing does not reliably describe the source values.
                preserved_categorical.append(str(column))
        elif column not in categorical_fields and not pd.api.types.is_datetime64_any_dtype(frame[column]):
            candidate = _coerce_numeric(frame[column])
            nonmissing = frame[column].notna().sum()
            if nonmissing and candidate.notna().sum() / nonmissing >= 0.9:
                frame[column] = candidate
                converted_numeric.append(str(column))

    target = config.target_column
    target_rows_dropped = 0
    if target and target in frame:
        normalized = frame[target].astype("string").str.strip().str.casefold()
        positive = set(config.target_positive_values)
        negative = set(config.target_negative_values)
        mapped = frame[target].astype("string").str.strip()
        mapped = mapped.mask(normalized.isin(positive), "Yes")
        mapped = mapped.mask(normalized.isin(negative), "No")
        frame[target] = mapped
        missing_target = frame[target].isna()
        target_rows_dropped = int(missing_target.sum())
        if target_rows_dropped:
            frame = frame.loc[~missing_target].reset_index(drop=True)

    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            missing_count = int(frame[column].isna().sum())
            if missing_count:
                median = frame[column].median()
                fill_value = 0 if pd.isna(median) else median
                frame[column] = frame[column].fillna(fill_value)
                imputed[str(column)] = missing_count
                values = frame[column]
                if values.notna().all() and (values % 1 == 0).all():
                    frame[column] = values.astype("int64")
        elif pd.api.types.is_datetime64_any_dtype(frame[column]):
            missing_count = int(frame[column].isna().sum())
            if missing_count:
                modes = frame[column].dropna().mode()
                if not modes.empty:
                    frame[column] = frame[column].fillna(modes.iloc[0])
                    imputed[str(column)] = missing_count
        elif column != target and not pd.api.types.is_datetime64_any_dtype(frame[column]):
            missing_count = int(frame[column].isna().sum())
            if missing_count:
                modes = frame[column].dropna().mode()
                fill_value = modes.iloc[0] if not modes.empty else "Unknown"
                frame[column] = frame[column].fillna(fill_value)
                imputed[str(column)] = missing_count

    available = set(mapping["canonical_to_raw"])
    report = {
        "rows_before": rows_before,
        "rows_after": len(frame),
        "duplicates_removed": duplicates,
        "target_rows_dropped": target_rows_dropped,
        "imputed_values": imputed,
        "numeric_columns_converted": sorted(set(converted_numeric)),
        "numeric_fields_preserved_as_categories": sorted(set(preserved_categorical)),
        "date_columns_parsed": sorted(set(parsed_dates)),
        "available_fields": sorted(available),
        "unsupported_columns": mapping["unsupported_columns"],
        "canonical_to_raw": mapping["canonical_to_raw"],
        "module_health": module_health(config, available),
    }
    return CleaningResult(frame=frame, report=report)
