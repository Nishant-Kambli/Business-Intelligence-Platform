"""Reusable, schema-driven training for independent workforce attrition models."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .config import PROJECT_ROOT, TRAINED_MODELS_DIR, DatasetConfig
from .dataset_registry import map_headers


_EXCLUDED_FIELDS = {
    "employee_id", "employee_count", "employee_name", "name", "first_name", "last_name",
    "full_name", "email", "over_18", "standard_hours",
}
_NULL_WORDS = {"", "nan", "none", "null", "na", "n/a", "<na>"}


def _label_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold().strip())


def _categorical_values(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    text = text.mask(text.str.casefold().isin(_NULL_WORDS).fillna(False), pd.NA)
    return text.astype(object).where(text.notna(), np.nan)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    return slug[:48] or "workforce_dataset"


def _canonical_frame(source: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    mapping = map_headers(source.columns, config)
    frame = source.rename(columns=mapping["raw_to_canonical"]).copy()
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame = frame.drop_duplicates().reset_index(drop=True)
    for column in frame.select_dtypes(include=["object", "string", "category"]).columns:
        cleaned = frame[column].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        frame[column] = cleaned.mask(cleaned.str.casefold().isin(_NULL_WORDS).fillna(False), pd.NA)
    return frame, mapping


def _prepare_target(values: pd.Series, config: DatasetConfig, positive_label: str | None) -> tuple[pd.Series, str]:
    raw = values.astype("string").str.strip()
    present = [str(value) for value in raw.dropna().unique()]
    if len(present) != 2:
        raise ValueError(f"Attrition training needs exactly two target classes; found {len(present)}.")

    if positive_label is not None:
        selected = next((value for value in present if value.casefold() == str(positive_label).casefold()), None)
        if selected is None:
            raise ValueError("The selected positive outcome is not present in the target column.")
        positive = selected
    else:
        configured = {_label_key(value) for value in config.target_positive_values}
        matched = [value for value in present if _label_key(value) in configured]
        if len(matched) != 1:
            raise ValueError("Choose which target outcome means attrition before training this dataset.")
        positive = matched[0]

    encoded = raw.map(lambda value: "Yes" if pd.notna(value) and str(value).casefold() == positive.casefold() else ("No" if pd.notna(value) else pd.NA))
    return encoded, positive


def _feature_groups(frame: pd.DataFrame, feature_fields: list[str], config: DatasetConfig) -> tuple[list[str], list[str], pd.DataFrame]:
    numeric: list[str] = []
    categorical: list[str] = []
    result = frame[feature_fields].copy()
    configured_categorical = set(config.categorical_columns)
    configured_numeric = set(config.numeric_columns)
    for field in feature_fields:
        series = result[field]
        if field in configured_categorical:
            categorical.append(field)
            result[field] = _categorical_values(series)
            continue
        converted = pd.to_numeric(series, errors="coerce")
        nonmissing = int(series.notna().sum())
        conversion_share = float(converted.notna().sum() / nonmissing) if nonmissing else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(series) or (conversion_share >= 0.9 and field not in configured_categorical)
        # Configured numeric fields with text labels remain categorical (for example High/Low ratings).
        if field in configured_numeric and conversion_share < 0.9:
            is_numeric = False
        if is_numeric:
            numeric.append(field)
            result[field] = converted.replace([np.inf, -np.inf], np.nan)
        else:
            categorical.append(field)
            result[field] = _categorical_values(series)
    return numeric, categorical, result


def _pipeline(numeric_fields: list[str], categorical_fields: list[str], estimator: Any) -> Pipeline:
    transformers = []
    if numeric_fields:
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric, numeric_fields))
    if categorical_fields:
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical, categorical_fields))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("model", estimator),
    ])


def _model_searches() -> dict[str, tuple[Any, dict[str, Any]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
            {"model__C": loguniform(0.01, 20.0)},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {"model__max_depth": [None, 5, 10, 15, 25], "model__min_samples_split": randint(2, 21), "model__min_samples_leaf": randint(1, 11)},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"model__n_estimators": randint(80, 241), "model__max_depth": [None, 10, 20, 30], "model__min_samples_split": randint(2, 16), "model__min_samples_leaf": randint(1, 9), "model__max_features": ["sqrt", 0.5, 0.8]},
        ),
    }


_ATTRITION_F1_SCORER = make_scorer(f1_score, pos_label="Yes", zero_division=0)


def _input_groups(feature_fields: list[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    buckets = {
        "Employee Profile": {"age", "gender", "marital_status", "race", "education", "education_field", "state"},
        "Employment": {"department", "business_unit", "division", "job_role", "title", "experience", "total_working_years", "years_in_current_role", "years_since_last_promotion", "years_with_current_manager", "employment_status", "employee_type", "employee_classification", "promotion_count", "promotion_last_five_years"},
        "Compensation": {"salary", "pay_zone", "daily_rate", "hourly_rate", "monthly_rate", "percent_salary_hike"},
        "Engagement": {"satisfaction", "satisfaction_score", "engagement_score", "work_life_balance_score", "job_satisfaction", "environment_satisfaction", "relationship_satisfaction", "job_involvement", "work_life_balance", "employee_recognition"},
        "Performance": {"performance_score", "performance_rating", "employee_rating", "training_times_last_year", "training_program", "training_type", "training_outcome", "training_duration_days", "training_cost"},
    }
    groups: dict[str, list[str]] = {name: [] for name in buckets}
    groups["Other Workforce Factors"] = []
    labels = {}
    for field in feature_fields:
        group = next((name for name, members in buckets.items() if field in members), "Other Workforce Factors")
        groups[group].append(field)
        labels[field] = field.replace("_", " ").title()
    return {name: fields for name, fields in groups.items() if fields}, labels


def train_attrition_model(
    source: pd.DataFrame,
    config: DatasetConfig,
    dataset_name: str,
    target_column: str | None = None,
    positive_label: str | None = None,
    models_dir: str | Path = TRAINED_MODELS_DIR,
    random_state: int = 42,
    cv_folds: int = 5,
    n_iter: int = 4,
) -> dict[str, Any]:
    """Tune candidate classifiers and save the best complete pipeline for one dataset."""
    if source.empty:
        raise ValueError("The selected dataset contains no rows.")
    frame, mapping = _canonical_frame(source, config)
    configured_target = config.target_column or "attrition"
    selected_target = target_column or configured_target
    if selected_target in mapping["raw_to_canonical"]:
        target_raw = selected_target
        target_field = mapping["raw_to_canonical"][selected_target]
    elif selected_target in frame.columns:
        target_raw = mapping["canonical_to_raw"].get(selected_target, [selected_target])[0]
        target_field = selected_target
    else:
        raise ValueError("The selected dataset does not contain a configured attrition target.")
    if target_field != configured_target:
        raise ValueError("The selected target is not configured as an attrition target alias.")

    target, resolved_positive = _prepare_target(frame[target_field], config, positive_label)
    valid_target = target.notna()
    frame = frame.loc[valid_target].reset_index(drop=True)
    target = target.loc[valid_target].astype(str).reset_index(drop=True)
    class_counts = target.value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < cv_folds:
        raise ValueError(f"At least {cv_folds} rows from each attrition class are required for stratified {cv_folds}-fold cross-validation.")

    canonical_fields = set(config.aliases)
    excluded = set(_EXCLUDED_FIELDS) | {configured_target}
    excluded |= {field for field in canonical_fields if field.endswith("_id") or field.endswith("_name") or field.endswith("_email")}
    excluded |= set(config.date_columns)
    feature_fields = [
        field for field in frame.columns
        if field in canonical_fields and field not in excluded and field != target_field
    ]
    if not feature_fields:
        raise ValueError("No supported, non-identifying workforce features were found for this dataset.")
    numeric_fields, categorical_fields, features = _feature_groups(frame, feature_fields, config)
    if not numeric_fields and not categorical_fields:
        raise ValueError("No usable numeric or categorical workforce features were found.")

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=random_state, stratify=target,
    )
    minimum_training_class = int(y_train.value_counts().min())
    folds = int(cv_folds)
    if minimum_training_class < folds:
        raise ValueError(f"At least {folds + 2} examples per outcome are needed to retain {folds}-fold CV after the 80/20 split.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    comparisons = []
    fitted = {}
    for name, (estimator, distributions) in _model_searches().items():
        search = RandomizedSearchCV(
            estimator=_pipeline(numeric_fields, categorical_fields, estimator),
            param_distributions=distributions,
            n_iter=max(1, int(n_iter)),
            scoring=_ATTRITION_F1_SCORER,
            cv=cv,
            refit=True,
            random_state=random_state,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, list(model.classes_).index("Yes")]
        metrics = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "precision": float(precision_score(y_test, prediction, pos_label="Yes", zero_division=0)),
            "recall": float(recall_score(y_test, prediction, pos_label="Yes", zero_division=0)),
            "f1": float(f1_score(y_test, prediction, pos_label="Yes", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
            "confusion_matrix": confusion_matrix(y_test, prediction, labels=["No", "Yes"]).tolist(),
        }
        comparisons.append({"model": name, **metrics, "cv_f1": float(search.best_score_), "best_params": search.best_params_})
        fitted[name] = model

    # Select by cross-validated F1; keep the held-out test set as a final report.
    comparisons.sort(key=lambda item: item["cv_f1"], reverse=True)
    best = comparisons[0]
    best_model = fitted[best["model"]]
    source_signature = "|".join(sorted(str(column) for column in source.columns))
    schema_hash = hashlib.sha256(source_signature.encode("utf-8")).hexdigest()[:8]
    model_id = f"{_slug(dataset_name)}_attrition_{schema_hash}"
    relative_artifact = Path("models") / f"{model_id}.pkl"
    destination = Path(models_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = PROJECT_ROOT / relative_artifact if Path(models_dir) == TRAINED_MODELS_DIR else destination / relative_artifact.name
    with tempfile.NamedTemporaryFile(prefix=f"{model_id}_", suffix=".pkl", dir=destination, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(best_model, temporary_path)
        temporary_path.replace(artifact_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    groups, labels = _input_groups(feature_fields)
    metadata = {
        "model_id": model_id,
        "display_name": f"Attrition Prediction · {dataset_name}",
        "model_name": best["model"],
        "task": "classification",
        "target": configured_target,
        "target_source_column": target_raw,
        "positive_label": "Yes",
        "source_positive_label": resolved_positive,
        "dataset": str(dataset_name),
        "dataset_schema_hash": schema_hash,
        "artifact": str(relative_artifact).replace("\\", "/"),
        "active": True,
        "required_fields": feature_fields,
        "supported_features": feature_fields,
        "required_source_columns": [],
        "numeric_features": numeric_fields,
        "categorical_features": categorical_fields,
        "input_groups": groups,
        "field_labels": labels,
        "metrics": {key: best[key] for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1")},
        "selection_metric": "Stratified cross-validation F1-score for the positive attrition outcome",
        "confusion_matrix": best["confusion_matrix"],
        "best_params": best["best_params"],
        "training_rows": int(len(frame)),
        "training_class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "cross_validation_folds": int(folds),
        "all_model_results": comparisons,
    }
    metadata_path = destination / f"{model_id}.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination, suffix=".json", delete=False) as temporary:
        json.dump(metadata, temporary, indent=2, ensure_ascii=False)
        temporary.write("\n")
        metadata_temp = Path(temporary.name)
    metadata_temp.replace(metadata_path)

    return {
        "model_id": model_id,
        "artifact_path": artifact_path,
        "metadata_path": metadata_path,
        "best_model": best["model"],
        "metrics": best,
        "comparisons": pd.DataFrame(comparisons),
        "features": feature_fields,
        "rows": len(frame),
    }
