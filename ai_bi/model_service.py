"""Model loading and single-record inference for the saved pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import MODEL_PATH


def load_model(path: str | Path = MODEL_PATH):
    """Load the trusted local scikit-learn pipeline."""
    return joblib.load(path)


def predict_employee(model: Any, employee: dict[str, Any], feature_columns: list[str], positive_label: str = "Yes") -> dict[str, Any]:
    """Return class, probability, and an explicitly defined signal label."""
    missing = [column for column in feature_columns if column not in employee]
    if missing:
        raise ValueError(f"Missing model inputs: {', '.join(missing)}")
    row = pd.DataFrame([{column: employee[column] for column in feature_columns}])
    prediction = str(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = [str(value) for value in model.classes_]
    if positive_label not in classes:
        raise ValueError("The registered positive class is not present in the model output.")
    positive_index = classes.index(positive_label)
    probability = float(probabilities[positive_index])
    distance = abs(probability - 0.5) * 2
    signal = "High" if distance >= 0.5 else "Moderate" if distance >= 0.2 else "Low"
    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": "High Risk" if prediction == positive_label else "Lower Risk",
        "confidence": signal,
        "confidence_definition": "Signal strength based on distance from the 0.50 classification threshold; not calibrated certainty.",
    }


def model_feature_columns(model: Any) -> list[str]:
    """Read the raw feature order recorded by the saved pipeline."""
    preprocessor = model.named_steps.get("preprocessor") if hasattr(model, "named_steps") else None
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return [str(value) for value in preprocessor.feature_names_in_]
    raise ValueError("The saved model does not expose its expected input schema.")


def model_input_from_canonical(record: dict[str, Any], config: Any, columns: list[str]) -> dict[str, Any]:
    """Translate configured canonical values to the original pipeline column names."""
    reverse: dict[str, str] = {}
    for canonical, aliases in config.aliases.items():
        reverse[str(canonical).casefold()] = canonical
        for alias in aliases:
            reverse[str(alias).casefold()] = canonical
    missing = []
    result = {}
    for column in columns:
        canonical = reverse.get(column.casefold())
        if canonical is None or canonical not in record:
            missing.append(column)
        else:
            result[column] = record[canonical]
    if missing:
        raise ValueError(f"Missing model inputs: {', '.join(missing)}")
    return result


def local_contributions(model: Any, model_input: dict[str, Any], limit: int = 5, positive_label: str = "Yes") -> list[dict[str, str | float]]:
    """Return signed linear-model contributions when the saved estimator supports them."""
    try:
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        transformed = preprocessor.transform(pd.DataFrame([model_input]))
        names = preprocessor.get_feature_names_out()
        classes = [str(value) for value in estimator.classes_]
        positive = classes.index(positive_label)
        if estimator.coef_.shape[0] == 1:
            coefficients = estimator.coef_[0] * (1 if positive == 1 else -1)
        else:
            coefficients = estimator.coef_[positive]
        row = transformed.toarray()[0] if hasattr(transformed, "toarray") else transformed[0]
        impacts = row * coefficients
        top = sorted(enumerate(impacts), key=lambda item: abs(item[1]), reverse=True)[:limit]
        return [{"feature": str(names[index]), "contribution": float(value), "direction": "higher model score" if value > 0 else "lower model score"} for index, value in top]
    except (KeyError, AttributeError, ValueError, IndexError):
        return []
