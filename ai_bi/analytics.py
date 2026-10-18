"""Dataset-driven workforce, compensation, engagement, and retention metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import POSITIVE_LABEL


def _field(frame: pd.DataFrame, canonical: str, legacy: str | None = None) -> str | None:
    for name in (canonical, legacy):
        if name and name in frame.columns:
            return name
    return None


def _numeric(frame: pd.DataFrame, field: str | None) -> pd.Series:
    return pd.to_numeric(frame[field], errors="coerce") if field else pd.Series(dtype=float)


def workforce_kpis(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Compute workforce headline measures when the required fields exist."""
    dept = _field(frame, "department", "Department")
    age_field = _field(frame, "age", "Age")
    salary_field = _field(frame, "salary", "MonthlyIncome")
    target = _field(frame, "attrition", "Attrition")
    employee_id = _field(frame, "employee_id", "EmployeeNumber")
    departments = frame[dept].dropna().astype(str) if dept else pd.Series(dtype=str)
    departments = departments[~departments.str.casefold().isin(("unknown", "nan", ""))]
    attrition = frame[target].astype("string").str.casefold() if target else pd.Series(dtype="string")
    age = _numeric(frame, age_field)
    salary = _numeric(frame, salary_field)
    return {
        "employees": int(frame[employee_id].nunique(dropna=True)) if employee_id else int(len(frame)),
        "attrition_rate": float(attrition.eq(POSITIVE_LABEL.casefold()).mean() * 100) if target and len(frame) else None,
        "departments": int(departments.nunique()) if dept else None,
        "average_age": float(age.mean()) if age.notna().any() else None,
        "average_salary": float(salary.mean()) if salary.notna().any() else None,
    }


def field_distribution(frame: pd.DataFrame, field: str, limit: int = 20, entity_field: str | None = None) -> pd.DataFrame:
    """Return row counts for any available field, numeric or categorical."""
    if field not in frame:
        return pd.DataFrame(columns=["Value", "Employees"])
    source = frame
    if entity_field and entity_field in frame and field != entity_field:
        source = frame.drop_duplicates([entity_field, field])
    values = source[field].dropna()
    if values.empty:
        return pd.DataFrame(columns=["Value", "Employees"])
    if pd.api.types.is_numeric_dtype(values):
        bins = min(limit, max(1, values.nunique()))
        if values.nunique() > limit:
            counts = pd.cut(values, bins=bins, include_lowest=True).value_counts(sort=False)
            result = counts.rename_axis("Value").reset_index(name="Employees")
            result["Value"] = result["Value"].astype(str)
            return result
    return values.astype("string").value_counts().head(limit).rename_axis("Value").reset_index(name="Employees")


def group_summary(frame: pd.DataFrame, dimension: str, measure: str | None = None, agg: str = "mean") -> pd.DataFrame:
    """Aggregate a numeric measure by a categorical dimension."""
    if dimension not in frame or measure is None or measure not in frame:
        return pd.DataFrame(columns=["Group", "Value", "Employees"])
    values = frame[[dimension, measure]].copy()
    values[measure] = pd.to_numeric(values[measure], errors="coerce")
    grouped = values.groupby(dimension, dropna=False)[measure].agg(agg).rename("Value").to_frame()
    grouped["Employees"] = values.groupby(dimension, dropna=False)[measure].count()
    return grouped.reset_index().rename(columns={dimension: "Group"}).sort_values("Value", ascending=False)


def attrition_by(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Return employee counts and historical attrition rate by group."""
    target = _field(frame, "attrition", "Attrition")
    if dimension not in frame or not target:
        return pd.DataFrame(columns=["Group", "Employees", "Attrition rate"])
    grouped = frame.groupby(dimension, dropna=False)[target].agg(
        Employees="size",
        AttritionRate=lambda values: values.astype("string").str.casefold().eq("yes").mean() * 100,
    )
    return grouped.reset_index().rename(columns={dimension: "Group", "AttritionRate": "Attrition rate"})


def engagement_summary(frame: pd.DataFrame, fields: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Summarize configured satisfaction fields as means and nonmissing counts."""
    records: list[dict[str, Any]] = []
    for field in fields:
        if field in frame:
            values = pd.to_numeric(frame[field], errors="coerce")
            records.append({"Measure": field.replace("_", " ").title(), "Average": values.mean(), "Responses": int(values.notna().sum())})
    return pd.DataFrame(records)


def department_summary(frame: pd.DataFrame, config: Any) -> pd.DataFrame:
    """Create a department-level workforce and business metric summary when possible."""
    department = config.department_column
    if not department or department not in frame:
        return pd.DataFrame(columns=["Department", "Employees"])
    entity = config.employee_id_column
    source = frame.drop_duplicates([entity]) if entity and entity in frame else frame
    grouped = source.groupby(department, dropna=False)
    result = grouped.size().rename("Employees").to_frame()
    for canonical, label in (("salary", "Average salary"), ("age", "Average age"), ("experience", "Average experience")):
        if canonical in source and pd.to_numeric(source[canonical], errors="coerce").notna().any():
            result[label] = grouped[canonical].apply(lambda values: pd.to_numeric(values, errors="coerce").mean())
    target = config.target_column
    if target and target in source:
        result["Historical attrition rate"] = grouped[target].apply(
            lambda values: values.astype("string").str.casefold().eq(POSITIVE_LABEL.casefold()).mean() * 100
        )
    return result.reset_index().rename(columns={department: "Department"}).sort_values("Employees", ascending=False)


def executive_insights(frame: pd.DataFrame, satisfaction_fields: tuple[str, ...] | list[str] = ()) -> list[str]:
    """Create descriptive alerts supported by available fields, without causal claims."""
    if "employee_id" in frame:
        frame = frame.drop_duplicates("employee_id")
    insights: list[str] = []
    dimension = next((field for field in ("department", "business_unit", "division", "job_role", "title", "role") if field in frame), None)
    if dimension is None:
        dimension = next((field for field in frame.select_dtypes(include=["object", "string", "category"]).columns if field != _field(frame, "attrition", "Attrition")), None)
    if dimension:
        counts = frame[dimension].dropna().astype(str).value_counts()
        if not counts.empty:
            label = dimension.replace("_", " ").title()
            insights.append(f"Largest {label.lower()} group: {counts.index[0]} ({int(counts.iloc[0]):,} records).")

    target = _field(frame, "attrition", "Attrition")
    if target:
        positives = frame[target].astype("string").str.casefold().eq(POSITIVE_LABEL.casefold())
        insights.append(f"Recorded attrition rate: {positives.mean() * 100:.1f}% across {int(positives.sum()):,} records.")
        if dimension:
            group_rates = attrition_by(frame, dimension)
            group_rates = group_rates[group_rates["Employees"] >= max(5, int(len(frame) * 0.01))]
            if not group_rates.empty:
                top = group_rates.sort_values("Attrition rate", ascending=False).iloc[0]
                insights.append(f"Highest historical attrition rate among sufficiently sized {dimension.replace('_', ' ')} groups: {top['Group']} ({top['Attrition rate']:.1f}%).")

    available_satisfaction = [field for field in satisfaction_fields if field in frame]
    if not available_satisfaction:
        available_satisfaction = [field for field in frame if "satisfaction" in str(field).casefold() or "engagement" in str(field).casefold()]
    for satisfaction in available_satisfaction[:2]:
        values = pd.to_numeric(frame[satisfaction], errors="coerce")
        if values.notna().any():
            insights.append(f"Average {satisfaction.replace('_', ' ')}: {values.mean():.2f} on the source scale.")

    for field in ("salary", "experience", "age"):
        if field in frame and pd.api.types.is_numeric_dtype(frame[field]):
            values = pd.to_numeric(frame[field], errors="coerce").dropna()
            if not values.empty:
                insights.append(f"Median {field.replace('_', ' ')}: {values.median():,.1f} in dataset units.")

    for field in ("performance_score", "employee_rating", "performance_rating", "training_outcome", "employee_status"):
        if field in frame:
            values = frame[field].dropna().astype("string").str.strip()
            counts = values.value_counts()
            if not counts.empty:
                share = counts.iloc[0] / counts.sum() * 100
                insights.append(f"Most common {field.replace('_', ' ')}: {counts.index[0]} ({share:.1f}% of recorded values).")

    if "training_outcome" in frame:
        outcomes = frame["training_outcome"].astype("string").str.casefold()
        unsuccessful = outcomes.str.contains(r"fail|incomplet|unsuccess", regex=True, na=False)
        if unsuccessful.any():
            insights.append(f"Training records marked failed or incomplete: {int(unsuccessful.sum()):,} ({unsuccessful.mean() * 100:.1f}%).")
    if not insights:
        insights.append("Add more compatible HR fields to generate workforce insights.")
    return insights


def dataset_alerts(frame: pd.DataFrame, config: Any) -> list[str]:
    """Create snapshot observations from outcomes present in the active dataset."""
    alerts = []
    target = getattr(config, "target_column", None)
    if target and target in frame:
        summary = department_summary(frame, config)
        if "Historical attrition rate" in summary and not summary.empty:
            minimum_group_size = max(5, int(len(frame) * 0.01))
            eligible = summary[summary["Employees"] >= minimum_group_size]
            if not eligible.empty:
                highest = eligible.sort_values("Historical attrition rate", ascending=False).iloc[0]
                alerts.append(f"Highest recorded attrition rate: {highest['Department']} ({highest['Historical attrition rate']:.1f}%).")

    status_field = _field(frame, "employment_status", "EmployeeStatus")
    if status_field:
        statuses = frame[status_field].astype("string").str.casefold()
        exited = statuses.str.contains(r"terminated|inactive|separated", regex=True, na=False)
        if exited.any():
            alerts.append(f"Records marked inactive or terminated: {int(exited.sum()):,} ({exited.mean() * 100:.1f}%).")

    outcome_field = _field(frame, "training_outcome", "TrainingOutcome")
    if outcome_field:
        outcomes = frame[outcome_field].astype("string").str.casefold()
        unsuccessful = outcomes.str.contains(r"fail|incomplet|unsuccess", regex=True, na=False)
        if unsuccessful.any():
            alerts.append(f"Training records marked failed or incomplete: {int(unsuccessful.sum()):,} ({unsuccessful.mean() * 100:.1f}%).")

    performance_field = _field(frame, "performance_score", "PerformanceScore")
    if performance_field:
        performance = frame[performance_field].astype("string").str.casefold()
        needs_support = performance.str.contains(r"pip|needs improvement|below expectations", regex=True, na=False)
        if needs_support.any():
            alerts.append(f"Performance records marked PIP or needing improvement: {int(needs_support.sum()):,} ({needs_support.mean() * 100:.1f}%).")
    return alerts
