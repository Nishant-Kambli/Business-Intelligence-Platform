"""Reusable rendering helpers for schema-driven analytics modules."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .analytics import field_distribution, group_summary
from .app_state import active_dataset
from .module_registry import load_modules
from .utils import page_header


def _title(value: str) -> str:
    return value.replace("_", " ").title()


def render_module(module_id: str, description: str) -> None:
    frame, config, _report, _source = active_dataset()
    spec = load_modules()[module_id]
    title = spec["title"]
    page_header("People Intelligence / Analytics", title, description)

    if module_id == "performance":
        fields = [field for field in ("performance_score", "employee_rating", "performance_rating") if field in frame]
        if not fields:
            st.info("Performance analytics is unavailable because this dataset has no performance measures.")
            return
        cards = st.container(horizontal=True, gap="medium")
        with cards:
            for field in fields:
                values = pd.to_numeric(frame[field], errors="coerce")
                if values.notna().any():
                    cards.metric(_title(field), f"{values.mean():.2f}", border=True)
        selected = st.selectbox("Performance measure", fields, format_func=_title)
        distribution = field_distribution(frame, selected)
        st.bar_chart(distribution, x="Value", y="Employees", alt=f"Distribution of {_title(selected).lower()}.")
        st.dataframe(distribution, hide_index=True, alt=f"Counts for {_title(selected).lower()} values.")
        dimension = next((field for field in ("department", "business_unit", "division", "job_role", "title") if field in frame), None)
        if dimension:
            numeric = pd.to_numeric(frame[selected], errors="coerce")
            if numeric.notna().any():
                summary = group_summary(frame.assign(**{selected: numeric}), dimension, selected)
                st.markdown(f"### Average by {_title(dimension)}")
                st.bar_chart(summary, x="Group", y="Value", horizontal=True, alt=f"Average {_title(selected).lower()} by {_title(dimension).lower()}.")
                st.dataframe(summary, hide_index=True, alt=f"Performance summary by {_title(dimension).lower()}.")
        st.caption("Performance measures reflect the dataset's rating scales and definitions.")
        return

    if module_id == "training":
        st.metric("Training records", f"{len(frame):,}", border=True)
        employee_id = config.employee_id_column
        if employee_id and employee_id in frame:
            st.metric("Employees represented", f"{frame[employee_id].nunique(dropna=True):,}", border=True)
        numeric_metrics = [("training_cost", "Average training cost"), ("training_duration_days", "Average duration (days)")]
        cards = st.container(horizontal=True, gap="medium")
        with cards:
            for field, label in numeric_metrics:
                if field in frame:
                    values = pd.to_numeric(frame[field], errors="coerce").dropna()
                    if not values.empty:
                        cards.metric(label, f"{values.mean():,.2f}", border=True)
        dimensions = [field for field in ("training_program", "training_type", "training_outcome") if field in frame]
        if dimensions:
            selected = st.selectbox("Training breakdown", dimensions, format_func=_title)
            distribution = field_distribution(frame, selected)
            st.bar_chart(distribution, x="Value", y="Employees", alt=f"Training records by {_title(selected).lower()}.")
            st.dataframe(distribution, hide_index=True, alt=f"Training records by {_title(selected).lower()}.")
        if "training_cost" in frame:
            cost = pd.to_numeric(frame["training_cost"], errors="coerce")
            dimension = next((field for field in ("training_program", "training_type", "department", "business_unit") if field in frame), None)
            if dimension and cost.notna().any():
                summary = group_summary(frame.assign(training_cost=cost), dimension, "training_cost")
                st.markdown(f"### Average training cost by {_title(dimension)}")
                st.bar_chart(summary, x="Group", y="Value", horizontal=True, alt=f"Average training cost by {_title(dimension).lower()}.")
                st.dataframe(summary.rename(columns={"Value": "Average cost", "Employees": "Records"}), hide_index=True, alt="Average training cost by group.")
        st.caption("A row may represent a training event rather than a unique employee.")
        return

    if module_id == "recruitment":
        st.metric("Recruitment records", f"{len(frame):,}", border=True)
        for field in ("hiring_status", "source_channel", "job_role", "title"):
            if field in frame:
                st.markdown(f"### {_title(field)}")
                distribution = field_distribution(frame, field)
                st.bar_chart(distribution, x="Value", y="Employees", alt=f"Recruitment records by {_title(field).lower()}.")
                st.dataframe(distribution, hide_index=True, alt=f"Recruitment records by {_title(field).lower()}.")
        if "application_date" in frame:
            dates = pd.to_datetime(frame["application_date"], errors="coerce").dropna()
            if not dates.empty:
                trend = dates.dt.to_period("M").astype(str).value_counts().sort_index().rename_axis("Month").reset_index(name="Applications")
                st.markdown("### Applications over time")
                st.line_chart(trend, x="Month", y="Applications", alt="Recruitment records by application month.")
        st.caption("Recruitment totals describe available application records; they may not equal unique candidates or hires.")
        return

    st.info("This analytics module is not available for this dataset.")
