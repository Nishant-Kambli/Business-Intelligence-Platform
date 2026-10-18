"""Workforce composition and experience summaries from available fields."""

import streamlit as st

from ai_bi.analytics import field_distribution, group_summary
from ai_bi.app_state import active_dataset
from ai_bi.utils import page_header, setup_page


setup_page("Workforce Analytics")
frame, config, report, source_name = active_dataset()
page_header("People Intelligence / Analytics", "Workforce Analytics", "Explore workforce composition and experience measures available in this dataset.")

dimensions = [field for field in (config.department_column, "business_unit", "division", config.role_column, "title", "gender", "employment_status") if field and field in frame]
if dimensions:
    dimension = st.selectbox("Workforce breakdown", dimensions, format_func=lambda value: value.replace("_", " ").title())
    counts = field_distribution(frame, dimension, entity_field=config.employee_id_column)
    st.bar_chart(counts, x="Value", y="Employees", horizontal=True, alt=f"Employee distribution by {dimension.replace('_', ' ')}.")
    st.dataframe(counts, hide_index=True, alt=f"Employee counts by {dimension.replace('_', ' ')}.")

numeric_profiles = [field for field in ("age", *config.experience_columns) if field in frame]
if numeric_profiles:
    st.markdown("### Workforce profile")
    selected = st.selectbox("Profile measure", list(dict.fromkeys(numeric_profiles)), format_func=lambda value: value.replace("_", " ").title())
    distribution = field_distribution(frame, selected, entity_field=config.employee_id_column)
    st.bar_chart(distribution, x="Value", y="Employees", alt=f"Distribution of {selected.replace('_', ' ')}.")
    if dimensions:
        summary = group_summary(frame, config.department_column if config.department_column in frame else dimensions[0], selected)
        if not summary.empty:
            st.markdown(f"### Average {selected.replace('_', ' ')} by group")
            st.bar_chart(summary, x="Group", y="Value", horizontal=True, alt=f"Average {selected.replace('_', ' ')} by workforce group.")
            st.dataframe(summary.rename(columns={"Value": "Average", "Employees": "Records"}), hide_index=True, alt=f"Average {selected.replace('_', ' ')} and record counts by group.")

st.caption("Summaries describe the uploaded records. They do not explain why a workforce pattern occurred.")
