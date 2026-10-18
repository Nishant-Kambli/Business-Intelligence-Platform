"""Historical attrition analysis from the current workforce dataset."""

import streamlit as st

from ai_bi.analytics import attrition_by, field_distribution
from ai_bi.app_state import active_dataset
from ai_bi.utils import page_header, setup_page


setup_page("Attrition Analytics")
frame, config, report, source_name = active_dataset()
page_header("People Intelligence / Analytics", "Attrition Analytics", "Understand historical employee exits across the groups available in this dataset.")

target = config.target_column
if not target or target not in frame:
    st.info("Attrition Analytics is unavailable because an attrition or exit outcome was not detected.")
else:
    dimensions = [field for field in (config.department_column, config.role_column, *config.categorical_columns) if field and field != target and field in frame]
    dimensions = list(dict.fromkeys(dimensions))
    if dimensions:
        dimension = st.selectbox("Compare historical exits by", dimensions, format_func=lambda value: value.replace("_", " ").title())
        summary = attrition_by(frame, dimension)
        st.bar_chart(summary, x="Group", y="Attrition rate", horizontal=True, alt=f"Historical attrition rate by {dimension.replace('_', ' ')}.")
        st.dataframe(summary.sort_values("Attrition rate", ascending=False), hide_index=True, alt=f"Historical attrition rates and employee counts by {dimension.replace('_', ' ')}.")
    else:
        distribution = field_distribution(frame, target)
        st.bar_chart(distribution, x="Value", y="Employees", alt="Distribution of historical exit outcomes.")

    st.caption("This view summarizes historical records. It cannot establish causes or show change over time without dated snapshots.")
