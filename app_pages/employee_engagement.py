"""Engagement indicators based on recorded satisfaction and work-life fields."""

import pandas as pd
import streamlit as st

from ai_bi.analytics import field_distribution
from ai_bi.app_state import active_dataset
from ai_bi.utils import page_header, setup_page


setup_page("Employee Engagement")
frame, config, report, source_name = active_dataset()
page_header("People Intelligence / Analytics", "Employee Engagement", "Explore recorded satisfaction and work-life indicators in aggregate.")

available = [field for field in config.satisfaction_columns if field in frame]
if not available:
    st.warning("This dataset does not include configured satisfaction measures.")
else:
    labels = {field: field.replace("_", " ").title() for field in available}
    st.markdown("### Engagement indicators")
    cards = st.container(horizontal=True, gap="medium")
    with cards:
        for field in available[:4]:
            values = pd.to_numeric(frame[field], errors="coerce").dropna()
            if not values.empty:
                cards.metric(labels[field], f"{values.mean():.2f}", help="Average on the scale recorded in the source dataset.", border=True)
    selected = st.selectbox("Indicator", available, format_func=lambda value: labels[value])
    scores = pd.to_numeric(frame[selected], errors="coerce").dropna()
    distribution = field_distribution(frame, selected)
    st.bar_chart(distribution, x="Value", y="Employees", alt=f"Response distribution for {labels[selected]}.")

    department = config.department_column
    if department and department in frame:
        st.markdown("### By department")
        if not scores.empty:
            grouped = frame[[department, selected]].copy()
            grouped[selected] = pd.to_numeric(grouped[selected], errors="coerce")
            summary = grouped.groupby(department, dropna=False)[selected].agg(Average="mean", Responses="count").reset_index().rename(columns={department: "Department"})
            st.bar_chart(summary, x="Department", y="Average", horizontal=True, alt=f"Average {labels[selected].lower()} by department.")
            st.dataframe(summary, hide_index=True, alt=f"Average {labels[selected].lower()} and response counts by department.")
        else:
            cross_tab = pd.crosstab(frame[department], frame[selected], dropna=False).reset_index()
            st.dataframe(cross_tab, hide_index=True, alt=f"Recorded {labels[selected].lower()} categories by department.")

st.caption("These are recorded survey or work-condition measures. Averages may hide differences within groups and do not prove causes.")
