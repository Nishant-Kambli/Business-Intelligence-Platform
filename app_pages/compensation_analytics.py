"""Compensation distribution and group comparisons."""

import pandas as pd
import streamlit as st

from ai_bi.analytics import field_distribution, group_summary
from ai_bi.app_state import active_dataset
from ai_bi.utils import page_header, setup_page


setup_page("Compensation Analytics")
frame, config, report, source_name = active_dataset()
page_header("People Intelligence / Analytics", "Compensation Analytics", "Review pay distributions in the units and categories supplied by the dataset.")

salary = config.salary_column
department = config.department_column
role = config.role_column
if not salary or salary not in frame:
    st.info("Compensation Analytics is unavailable because a salary or pay field was not detected.")
else:
    numeric = pd.to_numeric(frame[salary], errors="coerce")
    is_numeric_pay = numeric.notna().sum() > 0 and numeric.notna().sum() >= frame[salary].notna().sum() * 0.8
    if is_numeric_pay:
        values = numeric.dropna()
        a, b, c = st.columns(3)
        a.metric("Average pay", f"{values.mean():,.0f}" if not values.empty else "—", border=True)
        b.metric("Median pay", f"{values.median():,.0f}" if not values.empty else "—", border=True)
        spread = values.quantile(.75) - values.quantile(.25) if not values.empty else None
        c.metric("Interquartile range", f"{spread:,.0f}" if spread is not None else "—", border=True)
        st.caption("Pay is shown in dataset units. No currency or annualization is assumed.")
        st.markdown("### Pay distribution")
        distribution = field_distribution(frame.assign(**{salary: numeric}), salary)
        st.bar_chart(distribution, x="Value", y="Employees", alt="Employee distribution across pay values or ranges.")
        for dimension in (department, role):
            if dimension and dimension in frame:
                summary = group_summary(frame.assign(**{salary: numeric}), dimension, salary)
                st.markdown(f"### Average pay by {dimension.replace('_', ' ')}")
                st.bar_chart(summary, x="Group", y="Value", horizontal=True, alt=f"Average pay by {dimension.replace('_', ' ')}.")
                st.dataframe(summary.rename(columns={"Value": "Average pay", "Employees": "Records"}), hide_index=True, alt=f"Average pay and record counts by {dimension.replace('_', ' ')}.")
    else:
        distribution = field_distribution(frame, salary)
        st.markdown("### Compensation bands")
        st.bar_chart(distribution, x="Value", y="Employees", alt="Employee distribution across reported compensation bands.")
        st.dataframe(distribution, hide_index=True, alt="Counts for each reported compensation band.")
        if department and department in frame:
            cross_tab = pd.crosstab(frame[department], frame[salary], dropna=False).reset_index()
            st.markdown("### Compensation bands by department")
            st.dataframe(cross_tab, hide_index=True, alt="Compensation band counts by department.")

st.caption("Group differences are descriptive and may reflect differences in role, tenure, or other factors.")
