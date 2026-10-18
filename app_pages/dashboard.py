"""Dataset-driven executive workforce dashboard."""

import pandas as pd
import streamlit as st

from ai_bi.analytics import dataset_alerts, department_summary, executive_insights, field_distribution, workforce_kpis
from ai_bi.app_state import active_dataset
from ai_bi.utils import format_percent, page_header, setup_page


setup_page("Executive Dashboard")
frame, config, report, source_name = active_dataset()
kpis = workforce_kpis(frame)
page_header("People Intelligence", "Executive Dashboard", f"Workforce overview for {source_name}.")

st.markdown("### Executive KPIs")
metrics = [("Employees", f"{kpis['employees']:,}")]
if kpis["departments"] is not None:
    metrics.append(("Departments", f"{kpis['departments']:,}"))
if kpis["attrition_rate"] is not None:
    metrics.append(("Historical attrition", format_percent(kpis["attrition_rate"])))
if kpis["average_age"] is not None:
    metrics.append(("Average age", f"{kpis['average_age']:.1f}"))
if kpis["average_salary"] is not None:
    metrics.append(("Average salary", f"{kpis['average_salary']:,.0f}"))
for field, label in (("engagement_score", "Average engagement"), ("satisfaction_score", "Average satisfaction"), ("work_life_balance_score", "Average work-life balance")):
    if field in frame:
        values = pd.to_numeric(frame[field], errors="coerce").dropna()
        if not values.empty:
            metrics.append((label, f"{values.mean():.2f}"))
            break
if "training_outcome" in frame:
    metrics.append(("Training records", f"{frame['training_outcome'].notna().sum():,}"))
if "performance_score" in frame:
    performance = frame["performance_score"].dropna().astype(str)
    if not performance.empty:
        metrics.append(("Most common performance", performance.mode().iloc[0]))
cards = st.container(horizontal=True, gap="medium")
with cards:
    for label, value in metrics:
        st.metric(label, value, border=True)
st.caption("Salary is shown in dataset units. No currency is assumed.")

st.markdown("### Key Insights")
insights = executive_insights(frame, config.satisfaction_columns)
left, right = st.columns(2)
for index, insight in enumerate(insights[:6]):
    with (left if index % 2 == 0 else right):
        st.info(insight, icon="ℹ️")

st.markdown("### Recent Alerts")
summary = department_summary(frame, config)
alerts = dataset_alerts(frame, config)
if alerts:
    for alert in alerts[:4]:
        st.warning(alert, icon="⚠️")
else:
    st.caption("No outcome or status alerts can be calculated from this dataset snapshot.")
st.caption("Alerts describe the active dataset snapshot. Change over time requires dated workforce snapshots.")

if not summary.empty:
    st.markdown("### Department Summary")
    st.dataframe(summary, hide_index=True, alt="Department-level employee counts and available workforce measures.")

st.markdown("### Workforce Overview")
dimension = next((field for field in (config.department_column, "business_unit", "division", config.role_column, "title", "gender", "employment_status") if field and field in frame), None)
if dimension:
    counts = field_distribution(frame, dimension, entity_field=config.employee_id_column)
    st.markdown(f"### Workforce distribution by {dimension.replace('_', ' ')}")
    st.bar_chart(counts, x="Value", y="Employees", horizontal=True, alt=f"Employee distribution by {dimension.replace('_', ' ')}.")
overview_fields = [field for field in ("age", "experience", *config.experience_columns, config.role_column, "title", "gender", "employment_status") if field and field in frame]
overview_fields = list(dict.fromkeys(overview_fields))
if overview_fields:
    selected = st.selectbox("View workforce distribution", overview_fields, format_func=lambda field: field.replace("_", " ").title())
    distribution = field_distribution(frame, selected)
    st.bar_chart(distribution, x="Value", y="Employees", alt=f"Workforce distribution for {selected.replace('_', ' ')}.")
