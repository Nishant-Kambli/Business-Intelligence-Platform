"""Streamlit entry point with dataset-driven module navigation."""

import streamlit as st

from ai_bi.app_state import active_dataset
from ai_bi.config import APP_TITLE, LOGO_PATH
from ai_bi.model_registry import active_models, model_compatibility
from ai_bi.module_registry import load_modules
from ai_bi.utils import inject_styles


st.set_page_config(page_title=APP_TITLE, page_icon=str(LOGO_PATH), layout="wide")
inject_styles()
frame, _config, report, _source_name = active_dataset()
available_fields = set(report.get("available_fields", []))
source_columns = {column for values in report.get("canonical_to_raw", {}).values() for column in values}
health = report.get("module_health", {})
module_specs = load_modules()

overview_pages = [st.Page("app_pages/dashboard.py", title="Executive Dashboard", icon=":material/dashboard:", default=True)]
analytics_pages = []
for module_id in ("workforce", "compensation", "engagement", "performance", "training", "recruitment"):
    spec = module_specs.get(module_id, {})
    if health.get(module_id, {}).get("supported") and spec.get("page"):
        analytics_pages.append(st.Page(spec["page"], title=spec["title"], icon=spec.get("icon")))

retention_pages = []
retention_spec = module_specs.get("retention", {})
if health.get("retention", {}).get("supported") and retention_spec.get("page"):
    retention_pages.append(st.Page(retention_spec["page"], title=retention_spec["title"], icon=retention_spec.get("icon")))

prediction_pages = []
for spec in active_models():
    capability = model_compatibility(spec, available_fields, source_columns)
    if capability["compatible"]:
        prediction_pages.append(st.Page("app_pages/ai_prediction_center.py", title="AI Prediction Center", icon=":material/psychology:"))
        break

pages = {"Overview": overview_pages}
if analytics_pages:
    pages["Analytics"] = analytics_pages
if retention_pages or prediction_pages:
    pages["Retention & AI"] = retention_pages + prediction_pages
pages["Reports & Data"] = [
    st.Page("app_pages/reports.py", title="Executive Reports", icon=":material/description:"),
    st.Page("app_pages/data_sources.py", title="Data Sources", icon=":material/database:"),
]

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=72)
    st.markdown(f"### {APP_TITLE}")
    st.caption("Enterprise workforce intelligence")
    st.caption(f"Active dataset: {_source_name} · {len(frame):,} records")

navigation = st.navigation(pages, position="sidebar")
navigation.run()
