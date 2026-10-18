"""Upload, automatic processing, capability health, and data exploration."""

import hashlib

import pandas as pd
import streamlit as st

from ai_bi.app_state import active_dataset, clear_uploaded_dataset, dataset_configs, set_uploaded_dataset
from ai_bi.cleaning_service import clean_dataset
from ai_bi.config import PROJECT_ROOT
from ai_bi.data import missing_value_summary
from ai_bi.model_registry import active_models, model_compatibility
from ai_bi.upload_service import process_upload
from ai_bi.utils import page_header, setup_page


MODULE_LABELS = {
    "workforce": "Workforce Analytics",
    "compensation": "Compensation Analytics",
    "engagement": "Employee Engagement",
    "performance": "Performance Intelligence",
    "training": "Training Analytics",
    "recruitment": "Recruitment Analytics",
    "retention": "Attrition Analytics",
    "ai_prediction": "AI Prediction",
}

FIELD_LABELS = {
    "age": "Age", "department": "Department", "business_unit": "Business unit", "division": "Division",
    "salary": "Salary", "job_role": "Job role", "title": "Title", "experience": "Experience",
    "attrition": "Attrition outcome", "engagement_score": "Engagement score", "satisfaction_score": "Satisfaction score",
    "performance_score": "Performance score", "employee_rating": "Employee rating", "training_program": "Training program",
    "training_outcome": "Training outcome", "training_cost": "Training cost", "applicant_id": "Applicant ID",
    "application_date": "Application date", "hiring_status": "Hiring status", "source_channel": "Recruitment source",
}


def activate_source(source, source_name: str, signature: str, configs) -> None:
    """Read, clean, and activate a selected CSV for every application page."""
    with st.spinner("Processing dataset…"):
        uploaded = process_upload(source, configs)
        if uploaded.error_code:
            if uploaded.error_code in {"unsupported_schema", "ambiguous_columns"}:
                st.error("This dataset could not be matched to a supported HR schema. Check that it includes recognizable workforce fields such as department, salary, age, experience, job role, or attrition.")
            else:
                st.error("This file could not be processed. Please choose a valid, non-empty CSV.")
            return
        try:
            cleaned = clean_dataset(uploaded.frame, uploaded.config)
            cleaned.report["dataset_name"] = uploaded.config.dataset_name
            cleaned.report["source_name"] = source_name
            set_uploaded_dataset(cleaned.frame, uploaded.config, cleaned.report, raw_frame=uploaded.frame)
            st.session_state["processed_upload_signature"] = signature
        except ValueError:
            st.error("This dataset could not be cleaned. Please review its columns and try again.")
            return
    st.rerun()


setup_page("Data Sources")
page_header("People Intelligence / Data", "Choose a Workforce Dataset", "Choose a CSV already in the project or upload a new HR dataset. The platform processes it and refreshes compatible analytics automatically.")

configs = dataset_configs()
source_choice = st.radio(
    "Data source",
    ["Configured Workforce Dataset", "Choose a Project CSV", "Upload a CSV"],
    horizontal=True,
    label_visibility="collapsed",
    key="source_choice",
)
if source_choice == "Choose a Project CSV":
    csv_paths = sorted(
        [*PROJECT_ROOT.glob("*.csv"), *(PROJECT_ROOT / "data").glob("*.csv")],
        key=lambda path: (path.name.casefold(), str(path.parent).casefold()),
    )
    csv_by_name = {path.name: path for path in csv_paths}
    if csv_paths:
        selected_name = st.selectbox("Dataset file", [path.name for path in csv_paths], index=None, placeholder="Choose a CSV…")
        if selected_name is not None:
            selected_path = csv_by_name[selected_name]
            signature = hashlib.sha256(selected_path.name.encode("utf-8") + selected_path.read_bytes()).hexdigest()
            if signature != st.session_state.get("processed_upload_signature"):
                activate_source(selected_path, selected_path.name, signature, configs)
            else:
                st.success("Dataset processed successfully")
    else:
        st.info("No CSV files were found in the project folder. Upload a CSV to get started.")
elif source_choice == "Upload a CSV":
    upload = st.file_uploader("Choose an HR CSV", type=["csv"], help="The uploaded dataset is kept in this browser session.", key=f"workforce_upload_{st.session_state.get('upload_widget_version', 0)}")
    if upload is not None:
        signature = hashlib.sha256(upload.name.encode("utf-8") + upload.getvalue()).hexdigest()
        if signature != st.session_state.get("processed_upload_signature"):
            activate_source(upload, upload.name, signature, configs)
        else:
            st.success("Dataset processed successfully")

if st.session_state.get("uploaded_dataset") is not None:
    if st.button("Use configured Workforce Dataset"):
        clear_uploaded_dataset()
        st.session_state.pop("processed_upload_signature", None)
        st.session_state["upload_widget_version"] = st.session_state.get("upload_widget_version", 0) + 1
        st.rerun()
    if source_choice == "Configured Workforce Dataset":
        clear_uploaded_dataset()
        st.session_state.pop("processed_upload_signature", None)
        st.rerun()

frame, config, report, source_name = active_dataset()
if source_choice == "Configured Workforce Dataset":
    st.success("Dataset processed successfully")

module_health = report.get("module_health", {})
source_columns = {column for values in report.get("canonical_to_raw", {}).values() for column in values}
compatibilities = [model_compatibility(spec, report.get("available_fields", []), source_columns) for spec in active_models()]
compatible_model = next((result for result in compatibilities if result["compatible"]), None)
module_health["ai_prediction"] = {
    "supported": compatible_model is not None,
    "missing_fields": [] if compatible_model else (compatibilities[0]["missing_fields"] if compatibilities else []),
    "reason": None if compatible_model else "Required model features or a compatible source schema are not available.",
}

st.markdown("### Dataset Compatibility")
st.caption(f"Active source: **{source_name}** · {len(frame):,} records")
cols = st.columns(min(5, max(1, len(MODULE_LABELS))))
for index, (module, label) in enumerate(MODULE_LABELS.items()):
    status = module_health.get(module, {"supported": False})
    with cols[index % len(cols)]:
        st.markdown(f"{'✅' if status.get('supported') else '⛔'} **{label}**")
        if module == "ai_prediction" and not status.get("supported"):
            st.caption("Prediction unavailable: Required model features not found.")
        elif not status.get("supported"):
            missing_groups = status.get("missing_any_of", [])
            missing = status.get("missing_fields", [])
            names = [" / ".join(FIELD_LABELS.get(field, field.replace("_", " ").title()) for field in group) for group in missing_groups] + [FIELD_LABELS.get(field, field.replace("_", " ").title()) for field in missing]
            st.caption("Needs: " + (", ".join(names) if names else "compatible fields"))

st.markdown("### Dataset Explorer")
tab_rows, tab_schema, tab_quality, tab_stats = st.tabs(["First 10 rows", "Column types", "Missing values", "Descriptive statistics"])
with tab_rows:
    st.dataframe(frame.head(10), hide_index=True, alt="First 10 rows from the processed workforce dataset.")
with tab_schema:
    schema = pd.DataFrame({"Column": frame.columns, "Data type": frame.dtypes.astype(str).values, "Non-missing values": frame.notna().sum().values})
    st.dataframe(schema, hide_index=True, alt="Types and nonmissing counts for processed dataset fields.")
with tab_quality:
    quality = missing_value_summary(frame)
    if quality.empty:
        st.success("No missing or blank values found after processing.")
    else:
        st.dataframe(quality.reset_index(names="Column"), hide_index=True, alt="Missing and blank counts after dataset processing.")
with tab_stats:
    statistics = frame.describe(include="all").transpose().reset_index(names="Column")
    # A mixed-type CSV can put numeric values and timestamps in the same stats
    # column; render the summary as text so Streamlit's Arrow conversion is stable.
    statistics = statistics.astype("string")
    st.dataframe(statistics, hide_index=True, alt="Descriptive statistics for the active dataset.")
