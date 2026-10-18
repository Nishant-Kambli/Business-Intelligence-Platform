"""Registry-driven employee decision support for compatible prediction models."""

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_bi.app_state import active_dataset
from ai_bi.config import PROJECT_ROOT
from ai_bi.data import feature_defaults, feature_options
from ai_bi.model_registry import active_models, model_compatibility
from ai_bi.model_service import load_model, local_contributions, model_feature_columns, model_input_from_canonical, predict_employee
from ai_bi.utils import page_header, setup_page


ORDINAL_LABELS = {
    "education": {"1": "Below college", "2": "College", "3": "Bachelor's degree", "4": "Master's degree", "5": "Doctorate"},
    "job_satisfaction": {"1": "Low", "2": "Medium", "3": "High", "4": "Very high"},
    "environment_satisfaction": {"1": "Low", "2": "Medium", "3": "High", "4": "Very high"},
    "relationship_satisfaction": {"1": "Low", "2": "Medium", "3": "High", "4": "Very high"},
    "job_involvement": {"1": "Low", "2": "Medium", "3": "High", "4": "Very high"},
    "work_life_balance": {"1": "Poor", "2": "Fair", "3": "Good", "4": "Excellent"},
    "performance_rating": {"1": "Low", "2": "Good", "3": "Excellent", "4": "Outstanding"},
    "job_level": {"1": "Level 1", "2": "Level 2", "3": "Level 3", "4": "Level 4", "5": "Level 5"},
    "stock_option_level": {"0": "None", "1": "Basic", "2": "Standard", "3": "Enhanced"},
    "business_travel": {"Non-Travel": "Does not travel", "Travel_Rarely": "Travels occasionally", "Travel_Frequently": "Travels frequently"},
    "overtime": {"No": "No", "Yes": "Yes"},
}


@st.cache_resource(show_spinner="Preparing decision support…")
def cached_model(path: str, modified_ns: int, size: int):
    return load_model(path)


def _friendly(value, labels):
    return labels.get(str(value), str(value))


def _model_value(value, default):
    """Keep numeric form selections numeric for the saved pipeline input."""
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        try:
            numeric = float(value)
            return int(numeric) if numeric.is_integer() else numeric
        except (TypeError, ValueError):
            pass
    return value


def _factor_label(name: str, config, field_labels: dict[str, str]) -> str:
    cleaned = str(name).replace("categorical__", "").replace("remainder__", "")
    for field, aliases in config.aliases.items():
        for alias in (field, *aliases):
            prefix = str(alias) + "_"
            if cleaned.casefold().startswith(prefix.casefold()):
                value = cleaned[len(prefix):].replace("_", " ")
                return f"{field_labels.get(field, field.replace('_', ' ').title())}: {value}"
            if cleaned.casefold() == str(alias).casefold():
                return field_labels.get(field, field.replace("_", " ").title())
    return cleaned.replace("_", " ").title()


setup_page("AI Prediction Center")
frame, config, report, source_name = active_dataset()
page_header("People Intelligence / Decision Support", "AI Prediction Center", "Review an employee assessment and use it as one input to a supportive, human-led conversation.")

raw_columns = {column for values in report.get("canonical_to_raw", {}).values() for column in values}
specs = [
    spec for spec in active_models()
    if model_compatibility(spec, report.get("available_fields", []), raw_columns)["compatible"]
    and (PROJECT_ROOT / spec.artifact).exists()
]
if not specs:
    st.info("Prediction unavailable\n\nReason: Required model features not found.")
else:
    # active_models is ranked by recorded cross-validation F1, then ROC-AUC and test F1.
    # The first compatible entry is selected automatically for the active dataset.
    spec = specs[0]
    st.caption(f"Selected compatible model: **{spec.display_name}**" + (f" · Dataset: {spec.dataset}" if spec.dataset else ""))
    artifact_path = PROJECT_ROOT / spec.artifact
    artifact_stat = artifact_path.stat()
    model = cached_model(str(artifact_path), artifact_stat.st_mtime_ns, artifact_stat.st_size)
    model_columns = model_feature_columns(model)
    defaults = feature_defaults(frame, list(spec.required_fields))
    options = feature_options(frame, list(spec.required_fields))
    labels = spec.field_labels or {}
    configured_choices = spec.field_choices or {}
    groups = spec.input_groups or {"Employee Profile": spec.required_fields}
    record = dict(defaults)
    with st.form(f"prediction_form_{spec.model_id}"):
        st.caption("Provide the fields shown below. Other model inputs use representative values from the selected dataset.")
        for section, fields in groups.items():
            with st.expander(section, expanded=section == next(iter(groups))):
                columns = st.columns(2)
                for index, field in enumerate(fields):
                    if field not in spec.required_fields:
                        continue
                    with columns[index % 2]:
                        label = labels.get(field, field.replace("_", " ").title())
                        default = defaults.get(field, 0)
                        choices = options.get(field, [])
                        display_map = configured_choices.get(field, ORDINAL_LABELS.get(field, {}))
                        if choices:
                            values = [value for value in choices]
                            selected = next((value for value in values if str(value) == str(default)), values[0])
                            chosen = st.selectbox(
                                label, values, index=values.index(selected),
                                format_func=lambda value, mapping=display_map: _friendly(value, mapping),
                                key=f"profile_{spec.model_id}_{field}",
                            )
                            record[field] = _model_value(chosen, default)
                        elif field in display_map:
                            valid = list(display_map)
                            current = next((value for value in valid if str(value) == str(default)), valid[0])
                            chosen = st.selectbox(label, valid, index=valid.index(current), format_func=lambda value, mapping=display_map: _friendly(value, mapping), key=f"profile_{spec.model_id}_{field}")
                            record[field] = _model_value(chosen, default)
                        else:
                            try:
                                numeric_default = int(round(float(default)))
                            except (TypeError, ValueError):
                                numeric_default = 0
                            numeric_values = pd.to_numeric(frame[field], errors="coerce") if field in frame else pd.Series(dtype=float)
                            lower = int(numeric_values.min()) if numeric_values.notna().any() else numeric_default
                            upper = int(numeric_values.max()) if numeric_values.notna().any() else numeric_default + 1
                            if lower == upper:
                                upper = lower + 1
                            record[field] = st.slider(label, lower, upper, min(max(numeric_default, lower), upper), key=f"profile_{spec.model_id}_{field}")
        submitted = st.form_submit_button(f"Assess {spec.target.replace('_', ' ').title()}", type="primary")

    if submitted:
        try:
            model_record = model_input_from_canonical(record, config, model_columns)
            result = predict_employee(model, model_record, model_columns, spec.positive_label or "Yes")
            st.markdown("### Risk Assessment")
            risk_class = "risk-high" if result["prediction"] == (spec.positive_label or "Yes") else "risk-low"
            st.markdown(
                f'<div class="risk-card {risk_class}"><h2>{result["risk_level"]}</h2>'
                f'<p><strong>Probability:</strong> {result["probability"]:.1%}</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown("### Top Contributing Factors")
            factors = local_contributions(model, model_record, positive_label=spec.positive_label or "Yes")
            if factors:
                for factor in factors:
                    direction = "associated with a higher score" if factor["contribution"] > 0 else "associated with a lower score"
                    st.write(f"• {_factor_label(factor['feature'], config, labels)} — {direction}")
            else:
                st.caption("Contributing factors are not available for this assessment.")
            st.markdown("### Suggested HR Actions")
            st.write("Offer a private, supportive conversation. Ask what would improve the employee’s experience, and discuss workload, development, recognition, and flexibility based on their feedback.")
            assessment = pd.DataFrame([{"Risk level": result["risk_level"], "Positive outcome probability": result["probability"], "Source": source_name}])
            st.download_button("Export Assessment CSV", assessment.to_csv(index=False).encode("utf-8"), f"{spec.model_id}_assessment.csv", "text/csv")
        except (ValueError, KeyError):
            st.info("Prediction unavailable\n\nReason: Required model features not found.")

st.markdown('<div class="disclaimer">This prediction is decision support only. It is not a diagnosis, causal explanation, or basis for an employment decision.</div>', unsafe_allow_html=True)
