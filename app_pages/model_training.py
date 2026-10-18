"""Train and register a dataset-specific attrition model on demand."""

import pandas as pd
import streamlit as st

from ai_bi.app_state import active_dataset, active_raw_dataset
from ai_bi.dataset_registry import map_headers
from ai_bi.training_service import train_attrition_model
from ai_bi.utils import page_header, setup_page


setup_page("Model Training")
frame, config, report, source_name = active_dataset()
raw_frame, raw_config, raw_source_name = active_raw_dataset()
page_header(
    "People Intelligence / Model Lab",
    "Train an Attrition Model",
    "Train an independent model from this dataset. Features are selected from its recognized workforce fields and preprocessing is fitted within training folds.",
)

if raw_config.target_column is None:
    st.info("Model training is unavailable because this dataset has no configured attrition target.")
else:
    mapping = map_headers(raw_frame.columns, raw_config)
    target_fields = [
        raw for raw, canonical in mapping["raw_to_canonical"].items()
        if canonical == raw_config.target_column
    ]
    if not target_fields:
        st.info("Model training is unavailable because no recognized attrition target column was found in this dataset.")
    else:
        default_target = raw_config.target_column if raw_config.target_column in target_fields else target_fields[0]
        target_column = st.selectbox(
            "Target column",
            target_fields,
            index=target_fields.index(default_target),
            help="Choose which recognized target column contains the employee outcome.",
        )
        outcomes = raw_frame[target_column].dropna().astype(str).str.strip()
        outcome_options = sorted(outcomes.unique().tolist(), key=str.casefold)
        if len(outcome_options) != 2:
            st.warning(f"This column has {len(outcome_options)} observed outcomes. Attrition training requires exactly two.")
        else:
            positive_defaults = {value.casefold() for value in raw_config.target_positive_values}
            default_positive = next((value for value in outcome_options if value.casefold() in positive_defaults), outcome_options[-1])
            positive_label = st.selectbox(
                "Which outcome means the employee left?",
                outcome_options,
                index=outcome_options.index(default_positive),
            )
            eligible_fields = [
                field for field in frame.columns
                if field != raw_config.target_column and field != raw_config.employee_id_column
            ]
            st.caption(
                f"Source: **{raw_source_name or source_name}** · {len(raw_frame):,} source rows · "
                f"{len(eligible_fields):,} canonical columns available. ID, name, date-only, target, and unrecognized fields are excluded from model features."
            )
            train_button = st.button("Train and Register Model", type="primary", disabled=len(outcome_options) != 2)
            if train_button:
                try:
                    with st.spinner("Training and comparing balanced models with stratified cross-validation…"):
                        result = train_attrition_model(
                            source=raw_frame,
                            config=raw_config,
                            dataset_name=raw_source_name or source_name,
                            target_column=target_column,
                            positive_label=positive_label,
                        )
                    st.session_state["last_training_result"] = {
                        "model_id": result["model_id"],
                        "artifact_path": str(result["artifact_path"]),
                        "metadata_path": str(result["metadata_path"]),
                        "dataset_name": raw_source_name or source_name,
                        "best_model": result["best_model"],
                        "metrics": result["metrics"],
                        "comparisons": result["comparisons"].to_dict(orient="records"),
                        "features": result["features"],
                        "rows": result["rows"],
                    }
                    st.rerun()
                except (ValueError, OSError, RuntimeError) as error:
                    st.error(f"Training could not be completed: {error}")

training_result = st.session_state.get("last_training_result")
if training_result and training_result.get("dataset_name") != (raw_source_name or source_name):
    training_result = None
if training_result:
    st.markdown("### Latest Training Result")
    st.success(f"Registered **{training_result['best_model']}** for {source_name}.")
    st.caption(f"Saved model: `{training_result['model_id']}` · {training_result['rows']:,} labeled rows · {len(training_result['features']):,} supported features")
    results = pd.DataFrame(training_result["comparisons"])
    display_columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1"]
    st.dataframe(
        results[display_columns].rename(columns={
            "model": "Model", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
            "f1": "F1-score", "roc_auc": "ROC-AUC", "cv_f1": "CV F1-score",
        }).style.format({column: "{:.3f}" for column in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "CV F1-score"]}),
        hide_index=True,
        alt="Held-out test and cross-validation metrics for balanced attrition classifiers.",
    )
    with st.expander("Selected model settings and saved metadata"):
        st.write(f"Selected by {training_result['metrics']['cv_f1']:.3f} stratified CV F1-score; held-out test F1 is {training_result['metrics']['f1']:.3f}.")
        st.json(training_result["metrics"]["best_params"])
        st.write("Confusion matrix (actual rows: No, Yes; predicted columns: No, Yes)")
        st.dataframe(pd.DataFrame(training_result["metrics"]["confusion_matrix"], index=["Actual No", "Actual Yes"], columns=["Predicted No", "Predicted Yes"]))
        st.write(f"Model artifact: `{training_result['artifact_path']}`")
        st.write(f"Registry metadata: `{training_result['metadata_path']}`")

st.caption("Models are trained and registered per dataset. A model is offered for prediction only when every required canonical feature is present.")
