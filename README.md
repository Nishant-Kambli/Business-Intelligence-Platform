# People Intelligence Platform

## Project Overview

People Intelligence is a workforce analytics application that adapts its dashboard to the fields in an uploaded HR CSV. The backend reads and cleans the file, maps recognized headers to canonical fields, and enables the analytics modules supported by that dataset. New aliases and dataset rules can be added through JSON configuration.

The platform combines descriptive workforce analytics with independently registered attrition models. Attrition is the only prediction task with trained models; other workforce topics provide descriptive analytics.

## Architecture

```text
CSV upload or configured source
        ↓
Pandas reader → Header normalization and alias registry → Cleaning engine
        ↓
Module registry checks required and optional fields
        ↓
Dataset-driven analytics + compatible models from the model registry
        ↓
Executive dashboard, reports, and navigation
```

Business logic is in reusable `ai_bi/` services. Pages focus on presentation. Dataset aliases and capability requirements live in JSON configuration, and prediction models have separate registry declarations.

```text
.
├── .streamlit/
│   └── config.toml                   # Global light theme
├── ai_bi/
│   ├── datasets/
│   │   └── workforce_v1.json         # Canonical aliases and dataset rules
│   ├── models/
│   │   └── attrition_prediction.json # Original IBM model registry entry
│   ├── analytics.py                  # KPIs, summaries, and insights
│   ├── app_state.py                  # Active dataset selection
│   ├── charts.py                     # Reusable visualizations
│   ├── cleaning_service.py           # Type conversion and missing values
│   ├── config.py                     # Shared paths and configuration types
│   ├── data.py                       # Dataset loading and feature helpers
│   ├── dataset_registry.py           # Alias mapping and schema detection
│   ├── model_registry.py             # Model discovery and compatibility checks
│   ├── model_service.py              # Saved-model loading and inference
│   ├── module_analytics.py           # Shared analytics page rendering
│   ├── module_registry.py            # Module requirements
│   ├── modules.json                  # Module definitions and page registrations
│   ├── report_service.py             # Executive report and file exports
│   ├── training_service.py           # Offline, per-dataset attrition training
│   ├── upload_service.py             # CSV parsing and schema detection
│   └── utils.py                      # Shared styling and formatting helpers
├── app.py                            # Streamlit entry point and navigation
├── app_pages/
│   ├── dashboard.py                  # Executive dashboard
│   ├── workforce_analytics.py        # Workforce composition and experience
│   ├── compensation_analytics.py     # Salary summaries and distributions
│   ├── employee_engagement.py        # Engagement and satisfaction measures
│   ├── performance_intelligence.py   # Ratings and performance measures
│   ├── training_analytics.py         # Training outcomes, duration, and cost
│   ├── recruitment_analytics.py      # Applications, hiring, and source metrics
│   ├── retention_intelligence.py     # Historical attrition patterns
│   ├── ai_prediction_center.py       # Registry-driven decision support
│   ├── reports.py                    # Executive reports and downloads
│   └── data_sources.py               # CSV selection, upload, and data explorer
├── artifacts/                        # Existing evaluation and feature-importance outputs
├── assets/                            # Logo and global stylesheet
├── data/
│   ├── Cleaned_HR_Data_Analysis.csv  # Workforce, engagement, performance, and training data
│   └── train.csv                      # Attrition training dataset (59,598 rows)
├── models/                            # Dataset-specific model artifacts and metadata
├── tests/
│   └── test_app_contracts.py          # Application contract tests
├── IBM_HR_Cleaned.csv                # Configured default source and notebook input
├── attrition_model.pkl               # Original IBM HR attrition pipeline
├── AI_Business_Intelligence_Platform.ipynb # Exploratory analysis and original model workflow
├── README.md
└── requirements.txt
```

## Dataset

The configured default, `IBM_HR_Cleaned.csv`, contains 1,441 employee records and 35 columns. `data/train.csv` contains 59,598 rows with attrition outcomes (`Left` and `Stayed`). `data/Cleaned_HR_Data_Analysis.csv` contains 2,845 records about workforce, engagement, performance, and training, but has no attrition target. Uploads are parsed with pandas; recognized headers are mapped through the alias registry, and unrecognized headers are retained in normalized snake_case form. Analytics modules are enabled according to the fields detected in each dataset.

Processing removes exact duplicate rows, normalizes text and null markers, converts recognized numeric and date fields, imputes missing values, and standardizes supported attrition labels. A cleaning summary is kept for the Data Sources page. Workforce, compensation, engagement, performance, training, recruitment, and attrition pages are independently enabled by the available fields.

## Machine Learning Pipeline

The project has saved, class-balanced scikit-learn pipelines for **Attrition Prediction**. The original IBM HR model is saved as `attrition_model.pkl`. A separate model trained from `data/train.csv` is registered under `models/`; each saved artifact includes its preprocessing pipeline. The registry offers a model only when the active dataset contains its required features.

The Streamlit app loads the saved model for predictions; it does not train or retrain models when it starts or when a user uploads data. Workforce composition, salary, engagement, and historical retention views are descriptive analytics. Salary, satisfaction, or other outcomes do not have trained prediction models yet. Each future model needs a clearly defined target, suitable training examples, validation, and a separately registered artifact.

Each model’s input fields, target, positive class, artifact path, and metrics are stored in its registry metadata. The IBM model metadata is in `ai_bi/models/`; the `train.csv` model has a matching JSON file in `models/`. Model training runs offline; uploading a dataset does not automatically retrain a model.

## Model Comparison

The notebook’s original comparison used the IBM HR dataset. Its held-out test metrics for Attrition = Yes are:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.785 | 0.405 | 0.681 | 0.508 | 0.797 |
| Random Forest | 0.844 | 0.533 | 0.340 | 0.416 | 0.747 |
| Decision Tree | 0.706 | 0.243 | 0.383 | 0.298 | 0.581 |

Logistic Regression had the strongest held-out Attrition=Yes F1-score in that comparison. Random Forest had higher accuracy and precision. These metrics describe the notebook’s evaluation split and are not a guarantee of future performance.

The separately registered `data/train.csv` model uses 59,598 rows with `Left` treated as the positive attrition outcome. It was selected by stratified five-fold cross-validation F1-score. Held-out test metrics are:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (selected) | 0.748 | 0.727 | 0.754 | 0.740 | 0.841 |
| Random Forest | 0.744 | 0.719 | 0.760 | 0.739 | 0.838 |
| Decision Tree | 0.730 | 0.697 | 0.762 | 0.728 | 0.810 |

The selected model achieved cross-validation F1 of 0.744. It is registered independently from the IBM model and can be used when `data/train.csv` or a compatible dataset is active.

## Features

- Executive dashboard with data-derived KPIs, insights, alerts, department summary, and workforce overview.
- Workforce, compensation, engagement, performance, training, recruitment, and attrition analytics that appear when the dataset supports them.
- CSV upload with schema detection, automatic processing, data health checks, and compatible-module status.
- Decision Support with friendly grouped inputs, risk score, probability, available model factors, and supportive HR actions.
- Executive reports with summaries, insights, KPI tables, charts, and CSV, Excel, and PDF downloads.
- Dataset explorer with sample rows, shape, column types, missingness, and descriptive statistics.
- Model registry designed to add independent prediction modules as separately trained models become available.

## Installation

Use a Python environment compatible with the installed scikit-learn version used to serialize the model. Install dependencies and launch Streamlit:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app uses `IBM_HR_Cleaned.csv` as its configured default. You can also select project CSVs from the Data Sources page or upload a CSV in the app. Only load pickle files from a trusted source.

## Screenshots

Screenshots can be added after launching the app. Suggested views: Executive Dashboard, Workforce Analytics, Training Analytics, Performance Intelligence, AI Prediction Center, and Data Sources.

## Future Work

- Add dataset configurations and test them against representative CSV samples.
- Evaluate attrition predictions on newer, time-based data and reserve a final independent test set.
- Calibrate probabilities and choose supportive outreach thresholds with HR.
- Monitor data quality and model performance across relevant employee groups.
- Train additional prediction models only after defining the target, collecting appropriate labels, and validating the models.
- Measure outcomes from retention interventions.

## License

No license has been selected. Add a license file and update this section before redistribution.
