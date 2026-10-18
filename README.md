# People Intelligence

People Intelligence is a Streamlit dashboard for exploring workforce data. Load an HR CSV to review workforce composition, compensation, engagement, performance, training, and attrition. The available pages depend on the columns in the active dataset.

The project also includes saved machine learning pipelines for attrition prediction. They are offered only when a dataset has the fields a model expects. Other dashboard pages show descriptive analysis; they do not predict employee outcomes.

## What you can do

- Explore workforce metrics with interactive charts and summaries.
- Upload HR data and see which analytics are supported by its columns.
- Review data quality, missing values, and descriptive statistics.
- Generate executive reports and download CSV, Excel, or PDF exports.
- Use a compatible registered model to estimate attrition risk.

## Run locally

You’ll need Python and Git. The app expects its default data file at `IBM_HR_Cleaned.csv` in the project root. CSV files are excluded from this repository, so provide your own compatible HR dataset at that path before launching the app.

```bash
git clone https://github.com/Nishant-Kambli/Business-Intelligence-Platform.git
cd Business-Intelligence-Platform
python -m venv .venv
```

Activate the environment, install the requirements, and start Streamlit:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

**macOS or Linux**

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Once the app is open, use **Data Sources** to upload another CSV. The app maps recognized column names to its standard fields and enables pages when the required data is present. Your uploaded dataset is kept in the current Streamlit session.

## Project layout

```text
app.py                 Streamlit entry point and navigation
app_pages/             Dashboard, analytics, reports, and data pages
ai_bi/                 Data cleaning, analytics, registries, and services
ai_bi/datasets/        Dataset field aliases and requirements
ai_bi/models/          Model registry metadata
models/                Additional model artifacts and metadata
artifacts/              Saved evaluation outputs
assets/                 Logo and styles
tests/                  Application contract tests
```

## Models and data

The application loads saved scikit-learn pipelines for inference; it does not train a model when it starts or when a CSV is uploaded. Model training is a separate workflow. A model appears in the prediction page only when its registry entry is active, its artifact is present, and the current dataset meets its feature requirements.

CSV files are ignored by Git to keep employee data out of the repository. Add your dataset locally; don’t commit private or identifiable employee information. Model files use Python pickle formats, so only run artifacts from sources you trust.
