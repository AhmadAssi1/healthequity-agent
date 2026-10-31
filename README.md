# HealthEquity Agent

An AI agent for analyzing machine learning model performance and fairness, generating evidence-based reports, and monitoring drift.

The initial version focuses on binary classification, using 30-day hospital readmission prediction as a demonstration use case.

## Current Status

The project is under development. Completed work includes:

- Repository setup and automated linting and testing.
- Dataset exploration, binary target creation, and a draft Data Card.
- Data contract, column mapping, dataset validation, and fingerprinting.
- OpenAI connection and function-calling demonstration.
- Patient-level Train, Validation, and Test splits.
- Training-based categorical preprocessing.
- LightGBM baseline training and validation-based model and threshold selection.
- MLflow logging of parameters, metrics, model, and preprocessing configuration.
- Verification that the saved model and configuration reproduce predictions.
- Validation calibration assessment and fixed-model Test evaluation.
- A validated demo audit CSV with provenance metadata under data/samples/.
- Model Registry Version 1 and verification of its reloaded predictions.

The selected experimental baseline uses 15 leaves and a decision
threshold of 0.145. Validation average precision is 0.22211,
ROC-AUC is 0.67157, and F1 is 0.2822.

Test average precision is 0.21742, ROC-AUC is 0.66039, and F1 is 0.2753.
The model and threshold were fixed before Test evaluation. Fairness auditing,
Test and subgroup calibration, the auditing agent, PDF reporting, and monitoring
remain pending.

The Streamlit application currently contains a placeholder interface.

## Scope Guardrails

Version 1 targets one binary task: 30-day hospital readmission. Planned delivery
includes three fairness metrics (demographic parity, equalized odds, and PPV
differences) with confidence intervals, significance tests, suppression and severity
policies; a guarded function-calling agent; a 12-scenario evaluation harness; PDF
reporting; MLflow tracking; Docker and CI/CD; a public deployment; monitoring;
white-label report configuration; and a CLI batch mode. These are planned product
requirements, not claims that all features are currently implemented.

Outside Version 1: regression, multiclass and time-to-event tasks; image/text
models; causal explanations; automatic bias mitigation or retraining users' models;
real protected health information; EHR/FHIR integration; accounts, multi-tenant
storage, billing, and a third-party API. Intersectional groups, subgroup
calibration plots, and bilingual reporting are stretch work after required gates.
Clinical recommendations and legal compliance determinations are excluded.

See [Project Scope](docs/project-scope.md) and
[Implementation Decisions](docs/implementation-decisions.md) for details.


## Requirements

- Python 3.12
- Git

The commands below use Windows PowerShell and should be run from the project root.

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/AhmadAssi1/healthequity-agent.git
cd healthequity-agent
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

For application dependencies:

```powershell
python -m pip install -r requirements.txt
```

For development, including linting and testing:

```powershell
python -m pip install -r requirements-dev.txt
```

The development requirements also install the application dependencies.

## Run the Application

```powershell
python -m streamlit run app/app.py
```

Open the local URL shown in the terminal.

## Explore the Data

Open `notebooks/exploration.ipynb` in VS Code with the Python and Jupyter extensions installed. Select the project’s `.venv` as the notebook kernel.

The dataset is located at:

```text
data/diabetic_data.csv
```

Relative paths are resolved from the notebook kernel’s working directory. Confirm that the loading path matches that directory before running all cells.

The target is defined as:

- `1`: Readmission within fewer than 30 days (`<30`).
- `0`: No recorded readmission (`NO`) or readmission after more than 30 days (`>30`).

Each row represents a hospital stay. All stays for a patient are kept in one
partition to prevent patient overlap between Train, Validation, and Test.

Run notebooks/exploration.ipynb to generate the patient partitions, then
notebooks/model_development.ipynb to train, compare, evaluate, and save the
baseline. The latter exports data/samples/diabetes_audit_sample.csv and its
metadata JSON. The sample contains outcomes, predictions, scores, race, gender,
age bands, and payer codes; no patient or encounter identifiers are included.

The selected model is registered locally as healthequity-readmission-baseline,
Version 1. Model files and the MLflow database are excluded from Git and must be
regenerated on a fresh clone. Apply the saved preprocessing configuration before
using the model for prediction.

## Quality Checks

Run linting:

```powershell
python -m ruff check .
```

Run tests:

```powershell
python -m pytest --cov=app --cov-report=term-missing
```

Check dependency compatibility:

```powershell
python -m pip check
```

GitHub Actions runs linting and tests with a coverage summary on pushes and pull
requests. Coverage excludes the two live-API demonstration scripts.

With the project environment active, enable and run local commit checks:

```powershell
python -m pre_commit install
python -m pre_commit run --all-files
```

## Project Structure

- `app/`: Streamlit application.
- `data/`: Dataset files and identifier mappings.
- `docs/`: Project scope and data documentation.
- `notebooks/`: Exploratory analysis.
- `test/`: Automated tests.
- `.github/workflows/`: Continuous integration workflow.

## Documentation

- [Project Scope](docs/project-scope.md)
- [Data Card](docs/data_card.md)
- [Model Card](docs/model_card.md)
- [Implementation Decisions](docs/implementation-decisions.md)

## Data Source

Diabetes 130-US Hospitals for Years 1999–2008, from the UCI Machine Learning Repository.

## Secrets

Keep credentials in a local `.env` file. The file is excluded from Git. Never commit API keys or other secrets.
