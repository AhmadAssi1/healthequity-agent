# HealthEquity Agent

An AI agent for analyzing machine learning model performance and fairness, generating evidence-based reports, and monitoring drift.

The initial version focuses on binary classification, using 30-day hospital readmission prediction as a demonstration use case.

## Current Status

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

The selected experimental baseline uses 15 leaves and a decision
threshold of 0.145. Validation average precision is 0.22211,
ROC-AUC is 0.67157, and F1 is 0.2822.

Test remains reserved for final evaluation. Fairness auditing,
probability calibration assessment, the auditing agent, PDF reporting,
and monitoring remain pending.

The Streamlit application currently contains a placeholder interface.


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

Each row represents a hospital stay. A patient may have multiple stays, so future training and evaluation splits must separate patients to prevent data leakage.

## Quality Checks

Run linting:

```powershell
python -m ruff check .
```

Run tests:

```powershell
python -m pytest
```

Check dependency compatibility:

```powershell
python -m pip check
```

GitHub Actions runs linting and tests on pushes and pull requests.

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

## Data Source

Diabetes 130-US Hospitals for Years 1999–2008, from the UCI Machine Learning Repository.

## Secrets

Keep credentials in a local `.env` file. The file is excluded from Git. Never commit API keys or other secrets.