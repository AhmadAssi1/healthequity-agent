# HealthEquity Agent

An AI agent for analyzing machine learning model performance and fairness, generating evidence-based reports, and monitoring drift.

The initial version focuses on binary classification, using 30-day hospital readmission prediction as a demonstration use case.

## Current Status

The project is under development. Completed work includes:

- Repository setup and automated linting and testing.
- Dataset exploration, binary target creation, and a Data Card.
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
- Deterministic per-group metrics, reference comparisons, and maximum pairwise
  demographic parity, equalized odds, and PPV differences.
- Significance-test selection and reproducible bootstrap 95% confidence intervals.
- Configurable group suppression, recorded exclusion reasons, and severity bands.
- Synthetic scenarios with configurable planted selection-rate, TPR, and PPV gaps.
- Methodology documentation and automated tests for the fairness engine.
- Structured audit schemas, internal dataset profiles, and policy-filtered
  profiles for the agent.
- Deterministic analysis planning from eligible-group confusion counts.
- A plan guard that rebuilds the approved plan from trusted counts and records
  every addition, replacement, and removal from the proposed plan.
- Strict input/output contracts and a provider-neutral registry for the nine
  tools. Six computation adapters and a local claims validator work on a bound
  dataset; drafting and PDF rendering retain injectable backend interfaces.
- An OpenAI Responses provider and guarded agent with strict final JSON output,
  execution budgets, recorded tool history, and deterministic severity correction.
- Mandatory structured-evidence validation and English narrative screening,
  with verdicts retained in successful and failed traces. Human review remains
  required; see [Claims Validation](docs/claims-validation.md).

The selected experimental baseline uses 15 leaves and a decision
threshold of 0.145. Validation average precision is 0.22211,
ROC-AUC is 0.67157, and F1 is 0.2822.

Test average precision is 0.21742, ROC-AUC is 0.66039, and F1 is 0.2753.
The model and threshold were fixed before Test evaluation. The Week 1 and Week 2
technical benchmark checks passed in the November 5, 2026 review: 180 tests,
hand-calculation and Fairlearn reference checks, suppression checks, and seeded
bootstrap reproducibility. Reloaded baseline and registered-model predictions
matched the logged metrics and demo CSV.

The fairness engine can be used from Python without an LLM. Its functions have
also been checked together on the demo attributes. The guarded agent has completed
a live run on the 14,998-row demo dataset. A CLI entry point, PDF reporting, full
UI, deployment, and monitoring remain pending. Test and subgroup calibration have
not been assessed.

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

## Deterministic Fairness Engine

The `fairness/` package contains:

- `metrics.py`: per-group count, base rate, selection rate, TPR, FPR, PPV,
  optional AUROC, reference comparisons, and maximum pairwise gaps.
- `stats.py`: pooled two-proportion z-tests, multi-group chi-square tests,
  Fisher tests for sparse tables, and percentile bootstrap intervals
  (1,000 resamples and seed 42 by default).
- `policy.py`: eligibility checks, filtering with suppression reasons, and
  severity classification using `config.yaml`.
- `scenarios.py`: exact confusion-count fixtures and seeded scenarios with
  planted selection-rate, TPR, or PPV gaps from 0 to 0.50 in steps of 0.01.

Before computing gaps, tests, or bootstrap intervals, call
`filter_reportable_groups` separately for each attribute on the original dataset.
By default, a group needs at least 50 stays and 10 actual positive outcomes.
Retain the eligibility table so excluded groups and their reasons remain visible.
Missing values form a separate group, distinct from a literal category named
`Missing`. If fewer than two groups remain, report the comparison as unavailable
instead of calling the analysis functions on empty or single-group data.

Undefined rates remain unavailable, not zero. The default severity policy labels
an observed gap above 0.10 with a confidence-interval lower bound above zero as
`substantial`; otherwise a gap above 0.05 is `watch`. Severity is a reporting
convention, separate from statistical significance. The rate-test helpers do not
produce a combined equalized-odds p-value. The bootstrap assumes independent
stays; repeated stays from one patient are a documented limitation.

See [Methodology](docs/methodology.md) for formulas, test-selection rules, and
interpretation limits. The scenario generator is groundwork for the later
12-scenario agent evaluation harness, which is not implemented yet.

## Quality Checks

Run linting:

```powershell
python -m ruff check .
```

Run all tests and measure coverage for both application and fairness modules:

```powershell
python -m pytest --cov=app --cov=fairness --cov-report=term-missing
```

Check dependency compatibility:

```powershell
python -m pip check
```

The November 5, 2026 review and follow-up passed 180 tests and measured 98% combined line
coverage for `app` and `fairness` (100% for the included `app` modules). Coverage
excludes the two live-API demonstration scripts and does not measure notebook
training or complete agent behavior.

GitHub Actions, pre-commit, and `.coveragerc` are configured to cover both `app`
and `fairness`. The updated coverage settings have been verified locally; a new
GitHub CI run is required after these changes are committed and pushed.

With the project environment active, enable and run local commit checks:

```powershell
python -m pre_commit install
python -m pre_commit run --all-files
```

## Project Structure

- `app/`: Streamlit placeholder, data helpers, schemas, validation, profiling,
  analysis selection, and plan correction.
- `fairness/`: Deterministic metrics, statistics, policies, and scenario generation.
- `config.yaml`: Group eligibility and severity thresholds.
- `data/`: Source dataset, identifier mappings, and the demo audit sample.
- `docs/`: Scope, data/model cards, implementation decisions, and methodology.
- `notebooks/`: Exploration, baseline development, and fairness verification.
- `test/`: Automated tests.
- `.github/workflows/`: Continuous integration workflow.

## Documentation

- [Project Scope](docs/project-scope.md)
- [Data Card](docs/data_card.md)
- [Model Card](docs/model_card.md)
- [Data Contract](docs/data-contract.md)
- [Methodology](docs/methodology.md)
- [Analysis Guardrails](docs/analysis-guardrails.md)
- [Tool Interfaces](docs/tool-interfaces.md)
- [Claims Validation](docs/claims-validation.md)
- [Implementation Decisions](docs/implementation-decisions.md)

## Data Source

Diabetes 130-US Hospitals for Years 1999–2008, from the UCI Machine Learning Repository.

## Secrets

Keep credentials in a local `.env` file. The file is excluded from Git. Never commit API keys or other secrets.
