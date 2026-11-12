# HealthEquity Agent

An evidence-based fairness auditor for binary classification on synthetic or
de-identified healthcare data. The working CLI combines a guarded OpenAI agent
with deterministic Python tools and saves validated findings and execution traces.

The demonstration use case is 30-day hospital readmission prediction on the UCI
Diabetes 130-US Hospitals dataset. PDF reporting, the full web application, and
drift monitoring are planned deliverables.

## Current Status

Updated November 12, 2026.

| Component | Status |
|---|---|
| Data validation | Typed column mappings; binary labels; optional scores; structural fingerprint; readable input errors |
| Demo model | LightGBM baseline, patient-level splits, MLflow artifacts and registered Version 1 |
| Fairness engine | Per-group metrics, three headline gaps, rate tests, seeded bootstrap intervals, suppression and severity policies |
| Guarded agent | OpenAI Responses adapter, corrected analysis plans, strict output schemas, execution budgets and trace records |
| Claims validation | Structured quantities checked against evidence; controlled English screening; one claim rewrite before a hard stop |
| Quantitative summary | Rendered from trusted tool evidence and policy; original model draft and grounding verdict retained in the trace |
| CSV CLI | Custom mappings, automatic artifact saving, failure records, and offline review of saved runs |
| Web interface | Streamlit placeholder |
| Planned product work | Twelve-scenario evaluation harness, PDF reports, upload UI, Docker/deployment, audit monitoring and white-label configuration |

The selected provider is **OpenAI**, using `gpt-4.1-mini` in the recorded live
runs; the original brief names Gemini. The provider/model configuration remains
separate from the audit engine. Python tools compute quantitative results, and
only aggregate profiles and tool outputs reach the language model.

The registry declares nine tool interfaces. The agent selects the six computation
tools; structured drafting and mandatory claim validation are controlled by
Python. Standalone drafting and PDF rendering retain injectable backend interfaces.

### Latest verification

| Evidence | Confirmed result |
|---|---|
| Full local test suite | **1,083 tests passed** |
| Linting | Ruff passed |
| Consecutive live demo batch | **5/5 completed**, each auditing 14,998 rows across race, gender, age and payer_code |
| Batch total latency | **44.354-66.574 seconds**, mean **51.363 seconds** |
| Model activity | Eight model turns and six computation-tool calls per run; first claims verdict passed in all five |
| Follow-up summary grounding | All five saved candidates passed local grounding/revalidation with no additional provider calls |

The live batch preceded the final deterministic summary-grounding stage. Its
offline preview is separate evidence, not five new live runs. Earlier failed
batches are retained in the [verification record](docs/week3-agent-verification.md).
These measurements cover one dataset/model configuration; they do not establish
general reliability or replace the planned twelve-scenario evaluation.

The rule-based validator does not certify unrestricted prose. The generated
quantitative summary keeps policy labels consistent with the evidence; free-text
limitations and further interpretations still require human review. Results are
statistical associations, not clinical recommendations or legal determinations.
See [Claims Validation](docs/claims-validation.md) and
[Agent Protocol](docs/agent-protocol.md).

### Demonstration baseline

LightGBM with 15 leaves, operating threshold **0.145**. Model and threshold
selection used Validation; Test evaluated the fixed model.

| Metric | Validation | Test |
|---|---:|---:|
| Average precision | 0.22211 | 0.21742 |
| ROC-AUC | 0.67157 | 0.66039 |
| F1 at the selected threshold | 0.2822 | 0.2753 |

The [Model Card](docs/model_card.md) records preprocessing, partitions, artifacts,
and interpretation limits. Test and subgroup calibration remain unassessed.

## Scope Guardrails

Version 1 targets one binary task: 30-day hospital readmission. The complete scope
includes three fairness metrics (demographic parity, equalized odds, and PPV
differences) with confidence intervals, significance tests, suppression and severity
policies; a guarded function-calling agent; a 12-scenario evaluation harness; PDF
reporting; MLflow tracking; Docker and CI/CD; a public deployment; monitoring;
white-label report configuration; and a CLI batch mode. The status table above
distinguishes working components from remaining product requirements.

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

### 4. Configure OpenAI

Create `.env` in the repository root with your own key:

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_SECONDS=60
```

The timeout shown is per provider request. The agent has a separate total budget
of 180 seconds by default, checked between operations. Existing process environment
variables take precedence over `.env`. If a stale session key overrides the file:

```powershell
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
```

Keep `.env` local and excluded from Git.

## Run a CSV Audit

The bundled demo CSV contains ready-made predictions:

```powershell
python -m healthequity audit data/samples/diabetes_audit_sample.csv
```

This command makes live OpenAI requests and consumes API tokens. Default mappings
are `outcome`, `prediction`, optional `score`, and the present columns from
`race`, `gender`, `age`, and `payer_code`. For another de-identified CSV:

```powershell
python -m healthequity audit my_export.csv --outcome actual --prediction label --groups cohort region --score risk
```

Labels must be numeric 0/1, with 1 as the positive class. Use `--no-score` to ignore
a score column. The current CLI requires supplied predictions and accepts CSV.
XLSX and session-based training are planned upload-flow work.

Each invocation creates a unique directory under `outputs/agent_runs/`:

| Artifact | Contents |
|---|---|
| `run.json` | Lifecycle, timing, mapping, structural fingerprint, trace ID and any failure |
| `trace.json` | Model/tool history, usage, timing, policy corrections, claim verdicts and summary grounding when reached |
| `findings.json` | Final findings after successful validation |
| `claims_validation.json` | Verdict when claims validation was reached |

The CLI invocation directory ID differs from the agent trace ID; `run.json` links
them. Input/setup failures save a lifecycle record, while agent failures retain
the available failed trace. Output JSON files are written atomically. The default
artifact directory is excluded from Git. See [CLI](docs/cli.md) for exit codes,
storage/interruption limits, and custom output directories.

### Review saved outputs locally

```powershell
python -m healthequity.review_saved --last 5
python -m healthequity.review_saved --last 5 --ground-summary
python -m healthequity.review_saved --last 1 --ground-summary --show-summary
```

Review uses saved aggregate evidence with the current rules, makes no provider
requests, and leaves original artifacts unchanged. Grounding mode previews the
trusted-fact summary after draft validation. Original live status and offline
verdict are reported separately. Runs with no final response or invalid incomplete
findings are unavailable for rechecking.

## Streamlit Prototype

```powershell
python -m streamlit run app/app.py
```

Open the local URL shown in the terminal. The interface currently displays a
placeholder; use the CLI for the implemented audit workflow.

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

Run all offline tests:

```powershell
python -m pytest -q
```

Measure coverage for the application, fairness engine, and CLI:

```powershell
python -m pytest --cov=app --cov=fairness --cov=healthequity --cov-report=term-missing
```

Check dependency compatibility:

```powershell
python -m pip check
```

The latest confirmed full local run passed **1,083 tests**. Provider/orchestration
tests use scripted responses or injected clients; the standard test suite makes
no live OpenAI calls. Live CLI benchmarks are separate paid runs.

The test count does not provide a new coverage measurement. Coverage excludes
the two live-API demonstration scripts and does not measure notebook training.
GitHub Actions and pre-commit run Ruff and tests with coverage for `app`,
`fairness`, and `healthequity`. Confirm CI for this revision after pushing.

With the project environment active, enable and run local commit checks:

```powershell
python -m pre_commit install
python -m pre_commit run --all-files
```

## Project Structure

- `app/`: Agent/provider adapters, typed contracts, profiling, plan/policy guards,
  claims validation, summary grounding, data helpers, and the Streamlit placeholder.
- `healthequity/`: CSV audit CLI, artifact persistence, and offline saved-run review.
- `fairness/`: Deterministic metrics, statistics, policies, and scenario generation.
- `config.yaml`: Group eligibility and severity thresholds.
- `data/`: Source dataset, identifier mappings, and the demo audit sample.
- `docs/`: Scope, data/model cards, implementation decisions, and methodology.
- `notebooks/`: Exploration, baseline development, and fairness verification.
- `test/`: Automated tests.
- `outputs/agent_runs/`: Local audit artifacts, excluded from Git.
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
- [CLI](docs/cli.md)
- [Agent Protocol](docs/agent-protocol.md)
- [Live Verification Record](docs/week3-agent-verification.md)
- [Implementation Decisions - baseline review](docs/implementation-decisions.md)

## Data Source

Diabetes 130-US Hospitals for Years 1999–2008, from the UCI Machine Learning Repository.

## Secrets

Keep credentials in a local `.env` file. The file is excluded from Git. Never commit API keys or other secrets.
