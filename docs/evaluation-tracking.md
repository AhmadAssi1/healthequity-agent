# Evaluation automation and MLflow tracking

## Local commands

From the repository root with the project environment active:

```powershell
python -m evals --mlflow
python -m evals.probe
python -m evals --live --mlflow
```

The first command is offline. The other two call the configured provider.
MLflow tracking uses `MLFLOW_TRACKING_URI` when set, otherwise
`sqlite:///mlflow.db`. Override it explicitly with `--tracking-uri`.

Existing results can be logged without any new provider request:

```powershell
python -m evals.tracking outputs/evaluations/5bb6b88b848640f7b4e106a61d820389
```

Each export creates one parent suite run and one child per scheduled scenario
in the `agent-evals` experiment. Suite and case metrics, latency, known token
usage/cost, model/configuration metadata, scorecards, findings, and existing
traces are recorded. Unknown values remain absent from MLflow metrics rather
than becoming misleading zeros. Failed and unrun cases remain failures.
`FINISHED` means the applicable automatic checks passed; it does not grant
human narrative approval or satisfy the complete benchmark gate.

Only named scorecards, findings and trace files are uploaded. Source CSVs,
environment files, the tracking database and arbitrary files in the output
directory are not uploaded by the exporter. Scenario names are validated before
constructing artifact paths. Existing policy-filtered trace boundaries remain
the responsibility of the audit tools; the exporter does not loosen suppression.

An atomic `mlflow-export.json` receipt identifies the source scorecard and all
exported artifacts by SHA-256 and identifies the tracking URI by its hash.
Exporting unchanged outputs to the same store again reuses the receipt. Changed
outputs or a different tracking URI require a new export. Keep the receipt with
its tracking store; if that store is deleted, an old receipt no longer proves
that the stored run is available. Interrupted exports can leave failed partial
runs; retrying does not relabel those runs as successful.

Tracking failures return a nonzero CLI exit code while retaining local
evaluation evidence. A suite interrupted before final export can be exported
later with `python -m evals.tracking DIRECTORY`.

To inspect the local store:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` and select `agent-evals`.

## GitHub Actions

`.github/workflows/eval.yml` defines two jobs:

| Event | Deterministic job | Live job |
|---|---|---|
| Pull request | Evaluation tests and complete offline suite with MLflow | Does not run; no API secret is provided |
| Push to main | Same deterministic checks | Runs after deterministic success: configuration check, schema probe, full live suite with MLflow |

The existing general CI workflow remains in place. Evaluation jobs print their
Markdown scorecard into the job log and GitHub job summary. Scorecards, aggregate
traces/findings, the evaluation SQLite database and MLflow artifacts are retained
as workflow artifacts for fourteen days, including when a job fails. Each job
has a separate temporary local tracking store; no permanent hosted MLflow server
is provisioned by this workflow. Standalone scorecards and traces are readable
after download; MLflow artifact URIs inside a downloaded database may refer to
the original runner filesystem.

Repository configuration required before the first main-branch live run:

1. Add an Actions repository secret named `OPENAI_API_KEY`.
2. Optionally add an Actions repository variable named `OPENAI_MODEL`;
   the configured fallback is `gpt-4.1-mini`.
3. Commit and push the reviewed source and workflow changes.
4. Verify both the existing CI and the new evaluation workflow on GitHub.

If the secret is missing, the live configuration check fails explicitly. A
rejected schema probe prevents the full paid batch from starting. Live runs on
main consume provider tokens; deterministic pull-request checks do not call the
LLM provider. Workflow concurrency serializes runs for a branch without
cancelling in-progress paid evaluation batches.

## Verification: November 17, 2026

- Full local suite: 1,162 passed in 19.67 seconds. Ruff clean. One upstream
  SQLAlchemy deprecation warning arose during the real SQLite integration test.
- MLflow integration tests cover parent/child relationships, artifact retrieval,
  failed statuses, unknown metrics, artifact allowlisting, receipt reuse,
  changed-trace identity, export failure, and traversal rejection.
- The successful saved live batch was exported without new API calls:
  parent run `a6fc69f037644ba3bbdba00b81301f5b`, twelve FINISHED child runs.
  Its automatic numeric and executable tool-selection accuracy remain 1.0;
  `live_benchmark_passed` remains 0 and human review remains pending.
- A fresh complete offline CLI suite also passed and logged successfully:
  batch `b568c90249424727ae80f93e0ba0701e`, parent run
  `b84dd86039844800b41a29e51e440ec3`.
- GitHub-hosted execution is not yet verified. Local workflow creation does not
  establish that the repository secret is configured or the workflow is green.
