# Command-line audits

From the repository root, activate the project environment and configure
`OPENAI_API_KEY` and `OPENAI_MODEL` in the local `.env` file. Existing process
environment variables take precedence. Never commit credentials.

```powershell
python -m healthequity audit data/samples/diabetes_audit_sample.csv
```

This is a live OpenAI audit, consumes API tokens, and uses the same guarded
agent, tool session, severity correction, and mandatory claims validator as
the Python workflow. A failed claims check can trigger one additional model
response to rewrite findings before revalidation; this also consumes tokens.
It does not render a PDF or train a model.

For custom CSV column names:

```powershell
python -m healthequity audit my_export.csv --outcome actual --prediction label --groups cohort region --score risk
```

Defaults: `outcome`, `prediction`, and any present columns named `race`, `gender`,
`age`, or `payer_code`, in that order. Unrecognized columns are never automatically
selected as group attributes. Supply `--groups` when using other names. A column
named `score` is detected automatically; `--score NAME` selects another column,
and `--no-score` disables score use.

The command accepts UTF-8 CSV files, including a BOM, with numeric binary labels
and an optional score between zero and one. The positive class is `1`. Predictions
must already exist. XLSX input and session-based training are not implemented in
this CLI. Existing loader behavior preserves literal `None` and `NA` categories;
`?` represents missing data. Duplicate headers, invalid labels, empty files, and
missing mapped columns are rejected before creating the provider.

## Artifacts and failure handling

Each invocation creates a unique directory below `outputs/agent_runs/`:

| File | Purpose |
|---|---|
| `run.json` | CLI lifecycle, phase, timing, mapping, schema fingerprint, and agent trace ID when available |
| `trace.json` | Complete agent trace for a completed audit or an `AgentRunError` |
| `findings.json` | Final findings, written only after the agent successfully completes validation |
| `claims_validation.json` | Automated verdict when validation was reached, including a failing verdict |

The CLI invocation directory has its own UUID. The agent's independent run ID is
recorded as `trace_id` in `run.json` and `run_id` in `trace.json`. Separate runs
never overwrite each other's files. Individual artifacts are written atomically;
the directory as a whole is not a transaction. `run.json` is marked completed
only after final artifacts have been saved. If storage fails, inspect the files
that were successfully written and rerun with a writable output directory.

Input/configuration failures save `run.json`; there is no fabricated agent trace
when the agent never started. Provider setup exceptions are sanitized rather than
printing potential credentials. Ctrl+C during provider setup or the agent run
records an interruption, but cannot provide a complete partial agent trace. Forced
process termination, power loss, or an unwritable disk cannot guarantee final
artifacts; the initial running record may remain.

Artifacts contain aggregate audit evidence and metadata, not patient rows or API
keys. Do not map identifiers or free-text fields as group columns. The default
output directory is ignored by Git. With a custom `--output-dir`, ensure it is
excluded from version control before committing artifacts.

Options:

```powershell
python -m healthequity audit --help
python -m healthequity audit data/samples/diabetes_audit_sample.csv --max-seconds 120 --output-dir outputs/agent_runs
```

The agent budget must be positive and at most 180 seconds. Existing agent checks
enforce the budget between operations; it is not a hard process interruption.
CSV loading, provider initialization, and artifact storage are outside that agent
budget. The CLI prints total elapsed wall time separately.

Exit codes: `0` completed, `1` agent/claim validation failed, `2` invalid input,
configuration, or command syntax, `3` artifact storage failed, `130` interrupted
during setup or audit. Malformed command syntax is handled by argparse before a
run directory exists.

A passing automated claims verdict still requires human review of narrative
interpretation. See [Claims Validation](claims-validation.md).

## Recheck saved outputs without API calls

```powershell
python -m healthequity.review_saved --last 6
```

This reads the most recent six CLI run directories, uses each last structured
model response and its saved aggregate evidence, reapplies severity policy, and
checks it with the current local validator. It does not load credentials, make
provider requests, alter saved artifacts, or turn a failed live run into a success.
The original outcome and the new offline verdict are printed separately. A run
that never reached a structured final candidate is explicitly unavailable.
Passing this review is not a replacement for fresh repeated live-run verification.

To preview deterministic summary grounding for previously validated drafts:

```powershell
python -m healthequity.review_saved --last 5 --ground-summary
python -m healthequity.review_saved --last 1 --ground-summary --show-summary
```

These modes remain read-only and offline. They validate the original draft first,
then render the quantitative summary from trusted evidence and validate the
result. Original findings, traces, and live statuses are unchanged.
