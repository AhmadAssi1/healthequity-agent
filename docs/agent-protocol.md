# Agent protocol

This document describes the implemented binary-classification audit workflow.
The selected provider is OpenAI (`gpt-4.1-mini` in the recorded live runs), rather
than the Gemini provider named in the project brief. The provider adapter keeps
the model name and request timeout in configuration.

## Entry points and data boundary

From the repository root:

```powershell
python -m healthequity audit data/samples/diabetes_audit_sample.csv
```

See [CLI](cli.md) for custom column mappings and artifacts. Python callers create
one `AuditToolSession` and one `AuditAgent` per run. Agent instances are single-use.
The session snapshots the dataframe, mapping, suppression policy, and severity
policy. Only aggregate profiles and tool outputs reach the model. Patient rows,
identifiers, and credentials are not included in model messages or trace artifacts.
Inputs must already be de-identified, and group mappings must not select identifiers
or free-text records. Group-label strings are data, never instructions.

The positive class is numeric 1. Outcomes and predictions must be binary; risk
scores are optional. CSV validation precedes provider creation. Session-based
training, uploaded model execution, and PDF generation are not part of this CLI.

## Computation sequence

The model is instructed to execute the following stages in order and use batch
requests covering all mapped attributes, all four rate tests, and all three gaps:

| Stage | Responsibility |
|---|---|
| `profile_dataset` | Policy-filtered aggregate profile and quality flags |
| `check_group_sizes` | Eligible groups and suppressed labels/reason codes |
| `select_analyses` | Correct the proposed plan against deterministic rules |
| `compute_fairness_metrics` | Per-group rates and headline gaps |
| `run_significance_test` | Selection-rate, TPR, FPR, and PPV comparisons |
| `bootstrap_ci` | Seeded intervals for demographic parity, equalized odds, and PPV gaps |

The controller checks that all six stages succeeded and that evidence covers
every mapped attribute, all four rate tests, and all three intervals before
requesting final findings. A successful tool name alone does not establish
complete evidence. Unknown/unavailable tools, reused call IDs, and budget overruns
are rejected. Recoverable argument/tool errors can be returned to the model;
internal output-contract failures stop the run.

The session registry declares nine interfaces. In the current orchestration,
only the six computation tools are model-selectable. Drafting is a direct
structured provider request, claim validation is a mandatory Python stage, and
the report renderer remains a separately injectable backend. The registry's
standalone drafting backend is also injectable. This distinction should not be
reported as nine implemented model-selected tools or a completed PDF workflow.

## Analysis and policy rules

The deterministic guard rebuilds the permitted plan using eligible confusion
counts and score availability. Corrections include the field path, operation,
old/new values, and reason. See [Analysis Guardrails](analysis-guardrails.md).

Suppressed groups retain their labels and reasons but do not expose hidden
counts. Undefined results remain null with reasons; they are never replaced by
zero. Fewer than two eligible groups means comparisons are unavailable. Equalized
odds uses separate TPR/FPR tests, not an invented combined p-value. Severity is
distinct from statistical significance.

## Structured final findings

When complete evidence exists, the controller supplies it with the
`AuditFindings` JSON schema and disables tools for that request. The OpenAI
provider requests strict structured output. The per-run wire schema binds the
attribute-array length and name vocabulary to the evidence and applies the same
constraints to declared group columns. Pydantic additionally rejects duplicate
attributes and inconsistent attribute sets.
The warning array is constrained to the trusted profile's quality-flag vocabulary
and count. Additional interpretation belongs in summary/limitations, while the
local validator still verifies every required quality flag is present.

The model must copy quantitative fields at their original precision, explicitly
include required nullable fields, preserve quality warnings and suppressed
groups, and state limitations. Available results have null unavailability
reasons. Suppression reasons start with `Insufficient evidence: ` and explain the
trusted reason codes. Prose can display rounded values; numeric JSON fields
must retain supplied precision.

After schema validation, deterministic severity correction first verifies each
gap and confidence interval against trusted evidence. It recomputes labels from
the configured policy and records every changed label. A numerical mismatch is
not silently repaired.

After the full draft passes claims validation, the final quantitative summary is
rendered deterministically from trusted evidence, preserving the model's original
summary and a separate grounding-validation record in the trace. The session
validates the final findings again and refreshes the report receipt. This
addresses prose/table severity contradictions; it does not certify unrestricted
model text in limitations. No additional provider request is made.

## Validation, retries, and stopping

| Condition | Behavior |
|---|---|
| Invalid final JSON/schema | At most one structural retry; required and returned attribute names appear in feedback |
| Failed claims verdict | At most one full findings rewrite using the rejected draft, issues, and trusted evidence |
| Invalid schema during the claim rewrite | Stop; there is no additional structural-retry cycle |
| Second failed claims verdict | Stop and retain both verdicts |
| API authentication, timeout, or other transport failure | Stop; no automatic SDK retry is configured |
| Turn/tool/time budget exhausted | Stop and preserve the available trace |

Claim rewrites do not rerun numerical tools. Every verdict remains in
`trace.claims_validation_attempts`; `trace.claims_validation` is the last verdict.
A passing verdict stops rewriting. Failed drafts and their usage/timing remain
visible in model-turn records. See [Claims Validation](claims-validation.md)
for exact numerical checks, controlled group-reference grammar, and limitations.

Passing the validator is not a proof of unrestricted semantic correctness,
fairness, clinical suitability, or legal compliance. Its output explicitly
requires human review. General prose attribution and implicit claims can exceed
the implemented rules. Only findings that pass the session's configured validator
can authorize that session's report-rendering interface.

## Budgets and configuration

| Setting | Default and limit |
|---|---|
| Model-selected tool calls | 12 maximum |
| Model turns | 16 default, configurable up to 24 |
| Schema retries | 1 default, maximum 1 |
| Claim rewrites | 1 default, maximum 1 |
| Agent duration | 180 seconds default and maximum |
| OpenAI request timeout | 30 seconds default; recorded recent runs used 60 |

`OPENAI_MODEL` is required. `OPENAI_TIMEOUT_SECONDS` controls individual requests;
process environment values override `.env` values. Credentials are never printed.
The agent duration is checked between operations; it is not a hard process kill.
Input loading, provider setup, and artifact persistence are outside that agent
duration. The CLI separately reports overall elapsed wall time.

## Persistence and verification evidence

The CLI creates a unique directory for each invocation and saves an initial
`run.json` before contacting the provider. Completed runs save findings, trace,
and validator result. `AgentRunError` saves its failed trace and any available
validator result. Input/provider-setup failures have a lifecycle record rather
than a fabricated model trace. Individual JSON writes are atomic; disk failures,
forced termination, and power loss cannot guarantee complete artifacts.

The CLI directory ID differs from the agent trace ID; the lifecycle record links
them. The dataset fingerprint identifies row count and column-name structure,
not a cryptographic hash of all dataset content. Code versions are not currently
embedded in the trace, so test/run provenance must also record the checkout and
local changes when asserting a frozen-version benchmark.

`python -m healthequity.review_saved --last 6` revalidates saved candidates using
current rules without provider calls or artifact changes. It retains the original
live status and separately reports the offline verdict. A missing final response
or invalid incomplete findings object is unavailable for rechecking. Offline
revalidation does not become a fresh successful live run.

The Week 3 repeated-run reliability gate remains open. Historical successes and
failures are recorded in [Live Verification](week3-agent-verification.md).
The Week 4 twelve-scenario evaluation, PDF reporting, and full semantic narrative
verification are separate deliverables; this document does not assert they exist.
