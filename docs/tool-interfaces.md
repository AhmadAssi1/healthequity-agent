# Nine audit tool interfaces

`app/tool_contracts.py` defines strict Pydantic input/output models.
`app/tools.py` binds the tools to an `AuditToolSession`, with one in-memory
dataframe, column mapping, suppression policy and severity policy snapshot.
Dataframes, file paths, arbitrary counts, and patient records are not tool inputs.

## Interface status

| Tool | Input | Output | Implementation |
|---|---|---|---|
| profile_dataset | EmptyToolInput | AgentDatasetProfile | Working Python adapter |
| check_group_sizes | EmptyToolInput | GroupSizesOutput | Working Python adapter; suppressed counts omitted |
| select_analyses | SelectAnalysesInput | GuardedAnalysisPlan | Working Python rules and correction log |
| compute_fairness_metrics | AttributeToolInput | FairnessMetricsOutput | Working Fairlearn adapter |
| run_significance_test | SignificanceTestInput | SignificanceOutput | Working statistical-engine adapter |
| bootstrap_ci | BootstrapCIInput | BootstrapOutput | Working seeded bootstrap adapter |
| draft_findings | EmptyToolInput | DraftFindingsOutput | Injectable drafting backend; production implementation pending |
| validate_claims | ValidateClaimsInput | ClaimsValidationOutput | Default local evidence checks and English narrative screening; human review required |
| render_report | EmptyToolInput | RenderReportOutput | Injectable renderer; production PDF rendering pending |

Drafting and rendering fail explicitly with `ToolUnavailableError` when
their backend is absent. Claims validation uses the local validator by default;
an explicitly disabled validator also fails with `ToolUnavailableError`.
No fake findings, automatic passing verdict, or dummy
PDF path is returned by production code. Tests inject doubles to exercise their
contracts and lifecycle; those tests do not validate live LLM or PDF behavior.

## Usage

```python
from app.data import load_data
from app.schemas import ColumnMapping
from app.tools import AuditToolSession

df = load_data("data/samples/diabetes_audit_sample.csv")
mapping = ColumnMapping(
    outcome="outcome", prediction="prediction", score="score",
    groups=["race", "gender", "age", "payer_code"],
)
session = AuditToolSession(df, mapping)

profile = session.invoke("profile_dataset", {})
eligibility = session.invoke("check_group_sizes", {})
plan = session.invoke("select_analyses", {})
metrics = session.invoke("compute_fairness_metrics", {})
tests = session.invoke("run_significance_test", {})
intervals = session.invoke("bootstrap_ci", {"n_resamples": 1000, "seed": 42})
definitions = session.definitions()
```

Omitted attributes/rates/metrics select all, allowing the six deterministic
stages to complete in six calls rather than one call per group/rate. Subset
requests use `attributes`, `rates`, and `metrics`. An explicit `test_name`
requires exactly one attribute and rate and must match the deterministic plan.
The session rejects unmapped attributes and recomputes permitted analyses from
trusted counts. It requires mapped predictions for analysis tools; profiling
can precede prediction generation.

## Provider boundary

`definitions()` exports all nine names and provider-neutral input/output JSON
Schemas, plus backend availability. The availability flag reports implementation
configuration, not fulfillment of every call's data/state prerequisites. A
provider adapter must translate these schemas, including defaults, nulls and
references, to its API format. The selected OpenAI provider implements this
translation and strict final-output schemas. No network or LLM call occurs in
`AuditToolSession` itself.

All results are validated again after serialization. Undefined numerical
results are JSON null with a reason; NaN and infinity are not sent. Suppressed
groups expose labels and reason codes only. Integer group codes preserve typed
category identities through pandas/Fairlearn (for example `True`, `1`, and `1.0`
remain different groups). Overall dataset totals remain visible as specified
in the brief; suppression is not a differential-privacy guarantee.

## Backend and report lifecycle

`AuditAgent` invokes claim validation automatically after final findings and
severity correction. A failed verdict stops completion and is retained in the
trace. This deterministic stage does not add a model turn or consume the model's
tool-call budget. See [Claims Validation](claims-validation.md) for the exact
check coverage and narrative limitations.

`ToolBackends` accepts trusted callables for drafting, claim validation and
rendering. Drafting and claim validation receive an independent `AuditEvidence`
snapshot containing the profile, canonical plan, severity policy, per-group
metrics, all rate tests and all bootstrap gaps. Partial computations cannot
satisfy this evidence requirement.

Rendering consumes only the findings that passed the configured claim validator
in the same session. No `validated=true` JSON argument can authorize it. A failed
claim check, a new draft or recomputation of analysis evidence clears prior
approval. Backend outputs are revalidated, including previously constructed but
subsequently mutated Pydantic objects. A renderer must produce the actual PDF
and valid run/trace identifiers; this interface does not create files itself.

## Verification on November 9, 2026

The six deterministic tools ran on the 14,998-row demo sample with default
suppression, seed 42 and 1,000 bootstrap resamples. Results included four
attribute summaries, 16 rate-test results and 12 bootstrap results. Eligible /
suppressed group counts were race 6/0, gender 2/1, age 8/2, payer_code 10/7.
All outputs serialized with non-finite JSON values forbidden. Local execution,
including session setup, took approximately 1.4 seconds; this is not a complete
agent benchmark and excludes provider calls, narrative validation and rendering.

An integration test exposed SciPy returning a size-one p-value array for a sparse
six-group Fisher Monte Carlo table. The engine now extracts the scalar with
`np.asarray(p_value).item()`; a regression test covers that case.
