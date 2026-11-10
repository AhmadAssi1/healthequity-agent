# Analysis selection and plan correction

`app/analysis_selection.py` constructs a deterministic plan from validated,
policy-filtered confusion counts. `app/plan_guard.py` compares an agent proposal
with that plan. Neither module executes statistical tests, bootstraps, or LLM calls.

## Trusted inputs

Use `compute_analysis_counts` with the audit's selected suppression policy.
Pass score availability from the mapped dataset, not from an agent response.
The guard does not reapply suppression to arbitrary supplied aggregate objects;
the caller must use counts from the validated Python pipeline. Use one policy
snapshot consistently when building counts and agent-visible profiles.

## Decision rules

- Fewer than two eligible groups: fairness comparisons and rate tests unavailable.
- An undefined rate in any eligible group: that rate test unavailable; do not
  silently omit that group. EOD needs defined TPR and FPR in every eligible group;
  PPV difference needs defined PPV in every eligible group.
- A table containing only successes or only failures: test unavailable.
- Minimum expected cell count below 5: Fisher exact for two groups, Fisher with
  Monte Carlo estimation for more groups. Otherwise select a pooled two-proportion
  z-test for two groups or Pearson chi-square for more groups. These rules match
  `fairness.stats.compare_proportions`.
- Bootstrap is planned for available headline metrics. This does not guarantee
  every resample or the final interval is available.
- AUROC requires scores and at least one eligible group containing both outcome
  classes. Other groups' AUROCs can remain unavailable.

## Guard contract

`guard_analysis_plan(proposed_plan, counts, score_available)` accepts an
`AnalysisPlan` or a raw JSON-object dictionary. Raw proposals may have missing,
extra, or inconsistent fields. The returned `GuardedAnalysisPlan.plan` is always
rebuilt from trusted inputs, rather than patched using proposal claims.

Each `PlanCorrection` records an operation (`add`, `replace`, or `remove`), a
bracket-notation field path rooted at `$`, old and new JSON values, and a reason.
The operation distinguishes an absent field from an explicit JSON null. Paths
are descriptive and must never be evaluated as Python. Missing/extra list items
and object fields are recorded; replacing an incorrectly typed subtree records
the whole subtree. Only an exactly canonical proposal produces no corrections.
Ordering changes and replacements of explanations are recorded as well.

The guard takes independent snapshots and does not mutate the proposal or the
counts. Non-object proposals raise `TypeError`; non-JSON values (including tuples,
non-string dictionary keys and non-finite numbers) raise validation errors.
Invalid trusted inputs propagate errors instead of producing an executable plan.

For an already loaded dataframe `df`, a validated column mapping `mapping`, an
audit policy snapshot `policy`, and a JSON-object proposal `proposed_plan`:

```python
from app.analysis_counts import compute_analysis_counts
from app.plan_guard import guard_analysis_plan

counts = compute_analysis_counts(df, mapping, policy)
guarded = guard_analysis_plan(
    proposed_plan=proposed_plan,
    counts=counts,
    score_available=mapping.score is not None,
)

approved_plan = guarded.plan
corrections = guarded.corrections
trace_payload = guarded.model_dump_json(indent=2)
```

Only `approved_plan` should reach the future analysis executor. Do not execute
the original proposal before checking it.

## Trace integration

`GuardedAnalysisPlan.model_dump_json()` provides the validated plan and correction
records for the later run trace. This component does not persist traces to disk,
call a provider, implement a rewrite/retry loop, or validate narrative claims.
Those remain orchestration work. No conclusion about complete agent reliability
follows from these deterministic checks alone.

## Final severity policy

`app/findings_policy.py` applies the severity policy from the trusted
`AuditEvidence` snapshot after structural findings validation. It verifies the
headline gap, interval bounds and resample counts against tool evidence before
classifying severity. Numerical drift fails instead of silently repairing
numbers (floating-point tolerance: relative 1e-9, absolute 1e-12).

Severity-label changes are stored in `AgentRunTrace.severity_corrections`, with
attribute, metric, original label, corrected label and reason. The model's
original response is preserved in its turn record. This check does not approve
the other numerical fields or narrative; `claims_validation` remains unset.
