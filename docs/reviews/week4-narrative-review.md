# Week 4 Narrative Review

Review date: November 17, 2026.
Source evaluation batch: `5bb6b88b848640f7b4e106a61d820389`.

All ten reports were reviewed for consistency between summaries, limitations,
warnings, group eligibility, metrics, confidence intervals, and rate-test results.
This is an AI-assisted editorial and evidence review. No human approval is
attributed to Ahmad. Original findings, traces, and scorecards remain unchanged.

## Review results

| Scenario | Assessment | Main finding |
|---|---|---|
| no_gap | Clarification | No formal power or equivalence analysis was performed; results describe a synthetic fixture. |
| selection_gap | Clarification | AUROC describes artificially constructed scores, not clinical model performance. |
| tpr_gap | Clarification | Passing sample-size thresholds does not establish all statistical assumptions or overall reliability. |
| ppv_gap | Clarification | Limitations are broadly consistent; explicitly identify the deliberately planted synthetic gaps. |
| tiny_group | Correction required | Not all confidence intervals include zero. |
| multiple_groups | Correction required | The DPD interval excludes zero; omnibus tests do not establish significant pairwise differences involving C. |
| missing_score | Clarification | Repeated mapped-column combinations are expected here and do not establish duplicate underlying observations. |
| duplicated_rows | Clarification | Complete scored rows are deliberately duplicated; uncertainty estimation is not cluster-aware. |
| diabetes_female | Clarification | Nonsignificant rate tests do not establish fairness; distinguish those tests from gap intervals. |
| diabetes_male | Clarification | The audit uses outcomes, predictions, scores, and group labels; component tests do not all have the same significance result. |

## Required corrections

### tiny_group

The original limitation states that the confidence intervals include zero.
However, the EOD interval has a lower bound of `0.01610770339643733`, and
the PPV interval has a lower bound of `0.0020707819418089753`.
Only the demographic-parity interval includes zero. Suggested replacement:

> The demographic-parity bootstrap interval includes zero; the equalized-odds
> and PPV intervals have positive lower bounds. They must not be described
> collectively as including zero or substituted for the component rate tests.

### multiple_groups

The original text states that the demographic-parity interval includes zero,
but its lower bound is `0.010000000000000009`. That statement is incorrect.
It also attributes significant differences to group C, although the chi-square
tests compare all three groups jointly. No post-hoc pairwise tests were run.
Suggested replacement:

> The TPR, FPR and PPV chi-square tests are omnibus comparisons across the
> three groups. They do not establish which pair differs significantly;
> no post-hoc pairwise tests were performed.

Some nonnegative-gap percentile bootstrap intervals have positive lower bounds
even when the observed gap is zero. This requires careful methodological
interpretation. This review does not independently establish interval coverage,
change the resampling algorithm, or recompute the audit. The intervals must not
be treated automatically as substitutes for the recorded component rate tests.

## Proposed revisions and approval status

[Proposed report revisions](week4-narratives-20261117.json) contains the findings,
a lightly edited summary, and a complete replacement limitations list for each
report. Summary edits correct singular grammar and the wording used when no
groups were suppressed; numerical values remain unchanged. Each source report
has a recorded SHA-256 hash identifying the exact version reviewed.

Warnings match the recorded profiler outputs. However, repeated combinations
in mapped columns alone do not establish duplicate source encounters. The
proposed limitations clarify this distinction. The duplicated_rows fixture is
an exception because full-row duplication is known by construction.

The original reports cannot be certified as having zero unsupported narrative
claims after the two interpretation errors identified above. Automated batch
results remain recorded, while human narrative approval remains pending.
Proposed revisions do not establish clinical suitability, legal compliance, or
approval of future reports. No additional paid API requests were made for this
review.
