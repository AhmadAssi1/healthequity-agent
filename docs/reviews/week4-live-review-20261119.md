# Approved Week 4 live evaluation scorecard

**Live benchmark passed.** On November 19, 2026, Ahmad Assi approved the
AI-assisted evidence review for batch `24b01ac4df4c43f982bbd1fa1b54cf17`
using `gpt-4.1-mini`. The review found **zero unsupported claims across all
ten final reports** and confirmed both expected input refusals.

| Check | Result |
|---|---|
| Executed tool selection | 10/10 audits followed the expected analysis plan |
| Fixture-backed numbers | 176/176 checks passed |
| Schema, suppression and automated final claims checks | 10/10 audits passed each check |
| Invalid inputs | Both correctly refused before provider access |
| Final reports containing an unsupported claim in this AI-assisted review | 0/10 |
| Mean time per completed audit | 16.34 seconds, ranging from 14.56 to 19.23 seconds |
| Estimated cost of the evaluation calls | $0.1112052 total, or $0.01112 per completed audit |

## Evidence by scenario

Gap values in the table use percentage points (pp).

| Scenario | Evidence checked against the published findings |
|---|---|
| `no_gap` | All three observed gaps are zero. Each of the four rate tests returned p = 1.0. |
| `selection_gap` | The selection-rate gap is 20 pp and the positive predictive value (PPV) gap is zero. |
| `tpr_gap` | The equalized odds gap is 20 pp. The selection-rate gap is at the 10 pp policy boundary to displayed precision and carries the watch label. |
| `ppv_gap` | The PPV and equalized odds gaps are both 20 pp. The selection-rate gap is zero. |
| `tiny_group` | Only the demographic parity interval includes zero. The equalized odds and PPV intervals start at 1.61 and 0.21 pp. The two suppressed groups have distinct recorded reasons. |
| `multiple_groups` | The demographic parity interval starts at 1.00 pp despite a zero observed gap. The TPR, FPR and PPV chi-square tests each returned p = 0.00438 across all three groups. The final text correctly identifies the tests as joint comparisons. |
| `missing_score` | AUROC is unavailable for both eligible groups. The warning describes repeated values in the mapped columns. |
| `duplicated_rows` | All 800 input rows were retained. The three observed gaps are zero and each bootstrap interval has a positive lower bound. |
| `diabetes_female` | Two race groups qualified and four were suppressed. The largest observed gap is 2.96 pp. All four component p-values exceed 0.05. |
| `diabetes_male` | The demographic parity and equalized odds gaps carry substantial labels at 11.37 and 14.55 pp. The PPV gap is 1.81 pp, below the configured thresholds. |

The `missing_outcome` and `mislabelled_prediction` fixtures each produced the
expected input error. The scorecard records zero provider calls for each.

## Verification and sign-off

Replaying all 60 audit tool calls reproduced their saved outputs. The
review matched the final summaries, 55 limitations entries, four warnings,
ten suppression reasons, 30 gaps and their intervals, and 40 component tests
to the traces and documented methods. All ten audits delegated planning to the
deterministic planner.

The severity policy corrected four proposed labels before publication. In
`tpr_gap`, demographic parity changed from substantial to watch. In
`duplicated_rows`, all three labels changed from watch to below the policy
thresholds. The final values and labels matched the evidence in every case.
The saved summaries contain the copy edit identified during review. They say
"1 mapped attributes". The summary renderer now uses "1 mapped attribute"
when the audit has one mapped attribute.

MLflow experiment `agent-evals` records parent run
`17ea70bda6fa44fc8cd981c10a1abf82` and all 12 child runs as `FINISHED`.
All 22 archived scorecard, trace and findings files matched the reviewed files
byte for byte. The export's SHA-256 over the sorted file names and their bytes is
`c03e87e23b39733cd9c10d10b77c4e9e8068411b2fea1102d249912ec315cd82`.

The source scorecard's SHA-256 is
`794ea9413cdafb2b1fe98be2ff120912228d6c180bcc4d6247a06d3a55e0cb4a`.
It records the automated checks at run time. The approved decision is in
`scorecard.reviewed.json` and `scorecard.reviewed.md` in the batch directory
and in the parent run's `review/` artifacts.

The cost estimate uses the recorded 246,365 input tokens and 7,912 output
tokens across 80 evaluation calls, priced at the configured standard rates.
