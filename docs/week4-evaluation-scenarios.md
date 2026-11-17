# Week 4: known-answer evaluation scenarios

The first step supplies twelve reproducible fixtures and expected answers, not
an agent scorecard. Load with `from evals.scenarios import build_scenarios`.
Each case includes a dataframe, column mapping, and expected behavior. No API
requests, model training, MLflow writes, or changes to historical runs occur.

## Scenario catalogue

| ID | Rows | Expected behavior |
|---|---:|---|
| no_gap | 400 | All three headline gaps are zero. |
| selection_gap | 400 | DPD 0.20, EOD 0.20, PPV gap 0. |
| tpr_gap | 400 | DPD 0.10, EOD 0.20, PPV gap 1/12. |
| ppv_gap | 400 | DPD 0, EOD 0.20, PPV gap 0.20. |
| tiny_group | 480 | Suppress Tiny (20 rows, 12 positives) and LowPositive (60 rows, 5 positives); compare A/B only. |
| multiple_groups | 600 | Three eligible groups; chi-square for all four rate tests. |
| missing_score | 400 | Audit labels; exclude AUROC and other score-dependent analyses. |
| missing_outcome | 400 | Reject missing mapped outcome before any provider call. |
| mislabelled_prediction | 400 | Reject YES/NO text predictions before any provider call. |
| duplicated_rows | 800 | Warn; retain 400 repeated rows; group counts double, rates do not change. |
| diabetes_female | 1,000 | Audit race on a fixed Female slice of the existing baseline export. |
| diabetes_male | 1,000 | Audit race on a fixed Male slice of the existing baseline export. |

Planted gaps are not isolated interventions: changing one confusion matrix can
change several fairness metrics. All three expected gaps are provided rather
than falsely assuming the non-target metrics stay unchanged. Zero point gaps
are not proof of fairness; bootstrap uncertainty and narrative interpretation
will be assessed by the later harness.

## Ground truth and policy

Synthetic inputs use declared TN/FP/FN/TP counts and the existing seeded row
generator. An independent elementary-arithmetic oracle derives rates and gaps
from those declarations, never from agent output or production fairness/planner
functions. It fixes suppression to n >= 50 and positive outcomes >= 10, and
records expected statistical test methods per rate. Tests compare these
expectations against actual fixture counts, the fairness engine, and planner.

The real fixtures were originally selected by filtering the exported Test audit
sample by gender and sampling 1,000 rows with pandas `random_state=42`.
Those exact selections and their order are now pinned as zero-based source-row
indices; runtime loading does not repeat version-dependent sampling.
They are disjoint encounter slices;
this does not assert that patient identities are disjoint. Audit race only.
Model scores and predictions are reused unchanged; no tuning or retraining.
These are engineering evaluation examples, not new independent clinical test
sets, and the small slices do not replace the full-data audit.

`evals/diabetes_slice_answers.json` pins source and slice SHA-256 hashes,
model provenance, and independently counted confusion cells. Source bytes
normalize CRLF to LF. Slice identity hashes compact UTF-8 JSON containing the
original CSV header and selected string-valued source rows, before float parsing.
It never hashes a pandas CSV reserialization. The original pandas slice hashes
remain as legacy provenance, not runtime checks. Score parsing explicitly uses
round-trip precision. Runtime loading
checks hashes and derives answers from the stored cells rather than recomputing
expected answers from the current input. Changes to data, parsing, or sampling
must be reviewed explicitly; never regenerate golden answers to make a failed
agent result pass. CSV parsing preserves empty race strings as literal labels,
matching the current loader; it does not silently convert them to a missing group.

Only eligible groups enter headline gaps and test plans. Raw fixture expectations
include suppressed confusion cells for local fixture verification; they are not
LLM evidence or public audit results. The harness must retain the normal privacy
and suppression boundary when passing evidence to the agent.

Distinct synthetic scores avoid duplicate warnings except in missing_score,
where only eight label/group combinations exist and 392 repeated rows are
expected. This overlap is explicitly recorded. Synthetic scores are valid
probabilities for exercising the optional-score path, not calibrated clinical
predictions. The duplicate case repeats complete scored rows deliberately.

The mislabelled prediction fixture is detectably invalid text. Reversed numeric
0/1 labels cannot be identified reliably from values alone and require the user
to confirm positive-class semantics.

## Next step

Build the runner and scorecard around these fixtures: test actual agent calls,
refusals before provider access, structured findings, suppression explanations,
tool selection, numeric tolerance, schema validity, unsupported claims, latency,
and estimated cost. Bootstrap confidence intervals and severity also need their
own reference checks; these fixture checks alone do not establish any agent
accuracy percentage or satisfy the Week 4 benchmark gate.
