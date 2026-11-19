# Week 4 Narrative Review

The November 19 live run evaluated the agent after the limitations change.
[Review of the final live reports](week4-live-review-20261119.md)

The November 17 local review covered batch `5bb6b88b848640f7b4e106a61d820389`.
The November 19 follow-up checked the revised text against that batch and
compared all ten GitHub live reports with their saved evidence. Both reviews
cover summaries, limitations, warnings, group eligibility, confidence
intervals, and rate tests. Human approval of the revised text is pending.

## Local live narrative review

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

The local text incorrectly states that all intervals include zero. The
observed headline gaps are all zero and the four component tests each return
`p = 1.0`. The interval bounds below match both saved live batches.

| Gap | Observed (pp) | 95% bootstrap interval (pp) |
|---|---:|---:|
| Demographic parity | 0.00 | 0.00 to 10.51 |
| Equalized odds | 0.00 | 1.61 to 18.01 |
| Predictive parity (PPV) | 0.00 | 0.21 to 16.12 |

> The demographic parity bootstrap interval includes zero. The equalized odds
> and PPV intervals have positive lower bounds. Interpret each interval
> separately from the component rate tests.

### multiple_groups

The local text incorrectly states that the demographic parity interval includes
zero and attributes significance to group C. In both live batches, the observed
demographic parity gap is `0.00 pp`, its 95% bootstrap interval is `1.00 to 13.01 pp`,
and the selection-rate chi-square test returns `p = 1.0`. The other three tests
each return `p ≈ 0.00438` before any adjustment for multiple comparisons.

> The TPR, FPR and PPV chi-square tests compare rates jointly across the three
> groups. Identifying a statistically significant pair would require a post-hoc
> comparison.

Percentile bootstrap intervals for nonnegative maximum gaps can have positive
lower bounds even when the observed gap is zero. Interpret them alongside the
component rate tests under the documented resampling method.
[Methodology](../methodology.md)

## GitHub live narrative review

The live job produced batch `d18f1c34eeba4d92b4c234363fa81ff0` in artifact
`14319034889`. Its `tiny_group` and `multiple_groups` prose avoided the specific
interval and pairwise claims found in the local batch. The review identified
the following edits. [GitHub Actions run](https://github.com/AhmadAssi1/healthequity-agent/actions/runs/44064595229)

| Scenario | Review finding |
|---|---|
| `no_gap` | Report interval width without attributing it to sample size alone. The fixture had no formal power analysis. |
| `selection_gap` | The label-based tests use observed outcomes and binary predictions. The constructed scores support the AUROC calculation. |
| `tpr_gap` | Each z-test met the expected-cell-count rule. Independence is a sampling assumption. |
| `ppv_gap` | The profiler checked mapped inputs. Its warning list provides limited evidence about data quality. |
| `tiny_group` | Specify the two suppression reasons. `Tiny` is below the group size threshold and `LowPositive` is below the positive-outcome threshold. |
| `multiple_groups` | State that the chi-square tests are joint comparisons across three groups. |
| `missing_score` | Describe repeated mapped combinations and the absence of scores. Keep the zero observed gaps, rate tests, and reporting labels distinct. |
| `duplicated_rows` | The fixture deliberately repeats complete rows. Dependence between the copies can affect the uncertainty estimate. |
| `diabetes_female` | Describe the repeated mapped combinations and report the positive lower interval bounds separately from the nonsignificant rate tests. |
| `diabetes_male` | Apply the substantial label using both the observed gap threshold and the lower interval bound. Report selection-rate and FPR test results separately. |

## Verification and approval

The revised English summaries and limitations passed the findings schema and
claims validator for all ten completed reports in each source batch. The
quantitative summaries match the saved evidence after grammar edits. The
replacement text preserves the structured metrics, suppression results, and
profiler warnings. The two invalid inputs were correctly refused before any
provider request in each batch.
[English revisions and source hashes](week4-narratives-20261117.json)

The original local reports contain the errors documented above. This follow-up
used the saved artifacts and fixture definitions. On November 19, Ahmad Assi
approved a new live batch after the corrections were implemented.
[Approved live evaluation scorecard](week4-live-review-20261119.md)
