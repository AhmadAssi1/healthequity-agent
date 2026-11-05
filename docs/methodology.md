# Fairness Audit Methodology

## 1. Purpose and scope

HealthEquity Agent compares binary classification predictions across
demographic groups. The demonstration task is hospital readmission
within fewer than 30 days.

All numbers are calculated by deterministic Python functions.
The results describe associations and differences; they do not establish
their causes or determine clinical suitability or legal compliance.

## 2. Audit inputs

Each row represents one observation, such as a hospital stay.

Required columns:
- outcome: the actual binary outcome, coded as 0 or 1.
- prediction: the predicted binary label, coded as 0 or 1.
- At least one group attribute, such as race or age.

An optional score column contains predicted probabilities for outcome 1.

The prediction threshold is fixed before auditing. Different group
attributes are assessed independently, starting from the original data.

Missing group values are retained as a separate group, distinct from
a literal category named "Missing".

## 3. Confusion matrix

For each group:

- TP: outcome 1, prediction 1.
- FN: outcome 1, prediction 0.
- FP: outcome 0, prediction 1.
- TN: outcome 0, prediction 0.
- N = TP + FN + FP + TN.

## 4. Per-group metrics

| Metric | Formula | Interpretation |
|---|---|---|
| Count | N | Number of observations |
| Base rate | (TP + FN) / N | Fraction with actual positive outcomes |
| Selection rate | (TP + FP) / N | Fraction predicted positive |
| TPR / recall | TP / (TP + FN) | Fraction of actual positives detected |
| FPR | FP / (FP + TN) | Fraction of actual negatives incorrectly flagged |
| PPV / precision | TP / (TP + FP) | Fraction of positive predictions that are correct |

AUROC is calculated when a score column is supplied. It measures how
well scores rank actual positives above actual negatives.

A metric with a zero denominator is unavailable, not zero.
AUROC is unavailable when a group contains only one outcome class.

## 5. Group eligibility and suppression

The default reporting policy requires both:

- At least 50 observations.
- At least 10 actual positive outcomes.

These thresholds are configurable in config.yaml.

A group failing either rule is excluded from fairness comparisons,
significance tests, and bootstrap intervals. Every failed rule is
recorded as a suppression reason.

Suppression does not modify the original dataset. Missing-value groups
follow the same eligibility rules as other groups.

At least two eligible groups are required for a comparison.
If fewer remain, the comparison is unavailable, not a zero gap.

Eligibility thresholds are reporting rules. They do not guarantee
precise estimates or adequate denominators for every metric.

## 6. Headline fairness differences

The engine compares eligible groups using maximum pairwise differences.

### Demographic parity difference

DPD = maximum selection rate - minimum selection rate.

This describes differences in how often groups are flagged, regardless
of their actual outcome rates.

### Equalized odds difference

TPR gap = maximum TPR - minimum TPR.

FPR gap = maximum FPR - minimum FPR.

EOD = maximum of the TPR gap and FPR gap.

Both component gaps are reported. If a required group rate is undefined,
the corresponding comparison is unavailable. EOD requires both component
gaps to be available.

### Predictive parity difference

PPV difference = maximum PPV - minimum PPV.

This compares the reliability of positive predictions. It is interpreted
alongside base rates because PPV depends on outcome prevalence.

A difference of 0.20 means 20 percentage points.

The default reference is the largest eligible group. Signed reference
comparisons use rate(group) - rate(reference). The headline maximum
pairwise differences do not depend on the reference choice.

The supporting disparate impact ratio is:

minimum selection rate / maximum selection rate.

It is unavailable when all selection rates are zero.

## 7. Significance tests

Tests assess whether group rates differ under a null hypothesis of
equal rates, assuming independent observations.

The denominator depends on the rate:

| Rate | Positive count | Total |
|---|---|---|
| Selection rate | TP + FP | N |
| TPR | TP | TP + FN |
| FPR | FP | FP + TN |
| PPV | TP | TP + FP |

Expected cell count =
row total × column total / grand total.

The selection rule is:

- Any expected cell count below 5: Fisher's test.
- Otherwise, two groups: pooled two-proportion z-test.
- Otherwise, more than two groups: Pearson chi-square test.

For two groups:

pooled rate = (x_A + x_B) / (n_A + n_B).

SE = sqrt(pooled rate × (1 - pooled rate) × (1/n_A + 1/n_B)).

z = (x_A/n_A - x_B/n_B) / SE.

The z-test uses a two-sided p-value.

For multiple groups:

chi-square = sum((observed - expected)^2 / expected).

Degrees of freedom = number of groups - 1.

Fisher's test is exact for two-by-two tables. For larger sparse tables,
the implementation estimates its p-value using fixed-margin Monte Carlo
sampling with a fixed seed.

Tests with a zero required group total or a degenerate outcome table
cannot provide a valid comparison.

TPR and FPR tests are separate components of equalized odds.
The current helpers do not produce a combined EOD p-value.

A p-value is not the probability that the null hypothesis is true.
Current p-values are unadjusted for multiple testing.

## 8. Bootstrap confidence intervals

Each headline difference receives a percentile 95% confidence interval
using 1,000 resamples and seed 42 by default.

Resampling occurs independently within each eligible group, with
replacement, preserving its size and paired outcome/prediction values.

The implementation samples confusion-matrix counts from a multinomial
distribution. For these metrics, this is equivalent to resampling rows
within each group.

The 2.5th and 97.5th percentiles of valid bootstrap differences form
the interval.

Undefined resamples are excluded separately for each metric, and the
number of valid resamples is reported. No interval is returned when
the observed difference is undefined or no valid resamples remain.

These intervals estimate uncertainty under the resampling scheme.
Because maximum gaps are nonnegative, excluding zero from a percentile
interval is not a replacement for a hypothesis test of equal rates.

## 9. Severity policy

Default thresholds are configurable in config.yaml.

| Classification | Rule |
|---|---|
| substantial | Observed gap > 0.10 and CI lower bound > 0 |
| watch | Observed gap > 0.05 without meeting the substantial rule |
| no_evidence_found | Available result meeting neither rule |
| unavailable | Gap or confidence bounds are missing |

Threshold comparisons are strict: a gap equal to 0.10 does not meet
the substantial threshold.

Severity is a project reporting convention, separate from significance
testing. "No evidence found" does not prove fairness.

## 10. Verification

Synthetic scenarios contain known confusion-matrix counts and planted
selection-rate, TPR, or PPV gaps.

The current generator supports gaps from 0 to 0.50 in increments of
0.01, with 200 observations per group. Changing a targeted rate may
also change other metrics.

Tests check hand-calculated results, Fairlearn reference values,
eligibility boundaries, configuration, unavailable results, and
reproducibility. Numerical comparisons use a tolerance of 1e-6.

## 11. Limitations

- Different fairness metrics can disagree.
- Results depend on outcome definitions, data quality, and the fixed
  prediction threshold.
- Suppressed groups remain unassessed; their exclusion is not evidence
  of fairness.
- Small denominators can produce wide confidence intervals even when
  a group passes eligibility rules.
- Repeated stays from the same patient may violate independence.
  The current bootstrap resamples stays, not patient clusters.
- Historical readmission outcomes may reflect differences in access
  to care as well as patient health.
- Findings do not establish discrimination, causation, or the absence
  of harm.