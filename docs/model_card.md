# Model Card — Experimental Readmission Baseline

## Identity and intended use

- Author: Ahmad Assi.
- Model: LightGBM binary classifier with 15 leaves.
- Purpose: provide demonstration predictions for developing and testing the
  HealthEquity Agent performance and fairness auditing workflow.
- Prediction time: hospital discharge. Each prediction refers to one hospital stay.
- Positive class: recorded readmission within fewer than 30 days (`readmitted = "<30"`).
- Negative class: `readmitted = "NO"` or `">30"`.
- This is an experimental model, not a validated clinical decision-support system.
  It must not determine patient treatment, triage, or access to services.

## Data and cohort

The source is the UCI Diabetes 130-US Hospitals dataset (1999–2008), containing
101,766 hospital stays. See the [Data Card](data_card.md) for dataset details.

Exclude discharge disposition codes 11, 19, 20, and 21 under the documented
death-related discharge eligibility rule. This removes 1,652 stays and leaves
100,114. Unknown dispositions remain included; their presence does not confirm
survival. The original source file remains unchanged.

GroupShuffleSplit with random_state=42 separates patients, keeping all stays for
one patient in one partition. It is not a temporal or explicitly outcome-stratified split.

| Partition | Stays | Patients | Positive outcomes |
|---|---:|---:|---:|
| Train | 69,944 | 49,307 | 11.30% |
| Validation | 15,172 | 10,566 | 11.52% |
| Test | 14,998 | 10,566 | 11.35% |

Patient sets do not overlap. Test was evaluated after the model, preprocessing
schema, and validation-selected threshold were fixed. It was not used for selection.

## Inputs and preprocessing

The model uses 44 features: eight numerical and 36 categorical.

Numerical features: time_in_hospital, num_lab_procedures, num_procedures,
num_medications, number_outpatient, number_emergency, number_inpatient, and
number_diagnoses.

Other predictors are categorical, including admission/discharge codes, diagnoses,
medications, race, gender, age bands, and payer_code. Demographics are retained as
predictors in this baseline; their use does not establish fairness and requires
explicit examination during the audit.

- Drop weight because 96.86% of its source values are missing.
- Exclude target, readmitted, encounter_id, and patient_nbr from predictors.
- Exclude examide and citoglipton, which are constant in Train.
- Learn category levels from Train only; convert unseen categories to missing.
- Preserve diagnosis codes as strings and the literal laboratory category "None".
- Retain missing feature values for LightGBM's native handling; no imputer,
  scaling, or oversampling is applied.
- Use the saved config.json as the authoritative feature order and category schema.

Discharge-time inputs must not be used to claim that the model predicts risk at
admission, when those inputs would not yet be available.

## Training and model selection

| Setting | Value |
|---|---|
| Objective | Binary classification |
| Learning rate | 0.05 |
| Maximum boosting iterations | 1,000 |
| Leaves | 15 |
| Early-stopping metric | Validation average precision |
| Early-stopping patience | 50 iterations |
| Selected iteration | 54 |
| Seed | 42 |
| Deterministic settings | deterministic=True, force_col_wise=True |
| Class weighting | None |
| LightGBM version | 4.7.0 |
| MLflow version | 3.17.2 |

The alternative changes num_leaves from 31 to 15 while retaining the original
training settings. It was selected for slightly better validation average
precision, ROC-AUC, and Brier score. Statistical significance of the improvement
was not assessed.

Class imbalance is addressed in evaluation and operating-point selection using
average precision, precision, recall, and F1 rather than accuracy alone. No class
reweighting or resampling was performed; this remains a baseline design choice,
not a claim that class imbalance has been resolved.

## Operating threshold

Predict 1 when the positive-class score is at least **0.145**. The stored
floating-point value is approximately 0.14499999999999996.

Search thresholds from 0.05 through 0.30 in steps of 0.005 and select the highest
validation F1, choosing the larger threshold for exact ties. This rule selects an
experimental operating point; it does not represent agreed clinical error costs
or follow-up capacity. The original 31-leaf model's threshold of 0.135 is not used
for this selected model.

## Validation results

| Metric | Selected baseline |
|---|---:|
| Average precision (AP) | 0.22211 |
| ROC-AUC | 0.67157 |
| Brier score | 0.09738 |
| Precision at 0.145 | 0.2106 |
| Recall at 0.145 | 0.4273 |
| F1 at 0.145 | 0.2822 |
| Stays flagged positive | 23.38% |

AP is computed with scikit-learn average_precision_score; it is not a trapezoidal
integration of the precision–recall curve. Validation positive prevalence is
11.52%, providing a no-skill ranking reference.

Validation was used for early stopping, model comparison, and threshold selection.
These results are development estimates, not independent final Test performance.
Brier score measures probability error. Calibration was assessed on Validation
using 10 quantile bins. Mean predicted probabilities were descriptively close to
observed readmission rates across the displayed bins, with some overprediction
and underprediction. No confidence intervals were computed, and subgroup and
Test calibration remain unevaluated. No recalibration was applied; the model and
threshold of 0.145 remain unchanged. The plot, table, and binning settings were
saved to the selected MLflow run. No subgroup fairness results are claimed here.

## Final Test results

| Metric | Selected baseline |
|---|---:|
| Average precision (AP) | 0.21742 |
| ROC-AUC | 0.66039 |
| Brier score | 0.09645 |
| Precision at 0.145 | 0.2043 |
| Recall at 0.145 | 0.4219 |
| F1 at 0.145 | 0.2753 |
| Stays flagged positive | 23.43% |

Results cover all 14,998 Test stays and are logged with test_ metric names in
the selected run. No retraining or threshold adjustment followed this evaluation.
These are held-out results on the same source dataset, not external validation.
The Brier score alone does not establish Test calibration.

## Artifacts and reproduction

- Implementation: [model_development.ipynb](../notebooks/model_development.ipynb).
- Cohort and partitions: [exploration.ipynb](../notebooks/exploration.ipynb).
- MLflow experiment: readmission-baseline.
- Selected run: a68871a678b842828033d442632a63cc.
- Run name: lightgbm-baseline-15-leaves.
- Artifact files: baseline/model.txt and baseline/config.json.
- Calibration artifacts: calibration/validation_curve.png and
  calibration/validation_table.json (10 quantile bins on Validation).
- Local tracking: mlflow.db and mlartifacts/; both are excluded from Git.
- Registered model: healthequity-readmission-baseline, verified Version 1.
- Registry URI: models:/healthequity-readmission-baseline/1.
- The registered model includes the configuration file. Preprocessing must still
  be applied before prediction; the configuration is not an automatic pipeline.
- Reloading Version 1 reproduced the original validation scores.
- Demo audit file: data/samples/diabetes_audit_sample.csv, with 14,998 rows and
  seven columns: outcome, prediction, score, race, gender, age, and payer_code.
- Provenance file: data/samples/diabetes_audit_sample_metadata.json.
- The exported CSV passed schema checks and a round-trip comparison, including
  missing values and numerical tolerances for probability scores.

Generate the partitions with exploration.ipynb, then run model_development.ipynb
from top to bottom using the project environment. New executions create new run
IDs. Dependencies are not pinned, so exact reproduction across newly installed
environments is not guaranteed. The recorded versions and deterministic settings
describe the reviewed execution.

Reloading the model and JSON configuration reproduces the original validation
scores and thresholded labels. A separate quality review also reproduced both
models' results from fresh execution. The saved files are local artifacts, not
bundled with a fresh repository clone.

## Limitations and remaining evaluation

- The historical dataset does not establish performance on current hospitals or
  populations. There is no external or temporal validation.
- Readmission is the recorded outcome, not a direct measure of clinical need.
- Multiple stays from a patient are correlated. Future uncertainty estimates
  should account for that dependence.
- On Test at the selected threshold, recall is 42.19% and precision is 20.43%; many positive
  stays are missed and many flags are false positives.
- Missing and unseen categories, rare groups, and demographic predictors require
  further data-quality and subgroup review.
- Fairness metrics, suppression policies, uncertainty, and Test and subgroup
  calibration remain to be evaluated; no clinical, fairness, or compliance approval
  is implied.
- Keep the evaluated model, preprocessing schema, threshold, and demo provenance
  fixed. Do not use Test results to tune this baseline.
