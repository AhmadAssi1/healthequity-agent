# Data Card — UCI Diabetes

## Unit of observation
Each row represents one hospital stay.
The same patient may have multiple stays.

## Target
- 1: Readmission within fewer than 30 days.
- 0: Readmission after more than 30 days, or no recorded readmission.

## Dataset size and repeated patients
- Number of hospital stays: 101,766.
- Patients with more than one stay: 16,773.
- Maximum stays for one patient: 40.

## Target distribution
- Target 0: 90,409 stays (88.84%).
- Target 1: 11,357 stays (11.16%).
- The target classes are imbalanced.

## Missing values
- The "?" marker is treated as a missing value during loading.
- The literal string "None" is preserved as a category.
- Missing feature values are retained and handled directly by LightGBM.

## Length of stay
- time_in_hospital is right-skewed.
- Q1: 2 days; median: 4 days; Q3: 6 days.
- The middle 50% of stays fall between 2 and 6 days.
- Values of 13 and 14 days exceed the upper IQR fence of 12 days.
- These values are retained; being statistical outliers does not imply an error.

## Data source
- Dataset: Diabetes 130-US Hospitals for Years 1999–2008.
- Source: UCI Machine Learning Repository.
- Local file: diabetic_data.csv.
- Dataset page: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated
  by UCI. This is the dataset license, not a software license for this repository.
- Attribution: Clore, J., Cios, K., DeShazo, J., and Strack, B. (2014).
  Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5230J.

## Column overview and known quirks

The local source CSV has 50 columns. Categories and missing counts below refer
to the full source data, before cohort exclusions or splitting.

| Column family | Examples | Interpretation |
|---|---|---|
| Identifiers | encounter_id, patient_nbr | Encounter and patient IDs; excluded from predictors and the audit sample |
| Demographics | race, gender, age | Recorded categories; age is a band, not an exact age |
| Payer | payer_code | Categorical payer/insurance code; not a measured income value |
| Admission and discharge | admission_type_id, admission_source_id, discharge_disposition_id | Numeric category codes; consult IDS_mapping.csv |
| Quantitative inputs | time_in_hospital, num_medications, number_inpatient | Durations and counts; eight numerical predictors in the baseline |
| Diagnoses | diag_1, diag_2, diag_3 | Categorical codes retained as strings |
| Laboratory and treatment | A1Cresult, max_glu_serum, insulin, change | Test-result and medication categories |
| Outcome | readmitted | Source category used to derive the binary target |

- race has five observed non-missing categories: Caucasian, AfricanAmerican,
  Asian, Hispanic, and Other. There are 2,273 missing values (2.23%). Other is a
  recorded category and must not be merged with missing values.
- gender contains Female, Male, and the rare Unknown/Invalid category. The latter
  is retained as recorded, not relabelled as Female or Male. Later fairness
  reporting must apply the documented small-group suppression policy.
- age contains ten bands from [0-10) to [90-100). Each includes its lower bound
  and excludes its upper bound; exact ages cannot be recovered from these bands.
- payer_code has 17 observed non-missing codes and 40,256 missing values (39.56%).
  Preserve codes and missingness; do not infer income or silently invent code
  meanings. Any later socioeconomic interpretation must state its limitations.
- Admission/discharge codes can encode unavailable or unmapped information;
  a numeric storage type does not make them quantitative measurements.
- "None" in A1Cresult and max_glu_serum denotes a test not performed. Keep it
  separate from normal results and from missing CSV cells.
- Repeat encounters from the same patient are expected, not automatically duplicate
  records. Patient-level splitting prevents overlap, but does not make stays
  statistically independent.

## Working dataset
- The weight column was removed from the working copy because 96.86% of its values are missing.
- The original dataset remains unchanged.
- The target column was added to the working copy.


## Modeling cohort and data splitting

- Predictions are intended to be made at hospital discharge.
- Stays with death-related discharge codes 11, 19, 20, and 21
  were excluded before splitting.
- This removed 1,652 stays, leaving 100,114 stays.
- The original dataset remains unchanged.
- GroupShuffleSplit with random_state=42 created patient-disjoint
  Train, Validation, and Test partitions.

| Split | Hospital stays | Unique patients | Positive outcome rate |
|---|---:|---:|---:|
| Train | 69,944 | 49,307 | 11.30% |
| Validation | 15,172 | 10,566 | 11.52% |
| Test | 14,998 | 10,566 | 11.35% |

- No patient appears in more than one split.
- Combined partitions cover the modeling cohort without duplicate
  encounter IDs.
- Files are saved under data/splits/.
- Validation supports model and threshold selection.
- Test was evaluated after fixing the model and validation-selected threshold;
  it was not used for model or threshold selection.

## Baseline preprocessing

- The baseline uses 44 features: 8 numeric and 36 categorical.
- Identifiers, readmitted, target, examide, and citoglipton
  are excluded from predictors.
- Categorical levels are learned from Train only.
- Unseen categories are converted to missing values.
- LightGBM handles missing feature values directly.
- The literal laboratory category "None" is preserved.
- Model configuration records feature order and categorical levels.

## Standing demo audit file

- File: data/samples/diabetes_audit_sample.csv.
- Contains all 14,998 Test stays, with outcome, prediction, score, race, gender,
  age, and payer_code columns. Patient and encounter identifiers are excluded.
- Predictions use the selected 15-leaf LightGBM baseline and its saved threshold
  of approximately 0.145.
- Missing demographic values remain missing; they must be retained as distinct
  missing groups in the subsequent fairness audit.
- Age is categorical: [60-70), for example, includes 60 and excludes 70.
- The companion diabetes_audit_sample_metadata.json records the source split,
  model run, threshold, positive class, group columns, and row count.
- The CSV was validated before export and compared against the in-memory table
  after reloading. See the [Model Card](model_card.md) for Test results.
