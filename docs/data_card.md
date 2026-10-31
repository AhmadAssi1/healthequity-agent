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
- Test remains reserved for final evaluation.

## Baseline preprocessing

- The baseline uses 44 features: 8 numeric and 36 categorical.
- Identifiers, readmitted, target, examide, and citoglipton
  are excluded from predictors.
- Categorical levels are learned from Train only.
- Unseen categories are converted to missing values.
- LightGBM handles missing feature values directly.
- The literal laboratory category "None" is preserved.
- Model configuration records feature order and categorical levels.
