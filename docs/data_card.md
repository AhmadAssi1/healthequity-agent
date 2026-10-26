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
- Missing-value handling for model training has not yet been applied.

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


## Planned data splitting
- Training, validation, and test sets will be split by patient_nbr.
- Each patient must appear in only one split to prevent information leakage across repeated hospital stays.