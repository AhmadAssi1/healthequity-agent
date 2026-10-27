# Unit of observation

Version 1 is limited to binary classification of hospital readmission within 30 days. Other prediction tasks are outside the scope of this release.

Each row in the canonical audit dataset represents one hospital stay with a ground-truth outcome and a model prediction. For the UCI Diabetes demonstration, outcome 1 corresponds to readmitted = "<30"; outcome 0 corresponds to "NO" or ">30". Users must confirm that uploaded outcomes and predictions refer to the same task and positive class.

# Required fields

These fields are required in the canonical audit dataset before performance and fairness metrics are computed.

- outcome: The ground-truth outcome. It is required in uploaded data, must not be missing, and must be either 0 or 1. The value 1 represents the positive class, as defined by the user.

- prediction: The model-predicted label. It must not be missing and must be either 0 or 1 in the canonical audit dataset. The value 1 must represent the same positive class as outcome. Uploaded data may omit this column only when using the baseline training path described in Prediction availability.

## Optional fields

- **score**: The model-predicted probability of the positive class. This column is optional. If provided, all values must be numeric, must not be missing, and must fall between 0 and 1, inclusive. The positive class must match the class represented by 1 in outcome and prediction.

## Group columns

- At least one user-selected group column is required for fairness analysis, such as gender, race, or age_group.
- Group column names are not fixed; the user specifies which columns to analyze.
- Group columns must represent discrete categories. Numeric category codes are permitted when their meanings are documented.
- Continuous attributes, such as numeric age, must be converted to documented groups before fairness comparisons. Group boundaries must be explicitly defined; the system must not silently create one group per numeric value.
- Missing group values are reported as a separate "Missing" group and are not silently dropped.

## Column mapping

- Users must identify the outcome column and one or more group columns.
- Users must identify the prediction column when using uploaded predictions. Otherwise, predictions must be generated through the baseline training path before auditing.
- If a score column is provided, users must identify it as well.
- Input column names are not fixed. For example, actual_result may map to outcome, and model_output may map to prediction.
- Every mapped column must exist in the uploaded dataset.
- Users must define the positive class. The value 1 must represent the same positive class in outcome and prediction, and score must represent its predicted probability.

## General validation rules

- The dataset must contain at least one data row.
- Column names must be unique.
- Each row must pair the ground-truth outcome and model prediction for the same observation.
- If the dataset violates this contract, analysis must stop with a clear validation error.
- Invalid rows must not be silently dropped, and values must not be silently changed.
- Missing group values are handled according to the policy defined in the Group columns section.

## File requirements and privacy

- Supported upload formats are CSV and XLSX.
- The maximum upload size is 25 MB.
- Uploaded data must be synthetic or de-identified.
- Users must confirm that the uploaded file contains no protected health information (PHI).
- Uploaded data is processed in memory and discarded at the end of the session.
- Row-level data, free-text fields, and identifiers must not be sent to the language model.
- Only aggregated metrics for groups meeting the suppression thresholds may be logged.

## Dataset fingerprint

- The system generates a dataset fingerprint containing the row count and a hash of the column names.
- Users are not required to provide this fingerprint.
- Column names are hashed in their original order using a deterministic encoding and hash algorithm.
- The fingerprint describes the dataset structure; it does not verify that the row values are identical.

## Prediction availability

- The canonical audit dataset must contain a prediction column before performance and fairness metrics are computed.
- If an uploaded dataset contains predictions, users must map the prediction column.
- If predictions are absent, the project brief specifies a session-based baseline LightGBM training path. Its predictions must be clearly labelled as coming from a demo model, not the user's model.
- The baseline training path requires suitable input features and a documented validation procedure; outcome and group columns alone do not guarantee that training is possible.

## Validation errors and data quality warnings

- Validation errors block the audit. Examples include missing required columns, missing outcome or prediction values in the canonical audit dataset, non-binary labels, and invalid score values.
- Data quality warnings do not automatically block the audit. Examples include duplicate rows, class imbalance, and small groups.
- Duplicate rows must be reported and must not be silently removed.
- Groups with fewer than 50 members or fewer than 10 positive outcomes are suppressed, with the reason reported.
- Analysis may continue for eligible groups. If fewer than two eligible groups remain for an attribute, fairness gaps for that attribute are reported as unavailable, with an explanation.
- Undefined metrics must be reported as unavailable with a reason, rather than replaced with zero.

## Data types and value handling

- outcome and prediction must contain numeric binary labels: 0 or 1. Values such as 0.0 and 1.0 are accepted.
- Boolean labels and text labels are not accepted. Values must not be silently converted.
- If score is provided, its values must be numeric and finite, between 0 and 1 inclusive. Missing values and positive or negative infinity are not accepted.
- Additional columns are permitted and do not need to be mapped unless they are selected for analysis.
- Missing group values must remain distinguishable from any existing group whose literal name is "Missing".