# project scope 

Build an AI agent to assess data quality, analyze dataset characteristics and model performance, and audit fairness across demographic groups using evidence-based statistical analysis.

# inputs and outputs


## Inputs

- A synthetic or de-identified CSV or XLSX dataset containing:
  - Ground-truth outcomes.
  - Model-predicted labels, if available. If absent, suitable input features are required for the session-based baseline training path.
  - Demographic attributes for group comparisons, such as gender or age group.
  - Optional predicted probabilities for the positive class.
- User-provided column mappings and the definition of the positive class.
- The initial version supports one binary classification task: 30-day hospital readmission risk.
    
## Outputs

- A data quality and dataset summary covering missing values, duplicates, class balance, and demographic group sizes.
- Overall and per-group model performance metrics.
- A fairness audit covering demographic parity, equalized odds, and predictive parity differences, with bootstrap confidence intervals and significance tests where applicable.
- A downloadable PDF report explaining findings, limitations, and recommended next steps.
- All reported numbers computed by Python tools, with explanations grounded in verified tool outputs.

## In Scope — Version 1

- Audit binary classification predictions supplied in de-identified datasets.
- Assess data quality and summarize dataset characteristics.
- Evaluate model performance overall and across demographic groups.
- Compute demographic parity difference, equalized odds difference, and predictive parity (PPV) difference, with bootstrap confidence intervals and appropriate significance tests.
- Use an AI agent to select appropriate analyses and explain findings, with deterministic Python tools computing all reported numbers.
- Generate an evidence-based PDF audit report with findings, limitations, and recommended next steps.
- Validate the auditing workflow on 30-day hospital readmission predictions.
- Train a baseline LightGBM readmission model to provide demonstration predictions. If uploaded data has no prediction column and is suitable for baseline training, train the baseline within the session and audit its predictions. Clearly label it as a demo model, not the user's model.
- Structure the implementation to allow future extensions.

## Out of Scope — Version 1

- Regression, multiclass classification, and other prediction tasks.
- Training or running user-uploaded model artifacts within the auditing workflow. The application-provided session-based baseline described above is permitted.
- Automatically modifying models or applying bias mitigation.
- Clinical recommendations or legal compliance determinations.
- Claims of validated support for domains beyond the initial use case.