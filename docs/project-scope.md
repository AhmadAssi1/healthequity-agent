# project scope 

Build an AI agent to assess data quality, analyze dataset characteristics and model performance, and audit fairness across demographic groups using evidence-based statistical analysis.

# inputs and outputs


## Inputs

- A de-identified CSV or Excel dataset containing:
  - Ground-truth outcomes.
  - Model-predicted labels.
  - Demographic attributes for group comparisons, such as gender or age group.
  - Optional predicted probabilities for the positive class.
- User-provided column mappings and the definition of the positive class.
- The initial version supports binary classification.
    
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
- Train a baseline readmission model to provide demonstration predictions.
- Structure the implementation to allow future extensions.

## Out of Scope — Version 1

- Regression, multiclass classification, and other prediction tasks.
- Training or running user-uploaded models within the auditing workflow.
- Automatically modifying models or applying bias mitigation.
- Clinical recommendations or legal compliance determinations.
- Claims of validated support for domains beyond the initial use case.