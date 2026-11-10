# Claims validation

The agent now runs the local `validate_claims` backend after structured drafting
and deterministic severity correction. No additional model turn or API request
is needed. The six model-selected computation calls remain unchanged. Validation
is a mandatory Python stage, not a tool the model may choose to skip.

The verdict is stored in both `AgentRunResult.claims_validation` and
`AgentRunTrace.claims_validation`. A failed verdict stops the run and preserves
all issues in the failed trace. It does not silently alter unsupported numbers.
The existing report gate accepts only a passing verdict for findings in the same
session; recalculation or another validation request clears that receipt.

## Structured checks

- Dataset row/column counts, score availability, and the exact attribute set.
- Complete eligible and suppressed group sets, preserving label types.
- Per-group counts, six rates, and unavailable-result reasons.
- All headline gaps, confidence bounds, resample counts, and severity labels.
- Every rate test, its method, statistic, p-value, expected count, and reason.
- Explicit insufficient-evidence wording for suppressed groups.
- Preservation of trusted data-quality warnings.

Floating-point comparisons use relative tolerance `1e-9` and absolute tolerance
`1e-12`. Missing values remain missing; zero is never a substitute. Expected
counts may be null for unavailable statistical tests, and must be present for
available tests.

## Narrative screening and limits

The verdict explicitly reports scope `structured_evidence_and_english_rules` and
`requires_human_review: true`. Passing means the implemented automated checks
passed. It does not certify all narrative assertions, clinical suitability,
legal compliance, or model fairness.

English rules flag common affirmative causal claims, fairness or approval
guarantees, and explicit contradictions of group suppression. Narrative numeric
tokens must match visible aggregate evidence; decimal and percentage rendering
may round to the precision written. Arbitrary group labels and random seeds do
not supply numeric evidence.

These rules cannot resolve arbitrary paraphrases, negation, all languages,
cross-sentence reasoning, or whether an otherwise supported number has been
attributed to the correct group or metric. A number appearing somewhere in
trusted evidence is necessary, but not sufficient, for a correct prose claim.
Human review remains required, including for significance and severity wording.
Unavailable reasons are copied exactly from evidence to prevent unsupported
rewriting. A harmless paraphrase can therefore require correction.

The numerical evidence itself comes from the existing audit tools. Validation
does not rerun statistical calculations or independently establish data quality.
Rendering a PDF and unrestricted semantic narrative validation remain separate
work.
