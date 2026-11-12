# Claims validation

The agent now runs the local `validate_claims` backend after structured drafting
and deterministic severity correction. Validation itself makes no API request.
If it fails, one additional model response is allowed to rewrite the complete
findings using the rejected draft, validator feedback, and trusted evidence.
The six model-selected computation calls remain unchanged. Validation
is a mandatory Python stage, not a tool the model may choose to skip.

The verdict is stored in both `AgentRunResult.claims_validation` and
`AgentRunTrace.claims_validation`. `claims_validation_attempts` preserves every
verdict, including an initial failure followed by success. A second failure
stops the run and preserves all issues in the failed trace. The default
`max_claim_rewrites` is one and cannot exceed one; zero disables rewriting.
The rewrite receives no tools, must remain within existing turn/time budgets,
and has no additional structural-retry cycle. The original one-schema-retry
cap remains. Validation does not silently alter unsupported numbers.
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

Final-output instructions require every suppression reason to start with
`Insufficient evidence: ` and then explain the trusted reason codes. Bare codes
such as `below_min_group_size` do not provide the required explicit statement.
This wording rule was exercised against a saved live rejection containing ten
such suppression reasons; changing only the wording removed those ten issues.
That offline experiment is not evidence of a successful live rewrite.

## Narrative screening and limits

The verdict explicitly reports scope `structured_evidence_and_english_rules` and
`requires_human_review: true`. Passing means the implemented automated checks
passed. It does not certify all narrative assertions, clinical suitability,
legal compliance, or model fairness.

English rules flag common affirmative causal claims, fairness or approval
guarantees, and explicit contradictions of group suppression. Negation is checked
near each claim within its clause: `No score exists, but race causes the gap`
is rejected. Discrimination claims such as `because of discrimination`,
`due to racial bias`, and `driven by gender` are rejected. Policy explanations
such as suppression due to insufficient sample size remain allowed.
Generic `due to` and `because of` phrases are screened as disparity-cause claims
when their clause mentions demographic/model differences or discrimination.
Procedural descriptions such as bootstrap variability due to random sampling
are not automatically treated as demographic causal claims. Direct causal verbs
remain screened. This context rule is still a heuristic and can miss implicit
or paraphrased causal assertions.

Narrative group references use exact quoted labels. Prefer attribute-qualified
forms such as `race: "Caucasian"`, `age: "[70-80)"`, or `cohort: null`. JSON
numeric and Boolean values preserve their types. A quoted label alone must
identify a group in exactly one attribute, or have one explicit attribute anchor
in the same sentence segment (for example `For attribute race ... ""`). Scope
does not carry across sentences or semicolons, and conflicting anchors require
an inline attribute/value reference. Unknown names, wrong-attribute
references, type mismatches, ambiguous labels, and unknown bare group-X constructions
are rejected. An exact unquoted label in a recognized group-X or X-group phrase
can pass when it identifies one attribute unambiguously. Quotes are reserved for exact group labels and known column/metric
identifiers, not ordinary prose. Quoted age bands and numeric labels are masked
before numeric-claim checks so their labels are not mistaken for outcome values.
Suppression-reason text is screened as well as the summary, warnings, and
limitations. Verified quoted labels are data, not causal assertions.

Numeric annotations such as `p=...` and `CI: ...` are not group references unless
their key is itself a mapped group attribute. Their numeric values remain subject
to numeric checks; they are not masked or automatically approved. Recognized
group/attribute/row/resample count phrases from zero through twenty are normalized
from words to digits and checked. Counts of visible eligible/suppressed label lists
are derived from evidence, not invented or inferred hidden row counts. This is a
limited count grammar, not a general English-number parser.

Narrative numeric
tokens must match visible aggregate evidence; decimal and percentage rendering
may round to the precision written. Arbitrary group labels and random seeds do
not supply numeric evidence.

These controlled-language rules cannot resolve arbitrary paraphrases, all forms
of negation, all languages, or arbitrary unmarked group mentions. Model
instructions require quoted group references, but this is not a proof that
unrestricted prose contains no hidden group names. They also cannot resolve
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

## Quantitative summary grounding

Live inspection found that corrected structured severity could disagree with
the model's summary: race PPV was watch in the table but substantial in prose.
After the full draft passes validation (including any allowed rewrite), Python
now renders the final quantitative summary from `AuditEvidence`. It includes
the row/attribute counts, visible group counts, each gap, interval, policy label,
and rate-test p-value. It does not infer significance from a CI or invent a
combined EOD test or significance level.

The original model summary remains in its response and in
`trace.summary_grounding.proposed_summary`. The record includes the generated
summary, source `trusted_tool_evidence`, and a fresh validation verdict. The
report receipt is refreshed for the actual final findings. All other findings
fields remain unchanged. A failed initial draft is not erased to manufacture
approval; it must first pass the original checks. Failure of generated-summary
validation stops completion.

This is a deliberate tradeoff: the final quantitative summary is a deterministic
rendering rather than unrestricted model prose. The model still plans, supplies
the structured draft, and drafts limitations. Free-text limitations and additional
interpretations still require human review under the limits above. Grounding the
summary does not prove all narrative claims correct or render a PDF.
