# Week 4 evaluation runner

Scenario definitions and independent fixture answers are documented in
[the scenario catalogue](week4-evaluation-scenarios.md).

## Running

From the repository root with the project environment active:

```powershell
python -m evals
python -m evals --live
python -m evals --live --scenario no_gap
```

Default mode uses a scripted provider, real tools, and real agent orchestration.
It exercises the evaluation pipeline without network calls. Its measurements
are explicitly labelled offline and must not be presented as LLM accuracy.
Live mode uses the existing configured OpenAI adapter and model. Each valid
scenario creates a fresh session/provider; invalid inputs are validated before
any provider construction. There are ten valid cases and two refusal cases.

Outputs are unique directories under `outputs/evaluations`, with JSON and
Markdown scorecards, findings and traces. No input dataframes are saved.
Existing audit histories are never overwritten. The full planned set is written
before execution, checkpointed after each case, and failures/not-run cases stay
in the denominators. An interruption therefore cannot turn a partial suite into
a perfect twelve-case score. A normal exit code 0 means all selected automatic
checks passed, not that human review or the entire Week 4 gate is complete.

## Score definitions

- Tool-selection accuracy: fraction of scheduled valid scenarios with all six
  stages successfully executed, full attribute/rate/metric coverage, correct
  executable test methods, and bootstrap calls using 1,000 resamples/seed 42.
  Original model plan proposals are scored separately by decisions, not prose.
  Delegated deterministic planning is not credited as a correct model proposal.
- Numeric accuracy: fraction of scheduled valid cases with every fixture-backed
  group count, base rate, selection rate, TPR, FPR, PPV and headline gap correct,
  plus matching dataset shape. Relative tolerance 1e-6, absolute tolerance 1e-9.
  This is an all-checks-per-case rate, not a fraction inflated by many easy fields.
- Schema validity: completed findings pass the strict `AuditFindings` schema.
  Structured-request errors/retries are also recorded per run; eventual success
  does not imply a valid first attempt.
- Suppression correctness: exact reportable/suppressed sets and an explicit
  insufficient-evidence explanation; no suppressed rows enter headline gaps.
- Refusal correctness: both invalid cases raise the expected validation error
  without creating a provider. A transport failure is not a correct refusal.
- Automated claims validation: final trace verdict and final grounded-summary
  verdict pass. This checks CIs, test statistics, AUROC and severity against
  deterministic tool evidence, not independent fixture reference numbers.
- Unsupported claim rate: unknown until human narrative review; the rule-based
  validator cannot certify unrestricted prose. Benchmark remains open.
- Latency: end-to-end wall seconds per case, including validation and provider
  work; setup/saving the suite scorecard is outside each case timer.
- Cost: complete trace input/output usage multiplied by explicitly supplied USD
  per million token prices (`--input-price` and `--output-price`). No prices are
  guessed; missing usage or prices remains null/unknown. Cached input pricing
  is not differentiated, so this is a conservative estimate at supplied rates,
  not a provider invoice. Offline cost is zero. Failed transport requests can
  have unknown billed usage.

The scoring module does not change historical evidence, fix findings, or hide
failures. The scorecard's `live_benchmark_passed` remains false while unrestricted
narrative review is pending. Workflow automation/MLflow integration and PDF
reporting are subsequent Week 4 deliverables.

## Local evidence: November 16, 2026

- Full local suite: 1,131 passed in 15.29 seconds; Ruff clean.
- New fixture/runner tests: 48 passed, including deliberate numeric errors,
  missing tools/rates, wrong bootstrap configuration, malformed plan proposals,
  failures retained in denominators, refusal before provider construction,
  secret-text redaction, artifact persistence, and missing token usage.
- Offline suite `619221907c41422aa0ac62bc69fca86e`: ten valid scenarios
  completed, both invalid cases refused; all scoped automatic checks passed.
  This uses a scripted provider and establishes no live model accuracy.
- Live attempt `e5ddde5156f24c8e8e55a814825fccfd` on configured
  `gpt-4.1-mini`: ten valid cases failed on the first provider request with
  `APIConnectionError`; both invalid cases were correctly refused without
  provider calls. Traces are retained. No successful model response or complete
  token usage was available. This attempt cannot measure model selection,
  numeric reliability or narrative quality. Scorecard audit rates count these
  failed cases as unsuccessful rather than hiding them.
- Run `python -m evals --live` from an environment that can reach the configured
  provider to obtain the first complete live scorecard. Do not replace the
  failed attempt or describe the offline score as live performance.

## First user-run full live batch and follow-up

Run `4942523a675148a8b12b7c7d81e221fc` completed nine of ten valid audits;
both invalid cases were correctly refused. All nine completed audits passed
their independent fixture numeric checks. Audit-level numeric accuracy was
90%, including the failed case in its denominator. This does not establish
100% accuracy across the suite.

Executable tool-selection accuracy was 0% under the strict no-invalid-requests
criterion: the model repeatedly supplied a single `test_name` with all four
rates, which the contract rejects before recovering with a valid batch request.
Several plan proposals also used invented shorthand structures. Correct
final numbers do not erase these request errors or prove correct model proposals.
Follow-up protocol/schema descriptions give canonical batch examples, explicitly
omit `test_name` for batches, and describe the full plan proposal shape.

The failed Female slice's first draft contained an unsupported rounded bound
`p > 0.37`. Its rewrite removed that bound but was incorrectly rejected for
explaining suppression as `due to having fewer than 50 total samples or fewer
than 10 positive outcomes`. The controlled causal-language rule now recognizes
this threshold wording only in a suppression/unavailability context. Regression
tests retain rejection of assertions that thresholds cause demographic gaps,
and of discrimination-cause assertions.

Offline replay of the successful computation records followed by validation of
that final saved draft passed; deterministic summary grounding also passed,
with every non-summary field unchanged. This is local revalidation, not a new
completed live audit. The historical failed trace remains unchanged. A local
attempt `2e145bc8b1d9416db5ff8fbe037ec381` could not verify the revised agent
because the provider connection failed before a response.

The initial runner counted claim-rejected structured drafts as schema errors.
The schema counter now counts actual JSON/schema failures only; claims failures
remain recorded separately in verdicts/rewrite counts. Earlier scorecard files
retain their original values. Successful saved audits can still contain failed
tool requests, and live numeric/selection gates must be re-established on a
fresh full batch with the updated protocol. Human narrative review remains pending.

Follow-up local verification: full suite 1,139 passed in 85.20 seconds; Ruff clean.

## Second full live batch: constrained summary drafting

Run `50d2a9231d2e45d4bf3501c543d5a75c` completed five of ten valid
audits and correctly refused both invalid cases. Every completed case had
correct tool requests and passed fixture-backed numeric checks. Five failed
at claims validation after completing computation: selection_gap, tpr_gap,
ppv_gap, tiny_group, duplicated_rows. Their records remain failed.

One common parser defect treated prose such as `two groups: "A" and "B"`
as a reference to a nonexistent attribute named `groups`. This now leaves
those quoted labels for the existing exact-name/ambiguity checks. A real mapped
column named `groups` still takes precedence; unknown labels, ambiguous labels,
and unknown attribute names remain rejected.

Other failures came from free-form summaries adding unsupported bounds or
causal wording. The quantitative summary was already rendered deterministically
after successful full-draft validation. Its pure renderer is now also used
before drafting: the per-run wire schema supplies the complete canonical
summary as its sole allowed string value, and drafting instructions require an
exact copy. Thus generation is constrained before the claims check; rejected
drafts are not silently replaced or reclassified as successes. The final full
draft must still pass the existing validator before postprocessing. Qualitative
limitations remain model-generated, screened, and subject to human review.

Offline checks of the five saved final drafts found three now pass unchanged
after the parser correction. Two still fail on their original summary wording.
A separate prospective drafting check supplied the canonical summary for each
case and found all five full objects pass with every non-summary field unchanged.
This prospective check simulates the new drafting constraint and is not a live
recovery or an approval of the original failed text. Original artifacts were
not modified. Tests also verify that canonical summaries do not rescue wrong
structured counts or unsupported numbers in limitations. A fresh full live
batch remains necessary to measure the updated agent.

Constrained-summary follow-up verification: full local suite 1,150 passed in
84.79 seconds; Ruff clean. No additional paid runs were made for this follow-up.

## Provider-schema rejection and bounded compatibility probe

Run `aa65e743aeca416b9c0d4d76a1acdc12` failed all ten valid cases on
their structured findings request with `BadRequestError`; both invalid inputs
were correctly refused. The saved no_gap trace completed all six computation
tools before the failing final request. Its original artifacts remain unchanged.
The trace contains the exception type but not the API response body, so the
precise provider validation complaint is not established by that trace alone.

The runtime wire schema now uses a short fixed summary placeholder with only
`type` and `enum`, instead of embedding a lengthy, varying report string along
with string constraints. The full quantitative summary is still rendered from
trusted evidence locally, after the complete model draft passes validation.
Numerical mismatches, unsupported limitations, and rejected model summaries
continue to fail validation. The response still carries the normal per-run
attribute and warning constraints.

Before any further full live suite, check the exact runtime final schema with:

```powershell
python -m evals.probe
```

This prepares no_gap evidence entirely locally, then sends one API request with
the actual final schema, `store=False`, retries disabled, and output capped at
16 tokens. It does not run the model's tool loop or claim to complete an audit.
`incomplete` is an expected response status at the cap and still establishes
schema acceptance if the request was accepted and started. On a BadRequest the
probe prints the provider's diagnostic message with credentials redacted.
Acceptance does not establish numeric reliability, narrative quality, or a
passed evaluation benchmark. Only after acceptance should a full live suite be
run. The long-summary regression is not retroactively marked successful.

Probe/transport follow-up verification: 1,154 local tests passed in 94.41 seconds;
the 103 targeted provider/agent/summary tests also passed. Ruff clean. Runtime
API acceptance of the revised schema is still pending the compatibility probe.

## Verified successful full live batch: November 16, 2026

The user reported the compatibility probe accepted by the API (incomplete output
at the intentional 16-token cap), then ran the full suite. Scorecard and all ten
saved traces in `outputs/evaluations/5bb6b88b848640f7b4e106a61d820389`
were inspected locally.

| Measure | Result |
|---|---:|
| Valid audits completed | 10/10 |
| Invalid inputs correctly refused before provider access | 2/2 |
| Executable tool-selection accuracy | 100% |
| Fixture-backed numeric accuracy | 100% (176/176 checks across ten cases) |
| Final schema validity | 100% |
| Suppression correctness | 100% |
| Automated final claims validation | 100% |
| Tool execution/request errors | 0 |
| Schema retries and claim rewrites | 0 |
| Mean valid-audit latency | 17.1222 seconds |
| Valid-audit latency range | 15.587-24.211 seconds |

Every trace completed and its grounded-summary verdict passed. Two severity
corrections (tpr_gap and diabetes_male) were retained transparently; these are
policy corrections, not evidence of uncorrected model interpretation accuracy.
All plans were delegated to the deterministic planner; no model plan proposals
were submitted, so independent model proposal accuracy remains unavailable.

The numeric accuracy scope remains fixture counts, base rates, four group rates,
three headline gaps, and dataset shape. CIs, AUROC, test statistics/p-values,
and severity passed tool-evidence validation rather than an independent fixture
oracle. No unrestricted unsupported-claim rate is certified. Human narrative
review and dollar-cost configuration remain pending; the saved benchmark gate
therefore remains open. This is one twelve-case live batch, not a general
reliability guarantee. Earlier failed batches remain unchanged.
