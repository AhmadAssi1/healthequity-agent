# Week 3 live verification - November 12, 2026

The five reported live runs below used the 14,998-row demo CSV. Records were read
from their saved `run.json` and `trace.json`. Failed runs remain failures in this
record; later rule changes do not retroactively turn them into successful live runs.

| CLI run directory | Original result | Seconds | Claim rewrite used |
|---|---|---:|---|
| e9cd8d0493224782bca987f302a55a07 | completed | 86.787 | yes, second verdict passed |
| 2e6a21b470064b658359863324d153fb | completed | 104.022 | yes, second verdict passed |
| e5b177516ca5447ca31b38831a48df4d | failed | 89.850 | yes, second verdict failed |
| 0adbec3cd71c40eaabcfef9c47d2a1d2 | completed | 46.541 | no, first verdict passed |
| ae0d6353caaf48218f43f1f7431af3cf | failed | 83.851 | yes, second verdict failed |

Completion was 3/5 (60%); first-draft validation passed in 1/5 (20%). Every
recorded total duration was below 180 seconds. This batch does not establish
reliable five-run completion or unrestricted narrative correctness. Automated
claims verdicts retain `requires_human_review: true`.

## Findings from failed traces

The final rejected drafts in both failed runs explained suppression using
`due to small size or low positive outcome count` or
`due to insufficient group size and/or positive outcome counts`. These are policy
explanations, not claims about why demographic disparities exist. The causal
screen incorrectly rejected these forms.

Another initial rejection treated `Certain groups` as a named category. This is
a generic determiner, like `Some groups`, rather than a specific group label.

## Follow-up changes and remaining checks

- Extend recognized policy explanations to small size, low positive counts,
  and insufficient group size.
- Require a suppression/unavailability context for that exemption, rather than
  allowing an assertion that a disparity itself is caused by small samples.
- Recognize generic determiners such as Certain, Several, and Many.
- Add regression cases using the saved failing wording and ensure discrimination
  claims remain rejected.

These changes need test confirmation and live verification. The previously
reported targeted suite passed 194 tests before this follow-up. No new passing
test count is asserted here, and none of the saved live artifacts were modified.

The full Week 4 scenario scorecard is separate. Narrative checks still follow
the controlled-language limits in [Claims Validation](claims-validation.md).

## Subsequent frozen-version batch and extra attempt

| CLI run directory | Original result | Seconds | Failure |
|---|---|---:|---|
| ea98a9af116e40ad979fd44d2261b8c8 | completed | 53.186 | - |
| fcc59fcd2ce34c7080e70bc5702a0342 | completed | 45.756 | - |
| 2a08173efa944db0a112bed3f180a1c9 | completed | 46.719 | - |
| bcec30eae9754f30886d5050ba511cc5 | failed | 71.027 | Final provider request timed out at 60 seconds |
| 660d80c6ba8147eca10a4e727e659fb7 | failed | 31.581 | Only race findings returned for four declared attributes, twice |
| c8134487521b40f197e21b8d70f471ef | failed | 97.484 | Unquoted group labels, then an ambiguous shared empty label |

The first five-run batch completed 3/5; including the additional sixth attempt,
completion was 3/6 (50%). The earlier 227 targeted test passes do not establish
live robustness. This batch does not close the Week 3 reliability gate.

The wire schema now binds the attributes array length and allowed attribute
names to the complete evidence for the current run. Declared group columns have
the same size and vocabulary constraints. Local validation still rejects
duplicates and set mismatches. Schema-repair feedback prints the exact required
and returned names. In addition, explicitly scoped same-sentence group references
can disambiguate a shared label, and unresolved-label feedback identifies the
actual label and its possible attributes.

These follow-up changes require test confirmation and new live evidence. The
timeout remains recorded separately; these schema/parser changes do not prove
that provider latency has been resolved. Saved review is offline revalidation,
not recovery of a completed live audit. An incomplete saved findings object
cannot pass review merely because the live wire schema has since improved.

## Latest targeted test and saved-review evidence

The user subsequently reported 237 passing tests across narrative validation,
agent orchestration, CLI, and saved review. This is a targeted suite, not a new
full-project test or coverage result.

The latest six-output offline review passed four candidates, including the
previously ambiguous-label rejection. The provider-timeout run had no structured
candidate, and the race-only findings object remained invalid. Both were reported
as unavailable, not successes.

Additional run `5b6740a0a18e48e18bdbc59aef52b5bb` completed in 48.523 seconds,
with eight model turns, six tool calls, and a passing first claims verdict. It
started at 08:56:35 UTC; per-run schema-binding changes were written after that
start and the attribute-context parser fix was later still. It therefore cannot
establish a live pass on the complete latest implementation. Its original success
and passing offline recheck remain useful historical evidence. The latest
frozen-version live gate is still pending.

## Five-run batch after the full local suite

The user reported Ruff passing and 1,058 tests passing in 11.62 seconds, then ran
five consecutive audits without intervening code edits:

| CLI run directory | Original result | Seconds |
|---|---|---:|
| 6182733c958c44ef86d9b777f20f33ea | failed | 101.613 |
| 3db7daa05ea247d1a0b97f201bad204b | failed | 84.965 |
| 882a82b915c141aabd08a2c8f0a172d6 | completed | 49.649 |
| afa5085614be4467af8921bd68f77efc | completed | 54.578 |
| 90a313c651004d42baca37d3b999b1c3 | failed | 96.285 |

Completion was 2/5 (40%). All five durations were below 180 seconds. The three
failed runs ended in claims rejection, not a demonstrated reliable completion.
This gate remains open; no runs are excluded from the reported rate.

Trace review exposed a parser category error: numeric `p=...` and `CI: ...`
annotations were treated as references to nonexistent group attributes. Written
counts such as `Six groups` were also treated as category names. Further issues
included an omitted required quality warning, unquoted known category labels,
and narrative comparison bounds not directly supplied by evidence.

Follow-up code distinguishes numeric statistical annotations while retaining
their numeric validation; derives counts from visible evidence lists; normalizes
limited written count phrases; and accepts exact, unambiguous known bare labels.
Unknown or ambiguous names still fail. The per-run warning schema now binds to
the trusted quality-flag vocabulary and count. Unsupported-number feedback names
the token, and drafting instructions prefer evidence values over new bounds.

These changes need fresh test confirmation and offline replay before further
paid evaluation. Neither successful unit tests nor offline rechecks retroactively
alter the recorded 40% live completion rate.

## Latest completed five-run batch

After the user confirmed 1,074 tests passing and Ruff clean, the following five
consecutive runs completed without intervening implementation changes:

| CLI run directory | Original result | Seconds | Claims verdicts |
|---|---|---:|---|
| 466e9a5fdb0d410682cadfd71086604e | completed | 51.365 | passed first attempt |
| f9b75ca54e2c48c2ab6922e156a516a0 | completed | 45.830 | passed first attempt |
| fb71088232124064b9cb2359391f67cf | completed | 48.690 | passed first attempt |
| e718e663ba05403691aea2352da006f3 | completed | 44.354 | passed first attempt |
| 714405af53d549fba5a0a105c13a966d | completed | 66.574 | passed first attempt |

Completion was 5/5 (100%) for this small demo batch. Mean total latency was
51.3626 seconds; the range was 44.354-66.574 seconds, below 180 seconds in every
case. Each trace contains eight model turns, six computation-tool executions,
and one passing claims verdict. No schema or claim rewrite was needed.

All four mapped attributes are present. Comparing the 290 numeric fields in
each final findings object against the first run found identical field sets
and no differences outside relative tolerance 1e-9 / absolute tolerance 1e-12.
There were zero to two exact floating-point differences per run, so the objects
must not be described as byte-for-byte or exactly numerically identical.
The successful stored claims verdicts also validate structured quantities against
each run's trusted evidence within the implementation's tolerances.

Each run corrected race PPV severity from the model's substantial label to the
policy's watch label. The original model response and the correction are retained.
Historical failed batches above remain part of the record.

### Interpretation defect observed before summary grounding

Direct inspection of the final summaries found a contradiction not caught by
the rule-based validator. For example, run f9b75ca54e2c48c2ab6922e156a516a0 calls
the race PPV difference (0.09732620320855616) substantial in its summary even
though the final structured finding correctly says watch. Some other summaries
also generalize severity/significance too broadly. Automated checks passing
does not establish zero unsupported narrative claims.

The demo runtime/repeated-completion evidence is now available, but unrestricted
narrative grounding remains open. Human review is still required, and the Week 3
gate must not be presented as fully satisfying every narrative requirement until
these interpretation inconsistencies are addressed. No additional paid runs are
needed merely to rediscover this known defect. A suitable follow-up is to make
trusted severity and significance conclusions explicit in drafting evidence and
verify/render quantitative narrative claims from structured facts.

## Deterministic summary follow-up

The runtime batch above is retained unchanged. A subsequent implementation adds
`summary_grounding`: after a full draft passes validation, the quantitative
summary is rendered from trusted evidence, then validated again. The original
model summary, replacement, and validation are retained in the trace. The report
receipt is refreshed for the final findings. Numeric fields are not edited.

This stage is local and introduces no provider request. It is intended to remove
the observed race-PPV severity contradiction and overly broad significance prose
from the final summary. It does not certify unrestricted text in limitations.

### Confirmed follow-up verification

The user subsequently reported the full local suite passing: **1,083 tests in
23.67 seconds**. Ruff was also reported clean. These are recorded results from
the user's terminal, not a new test execution performed while updating this log.

The read-only command
`python -m healthequity.review_saved --last 5 --ground-summary` returned
`recheck: passed`, an empty issues list, and `summary_grounded: true` for each
of the five completed runs listed above. The user also displayed the grounded
summary for run `714405af53d549fba5a0a105c13a966d`; its race PPV difference
retained the policy severity `watch`.

This confirms offline verification of the new summary postprocessing on all
five saved candidates. It does not represent five additional live runs or
change their historical artifacts. The recorded live runtime evidence and the
subsequent offline grounding evidence remain separate.

Free-text limitations and further interpretations still require human review.
The scoped review below records the user's review of one saved report only.

### Human review of one report's limitations - November 12, 2026

Ahmad Assi reviewed the three limitations in saved run
`714405af53d549fba5a0a105c13a966d`, discussed their meaning, and explicitly
approved them with a clarification to the third limitation:

- Associations and statistical tests do not establish causal relationships.
- Suppressed groups have insufficient evidence; exclusion does not demonstrate
  fairness for those populations.
- Result quality depends on the accuracy of outcomes, predictions, and group
  labels. Metrics that use probability scores, such as AUROC, also depend on
  score accuracy. This clarifies the original score/outcome-only wording.

The saved report contains no separate recommendations section. Its historical
findings and trace were not modified. This review covers those limitations and
the approved clarification; it does not approve the original model summary or
certify every report, clinical suitability, or legal compliance. Future free-text
findings still require review.
