# General management report template verification - November 19, 2026

## Change and scope

New report renders use task-neutral English wording based on validated evidence
and explicitly supplied task/model metadata. Both saved-run rendering and new
`audit --pdf` reports use this presentation. Binary classification only; regression
and multiclass task metadata are rejected. No training or live provider calls
were needed to implement or verify this change.

The reviewed November 17 readmission report remains byte-for-byte unchanged
(SHA-256 `35a4bae8d87643eefcbf600f4a958720c807933b87da79a568d44d85b66a5931`).
Its approval remains pinned to its source audit hashes and cannot approve new
tasks, data or renders. New reports have human review pending.

## Verification

- Exercised non-healthcare task wording for loan-default and equipment-failure
  examples. Domain words are not inferred from rates or column names.
- Checked missing metadata/scores, all-suppressed groups, undefined PPV/FPR,
  single-class outcomes, chart pagination above eight groups, escaped metadata,
  and rejection of unsupported task types before provider setup.
- Confirmed input findings/evidence are unchanged by report presentation.
- Generated and rasterized the synthetic equipment example (10 pages, about 0.55
  seconds); inspected overall layout and updated metric/provenance pages.
- Ruff and `git diff --check` passed. All 57 reporting/CLI tests passed.
- Full-suite attempt: one integration test failed because Windows application
  control blocked SQLAlchemy's `_cache_key_cy` compiled extension. The failure
  occurs while importing SQLAlchemy in the existing MLflow integration test.
  The rerun excluding that one test passed 1,207 tests (one deselected).
  This is not a clean all-tests result or a new coverage measurement.

## Remaining limitations

Calibration evidence/plots, patient/entity-cluster uncertainty, bilingual PDF
typography and independent clinical validation are not supplied by this template.
Supplied task names, positive-class meanings and model metadata are not
independently verified. New reports require their own editorial review.
