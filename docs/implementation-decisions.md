# Implementation Decisions

This records the current implementation relative to the project brief. A working
alternative is not evidence that the brief's complete product requirements have
been delivered or that an external reviewer has approved the change.

| Area | Brief | Current implementation | Follow-up |
|---|---|---|---|
| Python | 3.11 | Existing project environment and CI use Python 3.12; local version 3.12.10 | Keep setup instructions and CI consistent; report the version difference at review |
| LLM provider | Gemini function calling, configurable provider adapter and model | OpenAI connection and function-calling smoke examples; model name comes from OPENAI_MODEL | Implement the provider adapter and full guarded agent later; document or review the provider choice before the agent gate |
| Data loader | data/loader.py | Shared loader and target helper in app/data.py, used by the notebook and tests | Retain the tested implementation; update imports if packaging changes |
| Model tracking | model-under-audit experiment | Existing readmission-baseline experiment | Retain run provenance; standardize naming before audit and evaluation experiments are added |
| Model artifacts | Logged model and Model Registry | Native model/config artifacts plus registered model healthequity-readmission-baseline, verified Version 1 | Apply saved preprocessing before prediction; registry registration does not execute the configuration |
| Training entry point | Retraining in one command | Ordered exploration and development notebooks | A dedicated training command is still pending; notebooks are the current reproduction path |
| Fairness engine | Deterministic metrics, tests, intervals, suppression, and severity | Implemented in fairness/metrics.py, stats.py, and policy.py; thresholds in config.yaml | Compose the helpers into the guarded audit workflow; apply eligibility before every comparison |
| Synthetic scenarios | Planted selection-rate, TPR, and PPV gaps | Exact confusion-count generator and seeded two-group scenarios; gaps 0 to 0.50 in steps of 0.01 | Use these building blocks for the Week 4 evaluation harness; the full 12-scenario agent suite is pending |
| Significance reporting | Statistical backing for each headline metric | Tests compare selection, TPR, FPR, and PPV rates; no combined EOD p-value or multiple-testing adjustment | Keep component-test results explicit; severity must not be presented as a significance result |
| Uncertainty | Percentile bootstrap 95% intervals, 1,000 resamples, fixed seed | Within-group confusion-count resampling equivalent to row bootstrap for the three gaps | Report valid resamples and the independent-stay assumption; patient-cluster uncertainty is not implemented |

## Current evidence (reviewed November 5, 2026)

- The local suite passed 180 tests after the count-validation follow-up.
  Measuring both app and fairness produced 98% combined line coverage:
  app 100%, metrics 97%, policy 95%,
  scenarios 100%, and stats 98%. Live-API smoke scripts are excluded. These
  figures do not establish notebook-training coverage or complete agent behavior.
- Ruff and pip check passed. GitHub CI run 17 succeeded for commit 5092e04;
  the initial project-setup pull request also had a successful CI run.
- CI, pre-commit, and .coveragerc now target both app and fairness. The updated
  settings passed a local coverage run; their first GitHub CI run is pending
  commit and push. Run 17 above predates this configuration change.
- Local pre-commit hooks run Ruff and the tests; they require the project
  environment to be active and must be installed per clone.
- Model and threshold selection used Validation. Test was evaluated only after
  fixing the selected baseline and threshold; results and artifacts are described
  in the Model Card. Do not tune the baseline using its Test results.
- The standing demo audit CSV contains Test outcomes, scores, predictions, and
  group columns, with a companion provenance JSON and no patient/encounter IDs.
- The review reconstructed the saved patient partitions from the source data:
  69,944 Train, 15,172 Validation, and 14,998 Test stays. Patient sets were
  disjoint and the partitions covered all 100,114 cohort stays with unique
  encounter IDs.
- Reloaded native and registered baseline models reproduced the logged
  Validation/Test metrics. The demo CSV matched Test outcomes, predictions,
  scores, group columns, and provenance metadata. This verification did not
  retrain the models or create new MLflow runs.
- Probability calibration was assessed descriptively on Validation only. Test
  and subgroup calibration have not been assessed; patient-cluster uncertainty
  remains unimplemented.
- Per-group metrics, gaps, seeded bootstrap intervals, severity, and rate tests
  were exercised together on race, gender, age, and payer_code. Eligible/suppressed
  group counts were respectively 6/0, 2/1, 8/2, and 10/7. This was a review of
  the Python helpers, not a saved end-to-end agent audit or PDF report.
- All 153 supported combinations of planted metric and gap were checked against
  their requested values. Hand-calculated fixtures, Fairlearn comparisons,
  suppression boundaries, and bootstrap reproducibility are covered by tests.
- docs/methodology.md documents the formulas, assumptions, configurable policies,
  unavailable results, and interpretation limits.
- Direct check_group_eligibility calls now reject fractional, text, and Boolean
  counts while accepting Python and NumPy integers. Nine regression cases cover
  this follow-up, bringing the suite from 171 to 180 tests.

## Follow-up from the quality review

- Verify GitHub CI after committing and pushing the count-validation and combined
  coverage changes. Both corrections are implemented and verified locally.
- Keep severity and significance separate. For example, the demo payer_code PPV
  gap meets the configured substantial rule while its multi-group PPV equality
  test has p approximately 0.781. These outputs answer different questions.

## Remaining product work

The guarded agent, reusable audit orchestration and structured tool wrappers,
claims validator, full 12-scenario agent evaluation harness, PDF reporting, full
UI and upload flow, deployment, monitoring, white-label configuration, and CLI
remain later deliverables. Existing notebook training is not a completed
session-based upload training path. Deterministic Week 2 helper functions and
the scenario generator are implemented; they are inputs to these later stages.

Weekly communications, reading notes, hours logs, and public progress posts are
separate brief obligations; their completion cannot be inferred from unit tests
or model artifacts.
