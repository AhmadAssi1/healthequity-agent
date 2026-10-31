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

## Current evidence

- The local unit suite has 58 passing tests and 100% line coverage for the
  included app modules. Live-API smoke scripts are explicitly excluded; these
  figures do not cover model fairness, a complete agent, or all possible inputs.
- CI is configured for lint and tests with coverage. The previous pushed run was
  green; the current coverage and pre-commit changes still require a new CI run.
- Local pre-commit hooks run Ruff and the tests; they require the project
  environment to be active and must be installed per clone.
- Model and threshold selection used Validation. Test was evaluated only after
  fixing the selected baseline and threshold; results and artifacts are described
  in the Model Card. Do not tune the baseline using its Test results.
- The standing demo audit CSV contains Test outcomes, scores, predictions, and
  group columns, with a companion provenance JSON and no patient/encounter IDs.
- Probability calibration was assessed descriptively on Validation only. Test
  and subgroup calibration and uncertainty have not been assessed.

## Remaining product work

The deterministic fairness engine, statistical tests and confidence intervals,
suppression implementation, guarded agent, claims validator, scenario evaluation,
reporting, full UI and upload flow, deployment, monitoring, white-label configuration,
and CLI remain later deliverables. Existing model training is not a completed
session-based upload training path.

Weekly communications, reading notes, hours logs, and public progress posts are
separate brief obligations; their completion cannot be inferred from unit tests
or model artifacts.
