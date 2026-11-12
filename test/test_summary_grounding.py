"""The final summary must use trusted quantities/policy, not model interpretations."""

from copy import deepcopy

import pandas as pd
import pytest
import test_agent as helpers

from app.audit_schemas import AuditFindings
from app.claims_validation import validate_claims
from app.schemas import ColumnMapping
from app.summary_grounding import ground_summary
from app.tools import AuditToolSession


@pytest.fixture
def grounding_case():
    rows = []
    for label, counts in (("A", (30, 20, 30, 20)), ("B", (20, 30, 10, 40))):
        for (actual, predicted), count in zip(((0, 0), (0, 1), (1, 0), (1, 1)), counts, strict=True):
            rows.extend([{"actual": actual, "predicted": predicted, "group": label}] * count)
    session = AuditToolSession(pd.DataFrame(rows), ColumnMapping(outcome="actual", prediction="predicted", groups=["group"]))
    for stage in helpers.STAGES:
        session.invoke(stage, {"n_resamples": 29} if stage in ("bootstrap_ci", "run_significance_test") else {})
    findings = AuditFindings.model_validate(helpers.findings_from_evidence(session))
    return findings, session.get_evidence()


def test_summary_corrects_live_style_ppv_severity_contradiction(grounding_case):
    findings, evidence = grounding_case
    findings.summary = "All three gaps are substantial."
    assert validate_claims(findings, evidence).passed  # Known limitation of prose screening.
    grounded, record = ground_summary(findings, evidence)
    assert "PPV difference is 0.0714285714285714; policy severity is watch." in grounded.summary
    assert record.proposed_summary == "All three gaps are substantial."
    assert record.grounded_summary == grounded.summary and record.validation.passed
    assert record.source == "trusted_tool_evidence"
    assert grounded.model_dump(exclude={"summary"}) == findings.model_dump(exclude={"summary"})


def test_summary_grounding_does_not_mutate_evidence_or_draft(grounding_case):
    findings, evidence = grounding_case
    before = deepcopy((findings.model_dump(), evidence.model_dump()))
    ground_summary(findings, evidence)
    assert (findings.model_dump(), evidence.model_dump()) == before


@pytest.mark.parametrize("problem", ["unsupported_summary_number", "incorrect_group_count"])
def test_grounding_does_not_manufacture_approval_for_failed_drafts(grounding_case, problem):
    findings, evidence = grounding_case
    if problem == "unsupported_summary_number":
        findings.summary = "This audit contains 99999999 rows."
    else:
        findings.attributes[0].groups[0].count -= 1
    with pytest.raises(ValueError, match="must pass claim validation"):
        ground_summary(findings, evidence)


def test_summary_copies_test_results_without_inventing_significance(grounding_case):
    findings, evidence = grounding_case
    grounded, record = ground_summary(findings, evidence)
    assert record.validation.passed
    for test in evidence.significance.results:
        assert f"with p={test.p_value!r}" in grounded.summary
    assert "all rates are significant" not in grounded.summary
    assert "No combined equalized-odds p-value is reported." in grounded.summary


def test_summary_records_unavailable_results_and_suppression_explicitly():
    frame = pd.DataFrame({"actual": [0, 1], "predicted": [0, 1], "group": ["A", "B"]})
    session = AuditToolSession(frame, ColumnMapping(outcome="actual", prediction="predicted", groups=["group"]))
    result = helpers.agent_for(session).run()
    assert "0 reportable groups" in result.findings.summary
    assert "2 suppressed groups have insufficient evidence" in result.findings.summary
    assert "PPV difference is unavailable; policy severity is unavailable." in result.findings.summary
    assert "comparison test is unavailable" in result.findings.summary
    assert result.trace.summary_grounding.validation.passed
    assert result.trace.summary_grounding.proposed_summary == "Observed associations require review."


def test_agent_preserves_raw_model_summary_and_publishes_grounded_summary():
    # Provider calls are scripted; no network client is instantiated.
    frame = pd.DataFrame({"actual": [0, 0, 1, 1] * 2, "predicted": [0, 1, 0, 1] * 2,
                          "group": ["A"] * 4 + ["B"] * 4})
    session = AuditToolSession(frame, ColumnMapping(outcome="actual", prediction="predicted", groups=["group"]))
    result = helpers.agent_for(session).run()
    raw = result.trace.turns[-1].response.structured_output["summary"]
    assert raw == result.trace.summary_grounding.proposed_summary
    assert result.findings.summary == result.trace.summary_grounding.grounded_summary
    assert result.findings.summary != raw
    assert len(result.trace.turns) == 8
    assert len(helpers.executed(result.trace)) == 6
    assert result.claims_validation == result.trace.summary_grounding.validation
