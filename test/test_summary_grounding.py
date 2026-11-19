"""The final summary must use trusted quantities/policy, not model interpretations."""

from copy import deepcopy

import pandas as pd
import pytest
import test_agent as helpers

from app.agent import AuditAgent
from app.agent_errors import AgentRunError
from app.audit_schemas import AuditFindings
from app.claims_validation import validate_claims
from app.llm_config import LLMConfig
from app.schemas import ColumnMapping
from app.summary_grounding import SUMMARY_PLACEHOLDER, ground_summary, render_summary
from app.tools import AuditToolSession
from evals.offline_provider import OfflineProvider
from evals.scenarios import build_scenarios
from fairness.policy import SuppressionPolicy


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


def test_renderer_produces_same_summary_without_a_model_draft(grounding_case):
    findings, evidence = grounding_case
    grounded, _ = ground_summary(findings, evidence)
    assert render_summary(evidence) == grounded.summary
    assert grounded.summary.startswith("This audit covers 200 rows and 1 mapped attribute.")
    assert validate_claims(findings.model_copy(update={"summary": render_summary(evidence)}), evidence).passed


def test_agent_supplies_canonical_summary_as_only_wire_schema_choice():
    # The scripted provider intentionally ignores wire constraints. Existing
    # claims-validation checks must still apply to such injected responses.
    frame = pd.DataFrame({"actual": [0, 0, 1, 1] * 2, "predicted": [0, 1, 0, 1] * 2,
                          "group": ["A"] * 4 + ["B"] * 4})
    session = AuditToolSession(frame, ColumnMapping(outcome="actual", prediction="predicted", groups=["group"]))
    provider = helpers.ScriptedProvider(session)
    result = helpers.agent_for(session, provider).run()
    final_requests = [r for r in provider.requests if r["response_schema"] is not None]
    expected = render_summary(session.get_evidence())
    assert all(r["response_schema"]["properties"]["summary"] == {
        "type": "string", "enum": [SUMMARY_PLACEHOLDER],
    }
               for r in final_requests)
    assert result.findings.summary == expected


@pytest.mark.parametrize("defect", ["unsupported_limitation", "wrong_count"])
def test_canonical_summary_does_not_rescue_invalid_other_fields(defect):
    frame = pd.DataFrame({"actual": [0, 0, 1, 1] * 2, "predicted": [0, 1, 0, 1] * 2,
                          "group": ["A"] * 4 + ["B"] * 4})
    session = AuditToolSession(
        frame, ColumnMapping(outcome="actual", prediction="predicted", groups=["group"]),
        suppression_policy=SuppressionPolicy(min_group_size=2, min_positive_outcomes=1),
    )

    class DefectiveProvider(helpers.ScriptedProvider):
        def generate(self, **kwargs):
            response = super().generate(**kwargs)
            if kwargs["response_schema"] is not None:
                response.structured_output["summary"] = kwargs["response_schema"]["properties"]["summary"]["enum"][0]
                if defect == "unsupported_limitation":
                    response.structured_output["limitations"] = ["There are 99999999 rows."]
                else:
                    response.structured_output["attributes"][0]["groups"][0]["count"] -= 1
            return response

    with pytest.raises(AgentRunError) as failure:
        helpers.agent_for(session, DefectiveProvider(session)).run()
    verdict = failure.value.trace.claims_validation
    assert not verdict.passed
    expected_code = "unsupported_number" if defect == "unsupported_limitation" else "evidence_mismatch"
    assert any(issue.code == expected_code for issue in verdict.issues)


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


@pytest.mark.parametrize(("scenario", "draft_text", "expected_text"), [
    (
        "tiny_group",
        "Confidence intervals for fairness metrics include zero.",
        "Some reported percentile intervals have positive lower bounds even when the observed gap is zero.",
    ),
    (
        "multiple_groups",
        'The rate tests show statistically significant disparities for group "C".',
        "Identifying a statistically significant pair would require a post-hoc comparison.",
    ),
])
def test_agent_renders_limitations_from_evidence(scenario, draft_text, expected_text):
    case = next(case for case in build_scenarios() if case.name == scenario)
    session = AuditToolSession(
        case.data, case.mapping,
        suppression_policy=SuppressionPolicy(**case.expected["policy"]),
    )

    class ProviderWithLegacyProse(OfflineProvider):
        def generate(self, **kwargs):
            response = super().generate(**kwargs)
            if response.structured_output is not None:
                response.structured_output["limitations"] = [draft_text]
            return response

    provider = ProviderWithLegacyProse(session)
    result = AuditAgent(
        provider=provider, session=session,
        llm_config=LLMConfig(model_name=provider.model_name),
    ).run()
    draft = AuditFindings.model_validate(result.trace.turns[-1].response.structured_output)
    assert draft.limitations == [draft_text]
    assert draft_text not in result.findings.limitations
    assert expected_text in " ".join(result.findings.limitations)
    assert result.findings.limitations == result.trace.summary_grounding.grounded_limitations
    excluded = {"summary", "limitations"}
    assert result.findings.model_dump(exclude=excluded) == draft.model_dump(exclude=excluded)
    assert result.claims_validation.passed


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
