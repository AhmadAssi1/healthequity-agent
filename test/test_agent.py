"""Offline orchestration tests with real tools and a scripted model provider."""

import json
from copy import deepcopy

import pandas as pd
import pytest

from app.agent import AUDIT_TOOL_NAMES, AuditAgent
from app.agent_config import AgentConfig
from app.agent_errors import AgentRunError
from app.agent_schemas import AgentRunTrace
from app.audit_schemas import (
    AttributeFindings,
    AuditFindings,
    BootstrapConfidenceInterval,
    DatasetSummary,
    FairnessFinding,
    StatisticalTestResult,
    SuppressedGroup,
)
from app.claims_validation import numbers_match, validate_claims
from app.findings_policy import apply_severity_policy
from app.llm_config import LLMConfig
from app.llm_provider import StructuredOutputError
from app.llm_schemas import LLMMessage, LLMResponse, TokenUsage, ToolCall
from app.schemas import ColumnMapping
from app.tools import AuditToolSession
from fairness.policy import SuppressionPolicy, classify_severity

STAGES = [
    "profile_dataset", "check_group_sizes", "select_analyses",
    "compute_fairness_metrics", "run_significance_test", "bootstrap_ci",
]


def response(*, calls=None, text=None, structured=None):
    return LLMResponse(
        tool_calls=calls or [], text=text, structured_output=structured,
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        finish_reason="completed", latency_ms=1.0,
    )


def tool_call(name, call_id, arguments=None):
    arguments = arguments if arguments is not None else (
        {"n_resamples": 29} if name in ("bootstrap_ci", "run_significance_test") else {}
    )
    return ToolCall(name=name, call_id=call_id, arguments=arguments)


def findings_from_evidence(session):
    """Test-only deterministic drafting; claims are deliberately not approved."""
    evidence = session.get_evidence()
    bootstrap = {(item.attribute, item.metric): item for item in evidence.bootstrap.results}
    attributes = []
    for item in evidence.metrics.results:
        findings = []
        for gap in item.gaps:
            boot = bootstrap[item.attribute, gap.metric]
            ci = None if boot.lower is None else BootstrapConfidenceInterval(
                lower=boot.lower, upper=boot.upper,
                valid_resamples=boot.valid_resamples, total_resamples=boot.total_resamples,
            )
            findings.append(FairnessFinding(
                metric=gap.metric, value=gap.value, reason=gap.reason,
                confidence_interval=ci, confidence_interval_reason=boot.reason,
                severity=classify_severity(gap.value, boot.lower, boot.upper, evidence.severity_policy),
            ))
        attributes.append(AttributeFindings(
            attribute=item.attribute, groups=item.groups, fairness_findings=findings,
            suppressed_groups=[SuppressedGroup(
                group_value=s.group_value,
                suppression_reason="Insufficient evidence: " + ", ".join(s.reason_codes),
            ) for s in item.suppressed_groups],
            statistical_tests=[StatisticalTestResult(**test.model_dump(exclude={"attribute"}))
                               for test in evidence.significance.results
                               if test.attribute == item.attribute],
        ))
    return AuditFindings(
        dataset=DatasetSummary(
            row_count=evidence.profile.row_count, column_count=evidence.profile.column_count,
            group_columns=[item.attribute for item in attributes], score_available=False,
        ),
        attributes=attributes, summary="Observed associations require review.",
        warnings=evidence.profile.data_quality_flags,
        limitations=["Rows are treated as independent observations."],
    ).model_dump(mode="json")


class ScriptedProvider:
    model_name = "test-model"

    def __init__(self, session, turns=None, finals=None):
        self.session = session
        self.turns = turns if turns is not None else [
            response(calls=[tool_call(name, f"call_{i}")]) for i, name in enumerate(STAGES)
        ] + [response(text="READY_FOR_FINDINGS")]
        self.finals = list(finals or [])
        self.requests = []
        self.final_count = 0

    def generate(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if kwargs["response_schema"] is not None:
            self.final_count += 1
            action = self.finals.pop(0) if self.finals else findings_from_evidence(self.session)
            if isinstance(action, Exception):
                raise action
            return response(text=json.dumps(action), structured=action)
        action = self.turns.pop(0) if self.turns else response(text="READY_FOR_FINDINGS")
        if isinstance(action, Exception):
            raise action
        return action


@pytest.fixture
def session():
    df = pd.DataFrame({
        "actual": [0, 0, 1, 1] * 2, "predicted": [0, 1, 0, 1] * 2,
        "group": ["A"] * 4 + ["B"] * 4,
    })
    return AuditToolSession(
        df, ColumnMapping(outcome="actual", prediction="predicted", groups=["group"]),
        SuppressionPolicy(min_group_size=2, min_positive_outcomes=1),
    )


def agent_for(session, provider=None, config=None):
    return AuditAgent(
        provider=provider or ScriptedProvider(session), session=session,
        llm_config=LLMConfig(model_name="test-model"), config=config,
    )


def executed(trace):
    return [item for turn in trace.turns for item in turn.tool_executions]


def test_full_six_tool_run_returns_structural_findings_and_complete_trace(session):
    provider = ScriptedProvider(session)
    result = agent_for(session, provider).run()
    assert result.trace.status == "completed"
    assert result.trace.finished_at >= result.trace.started_at
    assert result.claims_validation.passed is True
    assert result.trace.claims_validation == result.claims_validation
    assert [record.call.name for record in executed(result.trace)] == STAGES
    assert len(result.trace.turns) == 8
    assert all(turn.input_messages for turn in result.trace.turns)
    assert all(turn.response.usage.total_tokens == 15 for turn in result.trace.turns)
    assert result.trace.turns[-1].structured_response_requested is True
    assert provider.requests[-1]["tools"] == []
    assert {t.name for t in provider.requests[0]["tools"]} == AUDIT_TOOL_NAMES
    assert AgentRunTrace.model_validate_json(result.trace.model_dump_json()) == result.trace


def test_agent_instance_is_single_use(session):
    agent = agent_for(session)
    agent.run()
    with pytest.raises(RuntimeError, match="new agent"):
        agent.run()


@pytest.fixture
def claims_case(session):
    for stage in STAGES:
        session.invoke(stage, {"n_resamples": 29} if stage in ("bootstrap_ci", "run_significance_test") else {})
    return AuditFindings.model_validate(findings_from_evidence(session)), session.get_evidence()


@pytest.mark.parametrize(("claimed", "trusted", "expected"), [
    (None, None, True), (None, 0.0, False), (0.0, None, False),
    (0.2, 0.19999999999999996, True), (0.2, 0.21, False),
])
def test_claim_number_matching_preserves_unavailability(claimed, trusted, expected):
    assert numbers_match(claimed, trusted) is expected


def test_claims_accept_complete_evidence_without_mutation(claims_case):
    findings, evidence = claims_case
    originals = findings.model_dump(), evidence.model_dump()
    assert validate_claims(findings, evidence).passed
    assert (findings.model_dump(), evidence.model_dump()) == originals


@pytest.mark.parametrize("mutation", [
    "rows", "columns", "score", "count", "rate", "group", "missing_group",
    "missing_test", "method", "p_value", "statistic", "expected_count",
    "gap", "ci", "resamples", "severity", "attribute",
])
def test_claims_reject_structurally_valid_but_unsupported_values(claims_case, mutation):
    findings, evidence = claims_case
    attribute = findings.attributes[0]
    if mutation == "rows":
        findings.dataset.row_count += 1
    elif mutation == "columns":
        findings.dataset.column_count += 1
    elif mutation == "score":
        findings.dataset.score_available = True
    elif mutation == "count":
        attribute.groups[0].count -= 1
    elif mutation == "rate":
        attribute.groups[0].tpr.value = 0.25
    elif mutation == "group":
        attribute.groups[0].group_value = "Invented"
    elif mutation == "missing_group":
        # Keep two eligible groups so the schema itself does not reject comparisons.
        attribute.groups[0].group_value = True
    elif mutation == "missing_test":
        attribute.statistical_tests.pop()
    elif mutation == "method":
        attribute.statistical_tests[0].method = "chi_square"
        attribute.statistical_tests[0].statistic = 2.0
    elif mutation == "p_value":
        attribute.statistical_tests[0].p_value = 0.123
    elif mutation == "statistic":
        attribute.statistical_tests[0].statistic = 8.0
    elif mutation == "expected_count":
        attribute.statistical_tests[0].min_expected_count = 99.0
    elif mutation == "gap":
        attribute.fairness_findings[0].value = 0.4
    elif mutation == "ci":
        attribute.fairness_findings[0].confidence_interval.lower = 0.01
    elif mutation == "resamples":
        attribute.fairness_findings[0].confidence_interval.total_resamples += 1
    elif mutation == "severity":
        attribute.fairness_findings[0].severity = "watch"
    else:
        findings.dataset.group_columns = ["invented"]
        attribute.attribute = "invented"
    verdict = validate_claims(findings, evidence)
    assert not verdict.passed
    assert {issue.code for issue in verdict.issues} <= {"evidence_mismatch", "unsupported_group"}


@pytest.mark.parametrize(("text", "code"), [
    ("The audit used 9999999 rows.", "unsupported_number"),
    ("The rate was 1.2e-9.", "unsupported_number"),
    ("Race causes the disparity.", "causal_language"),
    ("The disparity is caused by race.", "causal_language"),
    ("The model is fair.", "evidence_mismatch"),
    ("Claims validation has passed.", "evidence_mismatch"),
    ("The model is legally compliant.", "evidence_mismatch"),
])
def test_narrative_screen_rejects_known_unsupported_claims(claims_case, text, code):
    findings, evidence = claims_case
    findings.summary = text
    verdict = validate_claims(findings, evidence)
    assert code in {issue.code for issue in verdict.issues}


@pytest.mark.parametrize("text", [
    "Association does not imply causation.",
    "The audit cannot prove fairness.",
    "No causal effect can be inferred.",
    "The audit used 8 rows and 3 columns.",
    "The selection rate is 50.0%.",
])
def test_narrative_screen_allows_supported_numbers_and_limitations(claims_case, text):
    findings, evidence = claims_case
    findings.summary = text
    assert validate_claims(findings, evidence).passed


def test_claims_require_suppression_and_explicit_insufficient_evidence(claims_case):
    from app.tool_schemas import SuppressedGroupSummary

    findings, evidence = claims_case
    evidence.metrics.results[0].suppressed_groups.append(
        SuppressedGroupSummary(group_value="Small", reason_codes=["below_min_group_size"]),
    )
    findings.attributes[0].suppressed_groups.append(
        SuppressedGroup(group_value="Small", suppression_reason="Insufficient evidence."),
    )
    assert validate_claims(findings, evidence).passed
    findings.summary = "No groups were suppressed."
    assert not validate_claims(findings, evidence).passed
    findings.summary = "Some groups have insufficient evidence."
    findings.attributes[0].suppressed_groups[0].suppression_reason = "Excluded."
    assert "missing_insufficient_evidence" in {i.code for i in validate_claims(findings, evidence).issues}
    findings.attributes[0].suppressed_groups.clear()
    assert "unsupported_group" in {i.code for i in validate_claims(findings, evidence).issues}


def test_claims_require_trusted_quality_warnings(claims_case):
    findings, evidence = claims_case
    evidence.profile.data_quality_flags.append("An internal quality concern requires review.")
    verdict = validate_claims(findings, evidence)
    assert not verdict.passed
    assert any(issue.message.startswith("warnings:") for issue in verdict.issues)


def test_claim_failure_stops_agent_and_preserves_verdict_in_failed_trace(session):
    class IncorrectSummaryProvider(ScriptedProvider):
        def generate(self, **kwargs):
            result = super().generate(**kwargs)
            if result.structured_output is not None:
                result.structured_output["summary"] = "The dataset contains 9999999 rows."
            return result

    with pytest.raises(AgentRunError, match="Claim validation failed") as caught:
        agent_for(session, IncorrectSummaryProvider(session)).run()
    trace = caught.value.trace
    assert trace.status == "failed"
    assert trace.claims_validation.passed is False
    assert trace.claims_validation.issues[0].code == "unsupported_number"
    assert AgentRunTrace.model_validate_json(trace.model_dump_json()) == trace


@pytest.mark.parametrize("all_suppressed", [False, True])
def test_claims_accept_unavailable_tests_without_inventing_expected_counts(all_suppressed):
    df = pd.DataFrame({
        "actual": [0, 0, 1, 1] * 2, "predicted": [0] * 8,
        "group": ["A"] * 4 + ["B"] * 4,
    })
    session = AuditToolSession(
        df, ColumnMapping(outcome="actual", prediction="predicted", groups=["group"]),
        SuppressionPolicy() if all_suppressed else SuppressionPolicy(min_group_size=2, min_positive_outcomes=1),
    )
    result = agent_for(session).run()
    assert result.claims_validation.passed
    assert result.claims_validation.requires_human_review
    assert any(test.min_expected_count is None for test in result.findings.attributes[0].statistical_tests)


def test_provider_model_must_match_recorded_configuration(session):
    with pytest.raises(ValueError, match="provider model"):
        AuditAgent(provider=ScriptedProvider(session), session=session, llm_config=LLMConfig(model_name="other"))


def test_configuration_is_snapshotted(session):
    config = AgentConfig(max_tool_calls=6)
    provider_config = LLMConfig(model_name="test-model")
    agent = AuditAgent(provider=ScriptedProvider(session), session=session, llm_config=provider_config, config=config)
    config.max_tool_calls = 1
    provider_config.model_name = "changed"
    result = agent.run()
    assert result.trace.agent_config.max_tool_calls == 6
    assert result.trace.llm_config.model_name == "test-model"


def test_batch_calls_can_use_exact_budget_and_still_request_final_findings(session):
    batch = response(calls=[tool_call(name, f"batch_{i}") for i, name in enumerate(STAGES)])
    provider = ScriptedProvider(session, turns=[batch, response(text="READY_FOR_FINDINGS")])
    result = agent_for(session, provider, AgentConfig(max_tool_calls=6)).run()
    assert len(executed(result.trace)) == 6
    assert len(result.trace.turns) == 3


def test_over_budget_batch_is_rejected_before_any_of_its_calls_execute(session):
    batch = response(calls=[tool_call(name, f"batch_{i}") for i, name in enumerate(STAGES)])
    provider = ScriptedProvider(session, turns=[batch])
    with pytest.raises(AgentRunError, match="remaining budget") as caught:
        agent_for(session, provider, AgentConfig(max_tool_calls=5)).run()
    assert caught.value.trace.status == "failed"
    assert executed(caught.value.trace) == []
    assert len(caught.value.trace.turns[0].response.tool_calls) == 6


def test_repeated_call_id_is_rejected_across_turns(session):
    provider = ScriptedProvider(session, turns=[
        response(calls=[tool_call("profile_dataset", "same")]),
        response(calls=[tool_call("check_group_sizes", "same")]),
    ])
    with pytest.raises(AgentRunError, match="reused") as caught:
        agent_for(session, provider).run()
    assert len(executed(caught.value.trace)) == 1


@pytest.mark.parametrize("bad_call, code", [
    (tool_call("render_report", "bad"), "tool_request_rejected"),
    (tool_call("bootstrap_ci", "bad", {"n_resamples": -1}), "invalid_arguments"),
    (tool_call("compute_fairness_metrics", "bad", {"attributes": ["unmapped"]}), "tool_request_rejected"),
])
def test_recoverable_errors_return_to_model_and_can_be_corrected(session, bad_call, code):
    normal = ScriptedProvider(session)
    provider = ScriptedProvider(session, turns=[response(calls=[bad_call]), *normal.turns])
    result = agent_for(session, provider).run()
    first = executed(result.trace)[0]
    assert first.result.is_error and not first.fatal_error
    assert first.result.output["code"] == code
    feedback = provider.requests[1]["messages"][-1]
    assert feedback.role == "tool" and feedback.tool_results[0].is_error
    assert len(executed(result.trace)) == 7


def test_internal_tool_failure_is_fatal_and_recorded(session, monkeypatch):
    original = session.invoke

    def broken(name, arguments):
        if name == "check_group_sizes":
            raise KeyError("PRIVATE_INTERNAL_MESSAGE")
        return original(name, arguments)

    monkeypatch.setattr(session, "invoke", broken)
    with pytest.raises(AgentRunError, match="internal tool error") as caught:
        agent_for(session).run()
    record = executed(caught.value.trace)[-1]
    assert record.fatal_error and record.result.is_error
    assert record.result.output["error_type"] == "KeyError"
    assert "PRIVATE_INTERNAL_MESSAGE" not in caught.value.trace.model_dump_json()


def test_invalid_tool_output_is_fatal_not_blamed_on_model_arguments(session, monkeypatch):
    original = session.invoke

    def broken(name, arguments):
        if name == "check_group_sizes":
            LLMConfig.model_validate({"model_name": ""})
        return original(name, arguments)

    monkeypatch.setattr(session, "invoke", broken)
    with pytest.raises(AgentRunError) as caught:
        agent_for(session).run()
    record = executed(caught.value.trace)[-1]
    assert record.fatal_error
    assert record.result.output["code"] == "invalid_tool_output"


def test_model_failure_records_attempt_without_hidden_retry(session):
    provider = ScriptedProvider(session, turns=[TimeoutError("PRIVATE_TRANSPORT_MESSAGE")])
    with pytest.raises(AgentRunError) as caught:
        agent_for(session, provider).run()
    trace = caught.value.trace
    assert len(provider.requests) == 1
    assert trace.turns[0].response is None
    assert "TimeoutError" in trace.turns[0].error
    assert "PRIVATE_TRANSPORT_MESSAGE" not in trace.model_dump_json()


def test_initial_profile_failure_still_has_a_failed_trace(session, monkeypatch):
    monkeypatch.setattr(session, "invoke", lambda *args: (_ for _ in ()).throw(ValueError("profile failure")))
    with pytest.raises(AgentRunError) as caught:
        agent_for(session).run()
    assert caught.value.trace.status == "failed"
    assert caught.value.trace.turns == []
    assert caught.value.trace.initial_messages


def test_text_only_model_cannot_finish_without_required_tools(session):
    provider = ScriptedProvider(session, turns=[])
    with pytest.raises(AgentRunError, match="model-turn limit") as caught:
        agent_for(session, provider, AgentConfig(max_model_turns=3)).run()
    assert len(caught.value.trace.turns) == 3
    assert provider.final_count == 0
    assert "required tools" in provider.requests[1]["messages"][-1].text


def test_successful_partial_tool_requests_do_not_bypass_evidence_completeness(session):
    partial = [response(calls=[tool_call(name, str(i), {"rates": ["tpr"]} if name == "run_significance_test" else None)])
               for i, name in enumerate(STAGES)]
    provider = ScriptedProvider(session, turns=partial + [response(text="READY_FOR_FINDINGS")])
    with pytest.raises(AgentRunError, match="model-turn limit"):
        agent_for(session, provider, AgentConfig(max_model_turns=8)).run()
    assert "Evidence is incomplete" in provider.requests[-1]["messages"][-1].text
    assert provider.final_count == 0


@pytest.mark.parametrize("first_final", [{}, {"dataset": []}])
def test_invalid_findings_schema_is_retried_once_and_logged(session, first_final):
    provider = ScriptedProvider(session, finals=[first_final])
    result = agent_for(session, provider).run()
    assert provider.final_count == 2
    assert len(result.trace.turns) == 9
    assert result.trace.turns[-2].error == "AuditFindings schema validation failed."
    assert result.trace.turns[-2].response.structured_output == first_final
    assert "Validation feedback" in provider.requests[-1]["messages"][-1].text


def test_bad_json_keeps_response_text_usage_and_latency_through_retry(session):
    invalid = response(text="not JSON")
    provider = ScriptedProvider(session, finals=[StructuredOutputError("Invalid JSON.", invalid)])
    result = agent_for(session, provider).run()
    failed_turn = result.trace.turns[-2]
    assert failed_turn.error == "Structured output parsing failed."
    assert failed_turn.response.text == "not JSON"
    assert failed_turn.response.usage.total_tokens == 15
    assert failed_turn.response.latency_ms == 1.0
    assert provider.final_count == 2


def test_live_style_zero_gap_watch_label_is_corrected_and_logged(session):
    class WrongSeverityProvider(ScriptedProvider):
        def generate(self, **kwargs):
            result = super().generate(**kwargs)
            if kwargs["response_schema"] is not None:
                for item in result.structured_output["attributes"][0]["fairness_findings"]:
                    if item["metric"] == "ppv_difference":
                        item["severity"] = "watch"
            return result

    provider = WrongSeverityProvider(session)
    result = agent_for(session, provider).run()
    ppv = next(item for item in result.findings.attributes[0].fairness_findings if item.metric == "ppv_difference")
    assert ppv.value == 0.0
    assert ppv.severity == "no_evidence_found"
    assert result.claims_validation.passed is True
    assert len(result.trace.severity_corrections) == 1
    correction = result.trace.severity_corrections[0]
    assert correction.proposed_severity == "watch"
    assert correction.corrected_severity == "no_evidence_found"
    original = result.trace.turns[-1].response.structured_output
    assert next(item for item in original["attributes"][0]["fairness_findings"] if item["metric"] == "ppv_difference")["severity"] == "watch"
    assert provider.final_count == 1


def severity_example(session):
    for stage in STAGES:
        session.invoke(stage, {"n_resamples": 29} if stage in ("run_significance_test", "bootstrap_ci") else {})
    return AuditFindings.model_validate(findings_from_evidence(session)), session.get_evidence()


@pytest.mark.parametrize("change", ["gap", "upper", "resamples", "missing_interval", "attribute"])
def test_severity_guard_rejects_quantitative_or_attribute_drift(session, change):
    findings, evidence = severity_example(session)
    finding = findings.attributes[0].fairness_findings[0]
    if change == "gap":
        finding.value = 0.15
    elif change == "upper":
        finding.confidence_interval.upper += 0.001
    elif change == "resamples":
        finding.confidence_interval.total_resamples += 1
    elif change == "missing_interval":
        finding.confidence_interval = None
        finding.confidence_interval_reason = "Omitted."
        finding.severity = "unavailable"
    else:
        findings.attributes[0].attribute = "invented"
        findings.dataset.group_columns = ["invented"]
    with pytest.raises(ValueError):
        apply_severity_policy(findings, evidence)


def test_severity_policy_is_independent_idempotent_and_preserves_numeric_evidence(session):
    findings, evidence = severity_example(session)
    findings.attributes[0].fairness_findings[0].severity = "watch"
    original_findings, original_evidence = findings.model_dump(), evidence.model_dump()
    corrected, corrections = apply_severity_policy(findings, evidence)
    assert len(corrections) == 1
    assert findings.model_dump() == original_findings
    assert evidence.model_dump() == original_evidence
    again, second_corrections = apply_severity_policy(corrected, evidence)
    assert again == corrected
    assert second_corrections == []
    assert corrected.attributes[0].fairness_findings[0].value == findings.attributes[0].fairness_findings[0].value


def test_severity_guard_accepts_negligible_floating_point_roundoff(session):
    findings, evidence = severity_example(session)
    findings.attributes[0].fairness_findings[0].value += 1e-13
    corrected, _ = apply_severity_policy(findings, evidence)
    assert corrected.attributes[0].fairness_findings[0].severity == "no_evidence_found"


def test_json_and_schema_failures_share_one_retry_budget(session):
    provider = ScriptedProvider(session, finals=[StructuredOutputError("Bad JSON.", response(text="bad")), {}])
    with pytest.raises(AgentRunError) as caught:
        agent_for(session, provider).run()
    assert provider.final_count == 2
    assert len(caught.value.trace.turns) == 9
    assert caught.value.trace.status == "failed"


def test_schema_retry_can_be_disabled(session):
    provider = ScriptedProvider(session, finals=[{}])
    with pytest.raises(AgentRunError):
        agent_for(session, provider, AgentConfig(max_schema_retries=0)).run()
    assert provider.final_count == 1


def test_final_transport_failure_is_not_a_schema_retry(session):
    provider = ScriptedProvider(session, finals=[TimeoutError("failed")])
    with pytest.raises(AgentRunError):
        agent_for(session, provider).run()
    assert provider.final_count == 1


def test_final_request_also_respects_model_turn_limit(session):
    provider = ScriptedProvider(session)
    with pytest.raises(AgentRunError, match="model-turn limit") as caught:
        agent_for(session, provider, AgentConfig(max_model_turns=7)).run()
    assert provider.final_count == 0
    assert len(caught.value.trace.turns) == 7


def test_time_limit_after_model_call_preserves_the_received_response(session, monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr("app.agent.perf_counter", lambda: clock["now"])
    provider = ScriptedProvider(session)
    original = provider.generate

    def slow(**kwargs):
        result = original(**kwargs)
        clock["now"] = 2.0
        return result

    monkeypatch.setattr(provider, "generate", slow)
    with pytest.raises(AgentRunError, match="duration limit") as caught:
        agent_for(session, provider, AgentConfig(max_duration_seconds=1.0)).run()
    assert len(provider.requests) == 1
    assert caught.value.trace.turns[0].response.usage.total_tokens == 15


def test_time_limit_after_tool_preserves_executed_tool_record(session, monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr("app.agent.perf_counter", lambda: clock["now"])
    agent = agent_for(session, config=AgentConfig(max_duration_seconds=1.0))
    original = agent._execute_tool

    def slow(call):
        record = original(call)
        clock["now"] = 2.0
        return record

    monkeypatch.setattr(agent, "_execute_tool", slow)
    with pytest.raises(AgentRunError, match="duration limit") as caught:
        agent.run()
    assert len(executed(caught.value.trace)) == 1


def test_failed_trace_is_independent_of_the_original(session):
    agent = agent_for(session)
    trace = agent._new_trace([LLMMessage(role="user", text="Audit.")])
    failed = agent._finish_trace(trace, "failed", "Stopped.")
    error = AgentRunError("Stopped.", failed)
    failed.error = "Changed."
    assert error.trace.error == "Stopped."
    with pytest.raises(ValueError, match="failed trace"):
        AgentRunError("Invalid.", trace)
