"""Nine tool contracts, numerical adapters, and injectable-backend lifecycle."""

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from app.audit_schemas import (
    AttributeFindings,
    AuditFindings,
    BootstrapConfidenceInterval,
    DatasetSummary,
    FairnessFinding,
    StatisticalTestResult,
    SuppressedGroup,
)
from app.schemas import ColumnMapping
from app.tool_contracts import (
    AttributeToolInput,
    BootstrapCIInput,
    BootstrapResult,
    ClaimIssue,
    ClaimsValidationOutput,
    EmptyToolInput,
    RenderReportOutput,
    SignificanceTestInput,
)
from app.tools import (
    METRICS,
    TOOL_SPECS,
    AuditToolSession,
    ToolBackends,
    ToolExecutionError,
    ToolUnavailableError,
)
from fairness.policy import SuppressionPolicy, classify_severity


@pytest.fixture
def df():
    return pd.DataFrame({
        "actual": [0, 0, 1, 1, 0, 0, 1, 1, 1],
        "predicted": [0, 1, 0, 1, 0, 0, 1, 1, 0],
        "group": ["A"] * 4 + ["B"] * 4 + ["small"],
        "risk": [0.1, 0.6, 0.4, 0.9, 0.1, 0.2, 0.8, 0.9, 0.3],
    })


@pytest.fixture
def mapping():
    return ColumnMapping(outcome="actual", prediction="predicted", groups=["group"], score="risk")


@pytest.fixture
def policy():
    return SuppressionPolicy(min_group_size=4, min_positive_outcomes=2)


@pytest.fixture
def session(df, mapping, policy):
    return AuditToolSession(df, mapping, policy)


def complete_evidence(session):
    session.invoke("compute_fairness_metrics", {})
    session.invoke("run_significance_test", {"n_resamples": 99})
    session.invoke("bootstrap_ci", {"n_resamples": 99})


def fake_drafter(evidence):
    """Test double assembles evidence; this is not an LLM drafting implementation."""
    bootstraps = {(item.attribute, item.metric): item for item in evidence.bootstrap.results}
    attributes = []
    for metrics in evidence.metrics.results:
        findings = []
        for gap in metrics.gaps:
            interval = bootstraps[metrics.attribute, gap.metric]
            ci = None if interval.lower is None else BootstrapConfidenceInterval(
                lower=interval.lower, upper=interval.upper,
                valid_resamples=interval.valid_resamples, total_resamples=interval.total_resamples,
            )
            findings.append(FairnessFinding(
                metric=gap.metric, value=gap.value, reason=gap.reason,
                confidence_interval=ci,
                confidence_interval_reason=interval.reason,
                severity=classify_severity(gap.value, interval.lower, interval.upper, evidence.severity_policy),
            ))
        attributes.append(AttributeFindings(
            attribute=metrics.attribute, groups=metrics.groups,
            suppressed_groups=[SuppressedGroup(
                group_value=item.group_value,
                suppression_reason="Insufficient evidence: " + ", ".join(item.reason_codes),
            ) for item in metrics.suppressed_groups],
            fairness_findings=findings,
            statistical_tests=[StatisticalTestResult(**item.model_dump(exclude={"attribute"}))
                               for item in evidence.significance.results
                               if item.attribute == metrics.attribute and item.min_expected_count is not None],
        ))
    return AuditFindings(
        dataset=DatasetSummary(
            row_count=evidence.profile.row_count, column_count=evidence.profile.column_count,
            group_columns=[item.attribute for item in attributes], score_available=evidence.profile.score_available,
        ),
        attributes=attributes, summary="Observed associations require further investigation.",
        warnings=evidence.profile.data_quality_flags,
        limitations=["Independent-row bootstrap does not account for repeated patient stays."],
    )


def test_registry_has_exactly_the_nine_brief_tools_and_exportable_schemas(session):
    expected = {
        "profile_dataset", "check_group_sizes", "select_analyses",
        "compute_fairness_metrics", "run_significance_test", "bootstrap_ci",
        "draft_findings", "validate_claims", "render_report",
    }
    assert set(TOOL_SPECS) == expected
    definitions = session.definitions()
    assert {item.name for item in definitions} == expected
    assert sum(item.available for item in definitions) == 7
    for definition in definitions:
        assert definition.input_schema["additionalProperties"] is False
        assert definition.output_schema["additionalProperties"] is False
        json.dumps(definition.model_dump(), allow_nan=False)


@pytest.mark.parametrize("name", list(TOOL_SPECS))
def test_all_tool_inputs_reject_row_payloads_and_arbitrary_extra_arguments(session, name):
    with pytest.raises(ValidationError):
        session.invoke(name, {"dataframe": [{"patient": "secret"}]})


def test_profile_and_eligibility_hide_suppressed_counts(session):
    profile = session.invoke("profile_dataset", {})
    eligibility = session.invoke("check_group_sizes", {})
    for output in [profile, eligibility]:
        suppressed = output.attributes[0].suppressed_groups[0].model_dump()
        assert suppressed == {
            "group_value": "small",
            "reason_codes": ["below_min_group_size", "below_min_positive_outcomes"],
        }


def test_session_snapshots_dataset_mapping_and_policy(df, mapping, policy):
    session = AuditToolSession(df, mapping, policy)
    df["actual"] = 0
    mapping.groups.clear()
    policy.min_group_size = 999
    profile = session.invoke("profile_dataset", {})
    assert profile.attributes[0].attribute == "group"
    assert profile.suppression_policy.min_group_size == 4
    assert profile.attributes[0].reportable_groups[0].positive_outcomes == 2
    profile.attributes.clear()
    assert session.invoke("profile_dataset", {}).attributes


def test_selection_guard_is_used_for_untrusted_proposals(session):
    result = session.invoke("select_analyses", {"proposed_plan": {"score_available": False}})
    assert result.plan.score_available is True
    assert result.corrections


def test_metrics_match_hand_calculation_and_contain_no_nan(session):
    result = session.invoke("compute_fairness_metrics", {}).results[0]
    groups = {item.group_value: item for item in result.groups}
    assert set(groups) == {"A", "B"}
    assert groups["A"].tpr.value == 0.5
    assert groups["A"].fpr.value == 0.5
    assert groups["B"].tpr.value == 1.0
    assert groups["B"].fpr.value == 0.0
    assert groups["A"].auroc.value is not None
    gaps = {item.metric: item.value for item in result.gaps}
    assert gaps == {
        "demographic_parity_difference": 0.0,
        "equalized_odds_difference": 0.5,
        "ppv_difference": 0.5,
    }
    json.dumps(result.model_dump(), allow_nan=False)


def test_without_scores_auroc_has_null_and_reason(df, mapping, policy):
    mapping.score = None
    result = AuditToolSession(df, mapping, policy).invoke("compute_fairness_metrics", {})
    assert all(g.auroc.value is None and g.auroc.reason for g in result.results[0].groups)


def test_all_suppressed_and_single_eligible_group_return_explicit_unavailable_gaps(df, mapping):
    for policy in [SuppressionPolicy(), SuppressionPolicy(min_group_size=4, min_positive_outcomes=2)]:
        subset = df if policy.min_group_size == 50 else df.iloc[:5]
        session = AuditToolSession(subset, mapping, policy)
        metrics = session.invoke("compute_fairness_metrics", {}).results[0]
        assert len(metrics.groups) < 2
        assert all(gap.value is None and gap.reason for gap in metrics.gaps)
        assert all(item.method == "unavailable" for item in session.invoke("run_significance_test", {}).results)
        assert all(item.lower is None and item.reason for item in session.invoke("bootstrap_ci", {}).results)


def test_significance_returns_each_rate_and_matches_approved_methods(session):
    plan = session.invoke("select_analyses", {}).plan.attributes[0]
    results = session.invoke("run_significance_test", {}).results
    assert len(results) == 4
    assert {item.rate: item.method for item in results} == {
        item.rate: item.method for item in plan.statistical_tests
    }
    assert all(item.p_value is not None for item in results)


def test_forced_wrong_test_is_rejected_before_execution(session):
    with pytest.raises(ToolExecutionError, match="not permitted"):
        session.invoke("run_significance_test", {
            "attributes": ["group"], "rates": ["tpr"], "test_name": "two_proportion_z",
        })


def test_permitted_explicit_test_can_be_executed(session):
    result = session.invoke("run_significance_test", {
        "attributes": ["group"], "rates": ["tpr"], "test_name": "fisher_exact",
    })
    assert len(result.results) == 1
    assert result.results[0].method == "fisher_exact"


def test_degenerate_rates_and_bootstrap_are_explicitly_unavailable(df, mapping, policy):
    df["predicted"] = 0
    session = AuditToolSession(df, mapping, policy)
    tests = {item.rate: item for item in session.invoke("run_significance_test", {}).results}
    assert tests["selection_rate"].method == "unavailable"
    assert tests["selection_rate"].min_expected_count == 0.0
    assert tests["ppv"].min_expected_count is None
    intervals = session.invoke("bootstrap_ci", {"n_resamples": 19})
    ppv = next(item for item in intervals.results if item.metric == "ppv_difference")
    assert ppv.observed is None and ppv.lower is None and ppv.reason
    json.dumps(intervals.model_dump(), allow_nan=False)


def test_bootstrap_is_seeded_and_carries_resample_metadata(session):
    arguments = {"n_resamples": 99, "seed": 17}
    first = session.invoke("bootstrap_ci", arguments)
    second = session.invoke("bootstrap_ci", arguments)
    assert first == second
    assert {item.metric for item in first.results} == set(METRICS)
    assert all(item.seed == 17 and item.total_resamples == 99 for item in first.results)
    assert all(item.lower is not None for item in first.results)


def test_multiple_attributes_batch_in_one_call(df, mapping, policy):
    df["other"] = df["group"]
    mapping.groups.append("other")
    session = AuditToolSession(df, mapping, policy)
    assert len(session.invoke("compute_fairness_metrics", {}).results) == 2
    assert len(session.invoke("run_significance_test", {}).results) == 8
    assert len(session.invoke("bootstrap_ci", {"n_resamples": 29}).results) == 6


def test_partial_requests_limit_output_and_do_not_satisfy_complete_evidence(df, mapping, policy):
    df["other"] = df["group"]
    mapping.groups.append("other")
    session = AuditToolSession(df, mapping, policy, backends=ToolBackends(draft_findings=fake_drafter))
    result = session.invoke("compute_fairness_metrics", {"attributes": ["other"]})
    assert [item.attribute for item in result.results] == ["other"]
    assert len(session.invoke("run_significance_test", {"rates": ["tpr"]}).results) == 2
    assert len(session.invoke("bootstrap_ci", {"metrics": ["ppv_difference"], "n_resamples": 19}).results) == 2
    with pytest.raises(ToolExecutionError, match="Compute metrics"):
        session.invoke("draft_findings", {})


@pytest.mark.parametrize("tool", ["compute_fairness_metrics", "run_significance_test", "bootstrap_ci"])
def test_unmapped_group_column_is_rejected(session, tool):
    with pytest.raises(ToolExecutionError, match="mapped group"):
        session.invoke(tool, {"attributes": ["risk"]})


@pytest.mark.parametrize("tool", ["select_analyses", "compute_fairness_metrics", "run_significance_test", "bootstrap_ci"])
def test_analysis_requires_predictions_but_profiling_does_not(df, mapping, policy, tool):
    mapping.prediction = None
    session = AuditToolSession(df, mapping, policy)
    assert session.invoke("profile_dataset", {}).row_count == len(df)
    with pytest.raises(ToolExecutionError, match="mapped predictions"):
        session.invoke(tool, {})


def test_typed_group_identity_survives_fairlearn_and_bootstrap_adapters():
    df = pd.DataFrame({
        "actual": [0, 1, 0, 1, 0, 1], "predicted": [0, 1, 1, 0, 0, 1],
        "group": pd.Series([1, True, "1", 1.0, None, "Missing"], dtype=object),
    })
    session = AuditToolSession(
        df, ColumnMapping(outcome="actual", prediction="predicted", groups=["group"]),
        SuppressionPolicy(min_group_size=1, min_positive_outcomes=0),
    )
    groups = session.invoke("compute_fairness_metrics", {}).results[0].groups
    assert [type(g.group_value) for g in groups] == [int, bool, str, float, type(None), str]
    assert all(g.count == 1 for g in groups)
    tests = session.invoke("run_significance_test", {}).results
    assert any(item.method == "unavailable" and item.min_expected_count is None for item in tests)
    bootstrap = session.invoke("bootstrap_ci", {"n_resamples": 19}).results
    assert next(item for item in bootstrap if item.metric == "equalized_odds_difference").observed is None


def test_absent_backends_fail_explicitly(session):
    for tool in ["draft_findings", "render_report"]:
        with pytest.raises(ToolUnavailableError, match="not configured"):
            session.invoke(tool, {})
    complete_evidence(session)
    findings = fake_drafter(session._evidence())
    assert session.invoke("validate_claims", {"findings": findings.model_dump()}).passed


def test_drafting_requires_complete_evidence(df, mapping, policy):
    session = AuditToolSession(df, mapping, policy, backends=ToolBackends(draft_findings=fake_drafter))
    with pytest.raises(ToolExecutionError, match="Compute metrics"):
        session.invoke("draft_findings", {})


def test_all_nine_interfaces_with_test_backends_and_render_validation_gate(df, mapping, policy):
    rendered = []

    def fake_render(findings):
        rendered.append(findings)
        return RenderReportOutput(pdf_path="test-only.pdf", run_id="test-run", trace_id="test-trace")

    session = AuditToolSession(df, mapping, policy, backends=ToolBackends(
        draft_findings=fake_drafter,
        validate_claims=lambda findings, evidence: ClaimsValidationOutput(passed=True, issues=[]),
        render_report=fake_render,
    ))
    assert all(item.available for item in session.definitions())
    for name in ["profile_dataset", "check_group_sizes", "select_analyses"]:
        session.invoke(name, {})
    complete_evidence(session)
    draft = session.invoke("draft_findings", {})
    with pytest.raises(ToolExecutionError, match="passed claim validation"):
        session.invoke("render_report", {})
    verdict = session.invoke("validate_claims", {"findings": draft.findings.model_dump()})
    assert verdict.passed
    assert session.invoke("render_report", {}).trace_id == "test-trace"
    assert rendered[0] == draft.findings
    # Any change/recomputation of evidence invalidates the previous approval.
    session.invoke("bootstrap_ci", {"n_resamples": 29})
    with pytest.raises(ToolExecutionError, match="passed claim validation"):
        session.invoke("render_report", {})


def test_failed_claim_validation_does_not_authorize_rendering(df, mapping, policy):
    session = AuditToolSession(df, mapping, policy, backends=ToolBackends(
        validate_claims=lambda findings, evidence: {
            "passed": False, "issues": [{"code": "unsupported_number", "message": "Unverified number."}],
        },
        render_report=lambda findings: pytest.fail("Must not render failed findings."),
    ))
    complete_evidence(session)
    findings = fake_drafter(session._evidence())
    assert not session.invoke("validate_claims", {"findings": findings.model_dump()}).passed
    with pytest.raises(ToolExecutionError):
        session.invoke("render_report", {})


def test_mutated_backend_models_are_revalidated(df, mapping, policy):
    bad = ClaimsValidationOutput(passed=True, issues=[])
    bad.issues = [ClaimIssue(code="unsupported_number", message="Contradictory verdict.")]
    session = AuditToolSession(df, mapping, policy, backends=ToolBackends(
        validate_claims=lambda findings, evidence: bad,
    ))
    complete_evidence(session)
    findings = fake_drafter(session._evidence())
    with pytest.raises(ValidationError):
        session.invoke("validate_claims", {"findings": findings.model_dump()})
    assert session._validated_findings is None


def test_unknown_tool_is_rejected(session):
    with pytest.raises(ToolExecutionError, match="Unknown audit tool"):
        session.invoke("read_any_file", {})


@pytest.mark.parametrize("model, payload", [
    (EmptyToolInput, {"extra": 1}),
    (AttributeToolInput, {"attributes": []}),
    (AttributeToolInput, {"attributes": ["A", "A"]}),
    (SignificanceTestInput, {"rates": ["tpr", "tpr"]}),
    (SignificanceTestInput, {"test_name": "chi_square"}),
    (SignificanceTestInput, {"n_resamples": True}),
    (BootstrapCIInput, {"metrics": ["ppv_difference", "ppv_difference"]}),
    (BootstrapCIInput, {"n_resamples": 0}),
    (BootstrapCIInput, {"seed": -1}),
    (BootstrapCIInput, {"n_resamples": 100_001}),
    (ClaimsValidationOutput, {"passed": False, "issues": []}),
])
def test_input_and_verdict_contracts_reject_invalid_requests(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize("changes", [
    {"lower": None}, {"upper": 0.05}, {"valid_resamples": 0},
    {"valid_resamples": 101}, {"observed": None}, {"reason": "Unavailable."},
])
def test_bootstrap_output_rejects_inconsistent_intervals(changes):
    with pytest.raises(ValidationError):
        BootstrapResult.model_validate({
            "attribute": "group", "metric": "ppv_difference", "observed": 0.2,
            "lower": 0.1, "upper": 0.3, "valid_resamples": 100, "total_resamples": 100,
            "seed": 42, "reason": None, **changes,
        })
