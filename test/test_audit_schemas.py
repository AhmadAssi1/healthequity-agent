"""Trust-boundary checks for structured audit output, not numeric claim verification."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.audit_schemas import (
    AttributeFindings,
    AuditFindings,
    BootstrapConfidenceInterval,
    DatasetSummary,
    FairnessFinding,
    GroupMetrics,
    MetricResult,
    StatisticalTestResult,
    SuppressedGroup,
)

METRICS = (
    "demographic_parity_difference",
    "equalized_odds_difference",
    "ppv_difference",
)


def metric(value=0.5, reason=None):
    return {"value": value, "reason": reason}


def interval():
    return {
        "lower": 0.1, "upper": 0.3,
        "valid_resamples": 980, "total_resamples": 1000,
    }


def group(label="A", rate=0.4):
    # Compatible with 50 actual positives and 50 actual negatives per group.
    return {
        "group_value": label,
        "count": 100,
        "base_rate": metric(0.5),
        "selection_rate": metric(rate),
        "tpr": metric(rate),
        "fpr": metric(rate),
        "ppv": metric(0.5),
        "auroc": metric(None, "No score column was provided."),
    }


def finding(name=METRICS[0], *, unavailable=False):
    if unavailable:
        return {
            "metric": name,
            **metric(None, "At least two eligible groups are required."),
            "confidence_interval": None,
            "confidence_interval_reason": "The observed gap is unavailable.",
            "severity": "unavailable",
        }
    return {
        "metric": name,
        **metric(0.0 if name == "ppv_difference" else 0.2),
        # Illustrative bounds for schema tests; not bootstrap estimates.
        "confidence_interval": (
            {**interval(), "lower": 0.0, "upper": 0.08}
            if name == "ppv_difference" else interval()
        ),
        "confidence_interval_reason": None,
        "severity": (
            "no_evidence_found" if name == "ppv_difference" else "substantial"
        ),
    }


def statistical_test(method="two_proportion_z", rate="selection_rate"):
    unavailable = method == "unavailable"
    return {
        "rate": rate,
        "method": method,
        "statistic": (
            -2.8284271247461903 if method == "two_proportion_z"
            else 8.0 if method == "chi_square" else None
        ),
        "p_value": None if unavailable else 0.004677734981047266,
        "min_expected_count": 0.0 if unavailable else 50.0,
        "reason": "Insufficient eligible groups." if unavailable else None,
    }


def attribute():
    return {
        "attribute": "group",
        "groups": [group("A", 0.4), group("B", 0.6)],
        "suppressed_groups": [{
            "group_value": "Small",
            "suppression_reason": "Insufficient evidence: below policy minimum.",
        }],
        "fairness_findings": [finding(name) for name in METRICS],
        "statistical_tests": [statistical_test()],
    }


def dataset():
    return {
        "row_count": 220,
        "column_count": 3,
        "group_columns": ["group"],
        "score_available": False,
    }


@pytest.fixture
def audit_payload():
    return {
        "dataset": dataset(),
        "attributes": [attribute()],
        "summary": "Observed group disparities require further investigation.",
        "warnings": ["One group was suppressed."],
        "limitations": ["Associations do not establish causation."],
    }


@pytest.mark.parametrize("value", [0.0, 0.75, 1.0])
def test_metric_accepts_available_values(value):
    assert MetricResult(**metric(value)).value == value


def test_metric_accepts_unavailable_with_reason():
    result = MetricResult(**metric(None, "No positive predictions."))
    assert result.value is None
    assert result.reason == "No positive predictions."


@pytest.mark.parametrize("payload", [
    metric(-0.01), metric(1.01), metric("0.75"), metric(True),
    metric(float("nan")), metric(float("inf")), metric(float("-inf")),
    metric(None), metric(None, ""), metric(None, " \t\n"),
    metric(0.75, "Unavailable."), {"value": 0.75}, {"reason": None},
    {**metric(), "extra_field": 123},
])
def test_metric_rejects_invalid_results(payload):
    with pytest.raises(ValidationError):
        MetricResult.model_validate(payload)


@pytest.mark.parametrize("lower, upper", [(0.0, 1.0), (0.2, 0.2), (0.1, 0.3)])
def test_interval_accepts_ordered_bounds(lower, upper):
    result = BootstrapConfidenceInterval(**{**interval(), "lower": lower, "upper": upper})
    assert result.lower == lower
    assert result.upper == upper


@pytest.mark.parametrize("changes", [
    {"lower": 0.4}, {"lower": -0.1}, {"upper": 1.1},
    {"lower": float("nan")}, {"upper": float("inf")},
    {"valid_resamples": 0}, {"total_resamples": 0},
    {"valid_resamples": 1001}, {"valid_resamples": True},
    {"total_resamples": 1000.5}, {"valid_resamples": "980"},
])
def test_interval_rejects_invalid_bounds_and_counts(changes):
    with pytest.raises(ValidationError):
        BootstrapConfidenceInterval(**{**interval(), **changes})


@pytest.mark.parametrize("label", [None, "Missing", "A", 1, "1", 1.5, True])
def test_group_preserves_category_type(label):
    result = GroupMetrics(**group(label))
    assert result.group_value == label
    assert type(result.group_value) is type(label)


@pytest.mark.parametrize("changes", [
    {"count": 0}, {"count": -1}, {"count": True}, {"count": 1.5},
    {"group_value": float("nan")}, {"group_value": float("inf")},
    {"tpr": metric(None)}, {"ppv": metric(2.0)},
])
def test_group_rejects_invalid_counts_labels_and_nested_metrics(changes):
    with pytest.raises(ValidationError):
        GroupMetrics(**{**group(), **changes})


@pytest.mark.parametrize("reason", ["", "   ", "\n\t", None])
def test_suppressed_group_requires_meaningful_reason(reason):
    with pytest.raises(ValidationError):
        SuppressedGroup(group_value="Small", suppression_reason=reason)


def test_suppressed_group_does_not_accept_metric_results():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SuppressedGroup(
            group_value="Small", suppression_reason="Insufficient evidence.",
            ppv=metric(),
        )


@pytest.mark.parametrize("severity", ["no_evidence_found", "watch", "substantial"])
def test_finding_accepts_available_severity_labels(severity):
    # Threshold correctness is checked against tool evidence later, not here.
    result = FairnessFinding(**{**finding(), "severity": severity})
    assert result.severity == severity


def test_finding_accepts_unavailable_gap():
    assert FairnessFinding(**finding(unavailable=True)).value is None


def test_finding_accepts_gap_without_interval_with_unavailable_severity():
    result = FairnessFinding(**{
        **finding(), "confidence_interval": None,
        "confidence_interval_reason": "No valid bootstrap resamples.",
        "severity": "unavailable",
    })
    assert result.value == 0.2
    assert result.confidence_interval is None


@pytest.mark.parametrize("changes", [
    {"metric": "accuracy"}, {"severity": "high"},
    {"severity": "unavailable"},
    {"confidence_interval_reason": "Unavailable."},
    {"confidence_interval": None, "confidence_interval_reason": None},
    {"confidence_interval": None, "confidence_interval_reason": "   "},
    {"confidence_interval": None, "confidence_interval_reason": "No interval."},
    {"value": None, "reason": "Undefined gap."},
])
def test_finding_rejects_invalid_or_inconsistent_results(changes):
    with pytest.raises(ValidationError):
        FairnessFinding(**{**finding(), **changes})


@pytest.mark.parametrize("method", [
    "two_proportion_z", "chi_square", "fisher_exact", "fisher_monte_carlo",
    "unavailable",
])
def test_statistical_test_accepts_supported_methods(method):
    assert StatisticalTestResult(**statistical_test(method)).method == method


@pytest.mark.parametrize("rate", ["selection_rate", "tpr", "fpr", "ppv"])
def test_statistical_test_accepts_component_rates(rate):
    assert StatisticalTestResult(**statistical_test(rate=rate)).rate == rate


@pytest.mark.parametrize("changes", [
    {"rate": "equalized_odds_difference"}, {"method": "unknown"},
    {"statistic": None}, {"statistic": float("inf")},
    {"statistic": float("nan")}, {"p_value": None},
    {"p_value": -0.1}, {"p_value": 1.1}, {"p_value": True},
    {"p_value": "0.05"}, {"p_value": float("nan")},
    {"min_expected_count": -1.0}, {"min_expected_count": float("inf")},
    {"reason": "Unavailable."},
])
def test_statistical_test_rejects_invalid_available_results(changes):
    with pytest.raises(ValidationError):
        StatisticalTestResult(**{**statistical_test(), **changes})


@pytest.mark.parametrize("changes", [
    {"p_value": 0.5}, {"statistic": 1.0},
    {"reason": None}, {"reason": ""}, {"reason": "   "},
])
def test_unavailable_test_requires_reason_and_no_results(changes):
    with pytest.raises(ValidationError):
        StatisticalTestResult(**{**statistical_test("unavailable"), **changes})


def test_chi_square_requires_statistic():
    with pytest.raises(ValidationError, match="require a statistic"):
        StatisticalTestResult(**{**statistical_test("chi_square"), "statistic": None})


@pytest.mark.parametrize("changes", [
    {"attribute": "   "}, {"fairness_findings": []},
    {"fairness_findings": [finding()] * 3},
    {"fairness_findings": [finding(name) for name in METRICS] + [finding()]},
    {"statistical_tests": [statistical_test()] * 2},
    {"groups": [group("A"), group("A")]},
    {"suppressed_groups": [
        {"group_value": "A", "suppression_reason": "Insufficient evidence."},
    ]},
    {"suppressed_groups": [
        {"group_value": "Small", "suppression_reason": "Insufficient evidence."},
    ] * 2},
])
def test_attribute_rejects_duplicate_or_incomplete_results(changes):
    with pytest.raises(ValidationError):
        AttributeFindings(**{**attribute(), **changes})


@pytest.mark.parametrize("labels", [(None, "Missing"), (1, "1"), (True, 1), (1, 1.0)])
def test_attribute_keeps_distinct_typed_group_identities(labels):
    result = AttributeFindings(**{
        **attribute(), "groups": [group(label) for label in labels],
    })
    assert len(result.groups) == 2


@pytest.mark.parametrize("group_count", [0, 1])
def test_fewer_than_two_groups_accepts_explicit_unavailable_comparisons(group_count):
    result = AttributeFindings(**{
        **attribute(), "groups": [group() for _ in range(group_count)],
        "fairness_findings": [finding(name, unavailable=True) for name in METRICS],
        "statistical_tests": [statistical_test("unavailable")],
    })
    assert all(item.value is None for item in result.fairness_findings)


@pytest.mark.parametrize("group_count", [0, 1])
def test_fewer_than_two_groups_rejects_available_gaps(group_count):
    with pytest.raises(ValidationError, match="Fairness comparisons require"):
        AttributeFindings(**{
            **attribute(), "groups": [group() for _ in range(group_count)],
            "statistical_tests": [],
        })


@pytest.mark.parametrize("group_count", [0, 1])
def test_fewer_than_two_groups_rejects_available_tests(group_count):
    with pytest.raises(ValidationError, match="Statistical comparisons require"):
        AttributeFindings(**{
            **attribute(), "groups": [group() for _ in range(group_count)],
            "fairness_findings": [finding(name, unavailable=True) for name in METRICS],
        })


@pytest.mark.parametrize("changes", [
    {"row_count": 0}, {"row_count": True}, {"row_count": "220"},
    {"column_count": 0}, {"group_columns": []},
    {"group_columns": [" "]}, {"group_columns": ["group", "group"]},
    {"group_columns": ["a", "b", "c", "d"]},
    {"score_available": "false"},
])
def test_dataset_summary_rejects_invalid_dimensions_and_columns(changes):
    with pytest.raises(ValidationError):
        DatasetSummary(**{**dataset(), **changes})


def test_complete_audit_round_trips_through_json(audit_payload):
    result = AuditFindings.model_validate(audit_payload)
    restored = AuditFindings.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.model_dump() == audit_payload


def test_no_warning_audit_is_valid(audit_payload):
    audit_payload["warnings"] = []
    audit_payload["attributes"][0]["suppressed_groups"] = []
    assert AuditFindings.model_validate(audit_payload).warnings == []


def test_counts_are_checked_per_attribute_not_summed_across_attributes(audit_payload):
    second = deepcopy(audit_payload["attributes"][0])
    second["attribute"] = "other_group"
    audit_payload["attributes"].append(second)
    audit_payload["dataset"]["group_columns"].append("other_group")
    assert len(AuditFindings.model_validate(audit_payload).attributes) == 2


def test_audit_accepts_available_auroc_when_scores_exist(audit_payload):
    audit_payload["dataset"]["score_available"] = True
    audit_payload["attributes"][0]["groups"][0]["auroc"] = metric(0.7)
    assert AuditFindings.model_validate(audit_payload).dataset.score_available


@pytest.mark.parametrize("changes", [
    {"attributes": []}, {"summary": ""}, {"summary": "   "},
    {"warnings": ["   "]}, {"limitations": []}, {"limitations": ["\t"]},
    {"attributes": [attribute(), attribute()]},
    {"dataset": {**dataset(), "group_columns": ["other_group"]}},
    {"dataset": {**dataset(), "row_count": 199}},
])
def test_audit_rejects_inconsistent_summary_and_attributes(audit_payload, changes):
    with pytest.raises(ValidationError):
        AuditFindings.model_validate({**audit_payload, **changes})


def test_audit_rejects_auroc_without_scores(audit_payload):
    audit_payload["attributes"][0]["groups"][0]["auroc"] = metric(0.7)
    with pytest.raises(ValidationError, match="AUROC must be unavailable"):
        AuditFindings.model_validate(audit_payload)


@pytest.mark.parametrize("model, factory", [
    (MetricResult, metric),
    (BootstrapConfidenceInterval, interval),
    (GroupMetrics, group),
    (SuppressedGroup, lambda: {
        "group_value": "Small", "suppression_reason": "Insufficient evidence.",
    }),
    (FairnessFinding, finding),
    (StatisticalTestResult, statistical_test),
    (AttributeFindings, attribute),
    (DatasetSummary, dataset),
])
def test_nested_schemas_forbid_unknown_fields(model, factory):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate({**factory(), "invented_field": 123})


def test_audit_forbids_unknown_top_level_fields(audit_payload):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AuditFindings.model_validate({**audit_payload, "invented_field": 123})


def test_json_schema_exposes_required_structured_output_contract():
    schema = AuditFindings.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "dataset", "attributes", "summary", "warnings", "limitations",
    }
    assert schema["$defs"]["MetricResult"]["additionalProperties"] is False
    assert set(schema["$defs"]["MetricResult"]["required"]) == {"value", "reason"}
