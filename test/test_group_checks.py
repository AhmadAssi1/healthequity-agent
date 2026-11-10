"""Eligibility tool integration and structured decision validation."""

import pandas as pd
import pytest
from pydantic import ValidationError

from app.group_checks import check_group_sizes
from app.profiling import profile_dataset
from app.schemas import ColumnMapping
from app.tool_schemas import (
    AttributeEligibility,
    DatasetProfile,
    GroupEligibility,
    GroupSizeCheckResult,
)
from fairness.policy import SuppressionPolicy


def profile_for_groups(specs):
    """Build a valid internal profile from (label, count, positives) fixtures."""
    total = sum(count for _, count, _ in specs)
    positives = sum(positive for _, _, positive in specs)
    return DatasetProfile(
        row_count=total,
        column_count=3,
        positive_outcomes=positives,
        outcome_prevalence=positives / total,
        score_available=False,
        attributes=[{
            "attribute": "group",
            "groups": [
                {"group_value": label, "count": count, "positive_outcomes": positive}
                for label, count, positive in specs
            ],
        }],
        data_quality_flags=[],
    )


def decision(label="A", reportable=True):
    return {
        "group_value": label,
        "reportable": reportable,
        "suppression_reason": None if reportable else "Insufficient evidence.",
    }


@pytest.mark.parametrize("count, positives, reportable, reason_parts", [
    (49, 10, False, ["Group size"]),
    (50, 9, False, ["Positive outcomes"]),
    (49, 9, False, ["Group size", "Positive outcomes"]),
    (50, 10, True, []),
    (51, 11, True, []),
    (50, 0, False, ["Positive outcomes"]),
])
def test_suppression_boundaries(count, positives, reportable, reason_parts):
    result = check_group_sizes(
        profile_for_groups([("A", count, positives)]),
        SuppressionPolicy(min_group_size=50, min_positive_outcomes=10),
    )
    actual = result.attributes[0].groups[0]
    assert actual.reportable is reportable
    for part in reason_parts:
        assert part in actual.suppression_reason
    if reportable:
        assert actual.suppression_reason is None


def test_custom_policy_is_used_instead_of_fixed_thresholds():
    profile = profile_for_groups([("A", 20, 4)])
    strict = SuppressionPolicy(min_group_size=50, min_positive_outcomes=10)
    relaxed = SuppressionPolicy(min_group_size=20, min_positive_outcomes=4)
    assert check_group_sizes(profile, strict).attributes[0].groups[0].reportable is False
    result = check_group_sizes(profile, relaxed)
    assert result.attributes[0].groups[0].reportable is True
    assert result.suppression_policy == relaxed


def test_omitted_policy_uses_configuration_loader(monkeypatch):
    policy = SuppressionPolicy(min_group_size=3, min_positive_outcomes=1)
    calls = []

    def load_policy():
        calls.append(True)
        return policy

    monkeypatch.setattr("app.group_checks.load_suppression_policy", load_policy)
    result = check_group_sizes(profile_for_groups([("A", 3, 1)]))
    assert calls == [True]
    assert result.attributes[0].groups[0].reportable is True
    assert result.suppression_policy == policy


def test_explicit_policy_does_not_load_configuration(monkeypatch):
    def unexpected_load():
        pytest.fail("An explicit policy must not access configuration.")

    monkeypatch.setattr("app.group_checks.load_suppression_policy", unexpected_load)
    result = check_group_sizes(
        profile_for_groups([("A", 50, 10)]), SuppressionPolicy(),
    )
    assert result.attributes[0].groups[0].reportable is True


def test_policy_snapshot_and_input_profile_are_independent():
    profile = profile_for_groups([("A", 50, 10), ("B", 49, 9)])
    original = profile.model_dump()
    policy = SuppressionPolicy()
    result = check_group_sizes(profile, policy)
    assert profile.model_dump() == original
    assert result.suppression_policy is not policy
    policy.min_group_size = 100
    policy.min_positive_outcomes = 20
    assert result.suppression_policy.min_group_size == 50
    assert result.suppression_policy.min_positive_outcomes == 10
    assert result.attributes[0].groups[0].reportable is True


def test_preserves_missing_and_typed_labels_without_dropping_suppressed_groups():
    labels = [None, "Missing", 1, "1", True, 1.0]
    profile = profile_for_groups([(label, 50, 9) for label in labels])
    result = check_group_sizes(profile, SuppressionPolicy())
    groups = result.attributes[0].groups
    assert len(groups) == len(labels)
    assert [g.group_value for g in groups] == labels
    assert [type(g.group_value) for g in groups] == [type(label) for label in labels]
    assert all(not g.reportable and g.suppression_reason for g in groups)


def test_multiple_attributes_are_checked_independently():
    profile = profile_for_groups([("A", 50, 10), ("B", 50, 10)])
    profile = DatasetProfile.model_validate({
        **profile.model_dump(),
        "attributes": [*profile.model_dump()["attributes"], {
            "attribute": "other",
            "groups": [
                {"group_value": "C", "count": 90, "positive_outcomes": 19},
                {"group_value": "D", "count": 10, "positive_outcomes": 1},
            ],
        }],
    })
    result = check_group_sizes(profile, SuppressionPolicy())
    assert [item.attribute for item in result.attributes] == ["group", "other"]
    assert [g.reportable for g in result.attributes[0].groups] == [True, True]
    assert [g.reportable for g in result.attributes[1].groups] == [True, False]


def test_profile_to_eligibility_pipeline_and_json_round_trip():
    df = pd.DataFrame({
        "actual": [1] * 10 + [0] * 40 + [1] * 9 + [0] * 40,
        "predicted": [0] * 99,
        "group": ["A"] * 50 + [None] * 49,
    })
    profile = profile_dataset(
        df, ColumnMapping(outcome="actual", prediction="predicted", groups=["group"]),
    )
    result = check_group_sizes(profile, SuppressionPolicy())
    assert [g.reportable for g in result.attributes[0].groups] == [True, False]
    assert result.attributes[0].groups[1].group_value is None
    assert GroupSizeCheckResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("reportable", [True, False])
def test_decision_accepts_consistent_availability(reportable):
    assert GroupEligibility(**decision(reportable=reportable)).reportable is reportable


@pytest.mark.parametrize("changes", [
    {"reportable": "true"}, {"reportable": 1},
    {"suppression_reason": "Unavailable."},
    {"group_value": float("nan")}, {"count": 50},
])
def test_reportable_decision_rejects_invalid_values(changes):
    with pytest.raises(ValidationError):
        GroupEligibility.model_validate({**decision(), **changes})


@pytest.mark.parametrize("reason", [None, "", " \t\n"])
def test_suppressed_decision_requires_nonblank_reason(reason):
    with pytest.raises(ValidationError):
        GroupEligibility(**{
            **decision(reportable=False), "suppression_reason": reason,
        })


@pytest.mark.parametrize("changes", [
    {"attribute": " "}, {"groups": []},
    {"groups": [decision(), decision(reportable=False)]}, {"unknown_field": 1},
])
def test_attribute_eligibility_rejects_invalid_structure(changes):
    with pytest.raises(ValidationError):
        AttributeEligibility.model_validate({
            "attribute": "group", "groups": [decision()], **changes,
        })


@pytest.mark.parametrize("changes", [
    {"attributes": []},
    {"attributes": [{"attribute": "group", "groups": [decision()]}] * 2},
    {"suppression_policy": {"min_group_size": 0}},
    {"unknown_field": 1},
])
def test_group_size_result_rejects_invalid_structure(changes):
    with pytest.raises(ValidationError):
        GroupSizeCheckResult.model_validate({
            "suppression_policy": SuppressionPolicy(),
            "attributes": [{"attribute": "group", "groups": [decision()]}],
            **changes,
        })
