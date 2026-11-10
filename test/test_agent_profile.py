"""Filtering and serialization at the internal-to-agent profile boundary."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.agent_profile import build_agent_profile
from app.tool_schemas import (
    AgentAttributeProfile,
    AgentDatasetProfile,
    DatasetProfile,
    SuppressedGroupSummary,
)
from fairness.policy import SuppressionPolicy


def internal_profile(specs=None, flags=None):
    if specs is None:
        specs = [("A", 50, 10), ("B", 49, 9)]
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
        data_quality_flags=flags or [],
    )


def suppressed(label="B"):
    return {
        "group_value": label,
        "reason_codes": ["below_min_group_size"],
    }


@pytest.fixture
def agent_payload():
    return build_agent_profile(internal_profile(), SuppressionPolicy()).model_dump()


def test_output_keeps_eligible_counts_but_not_suppressed_counts():
    result = build_agent_profile(internal_profile(), SuppressionPolicy())
    attribute = result.model_dump()["attributes"][0]
    assert attribute["reportable_groups"] == [{
        "group_value": "A", "count": 50, "positive_outcomes": 10,
    }]
    assert attribute["suppressed_groups"] == [{
        "group_value": "B",
        "reason_codes": ["below_min_group_size", "below_min_positive_outcomes"],
    }]
    # Assert fields, not digit substrings: overall totals are intentionally retained.
    assert set(attribute["suppressed_groups"][0]) == {"group_value", "reason_codes"}
    assert "positive_outcomes" not in result.model_dump()
    assert "suppression_reason" not in result.model_dump_json()


@pytest.mark.parametrize("count, positives, reasons", [
    (49, 10, ["below_min_group_size"]),
    (50, 9, ["below_min_positive_outcomes"]),
    (49, 9, ["below_min_group_size", "below_min_positive_outcomes"]),
])
def test_suppression_codes_identify_the_actual_policy_failures(count, positives, reasons):
    result = build_agent_profile(
        internal_profile([("B", count, positives)]), SuppressionPolicy(),
    )
    attribute = result.attributes[0]
    assert attribute.reportable_groups == []
    assert attribute.suppressed_groups[0].reason_codes == reasons


def test_all_groups_reportable_at_exact_policy_boundaries():
    result = build_agent_profile(
        internal_profile([("A", 50, 10), ("B", 50, 10)]), SuppressionPolicy(),
    )
    assert len(result.attributes[0].reportable_groups) == 2
    assert result.attributes[0].suppressed_groups == []


def test_custom_policy_changes_which_counts_may_be_reported():
    result = build_agent_profile(
        internal_profile([("A", 3, 1), ("B", 2, 1)]),
        SuppressionPolicy(min_group_size=3, min_positive_outcomes=1),
    )
    assert [g.group_value for g in result.attributes[0].reportable_groups] == ["A"]
    assert result.attributes[0].suppressed_groups[0].group_value == "B"


def test_omitted_policy_is_loaded_by_group_checker(monkeypatch):
    monkeypatch.setattr(
        "app.group_checks.load_suppression_policy",
        lambda: SuppressionPolicy(min_group_size=1, min_positive_outcomes=0),
    )
    result = build_agent_profile(internal_profile([("A", 1, 0)]))
    assert len(result.attributes[0].reportable_groups) == 1


def test_keeps_typed_group_identities_and_missing_group():
    labels = [None, "Missing", 1, "1", True, 1.0]
    result = build_agent_profile(
        internal_profile([(label, 1, 0) for label in labels]), SuppressionPolicy(),
    )
    groups = result.attributes[0].suppressed_groups
    assert [g.group_value for g in groups] == labels
    assert [type(g.group_value) for g in groups] == [type(label) for label in labels]
    assert AgentDatasetProfile.model_validate_json(result.model_dump_json()) == result


def test_output_does_not_share_mutable_groups_or_policy_with_inputs():
    profile = internal_profile()
    original = profile.model_dump()
    policy = SuppressionPolicy()
    result = build_agent_profile(profile, policy)
    assert profile.model_dump() == original
    profile.attributes[0].groups[0].count = 1000
    policy.min_group_size = 999
    assert result.attributes[0].reportable_groups[0].count == 50
    assert result.suppression_policy.min_group_size == 50


def test_retains_overall_summary_and_score_presence():
    profile = internal_profile()
    profile.score_available = True
    result = build_agent_profile(profile, SuppressionPolicy())
    assert result.row_count == profile.row_count
    assert result.column_count == profile.column_count
    assert result.outcome_prevalence == profile.outcome_prevalence
    assert result.score_available is True


def test_multiple_attributes_are_filtered_independently():
    payload = internal_profile().model_dump()
    second = deepcopy(payload["attributes"][0])
    second["attribute"] = "other"
    payload["attributes"].append(second)
    result = build_agent_profile(DatasetProfile.model_validate(payload), SuppressionPolicy())
    assert [item.attribute for item in result.attributes] == ["group", "other"]
    assert all(len(item.reportable_groups) == 1 for item in result.attributes)


def test_known_quality_flags_are_preserved_and_deduplicated():
    flags = [
        "Predictions are not mapped; fairness analysis requires predictions.",
        "Duplicate rows exist in the mapped audit columns; no rows were removed.",
        "The outcome column contains only one class.",
        "Missing values in 'group' form a separate group.",
    ]
    result = build_agent_profile(internal_profile(flags=flags + flags), SuppressionPolicy())
    assert result.data_quality_flags == flags


def test_arbitrary_internal_warnings_are_replaced_without_copying_counts_or_text():
    flags = [
        "Group B has SECRET_COUNT_49 cases and SECRET_POSITIVES_9 outcomes.",
        "PRIVATE_INTERNAL_TEXT",
    ]
    profile = internal_profile(flags=flags)
    result = build_agent_profile(profile, SuppressionPolicy())
    assert result.data_quality_flags == [
        "Additional internal data quality flags require review.",
    ]
    assert "SECRET_" not in result.model_dump_json()
    assert "PRIVATE_INTERNAL_TEXT" not in result.model_dump_json()
    assert profile.data_quality_flags == flags


@pytest.mark.parametrize("changes", [
    {"reason_codes": []}, {"reason_codes": ["unknown"]},
    {"reason_codes": ["below_min_group_size"] * 2},
    {"reason_codes": ["below_min_group_size"] * 3},
    {"count": 49}, {"positive_outcomes": 9},
    {"suppression_reason": "Group size 49 is too small."},
])
def test_suppressed_summary_rejects_invalid_codes_and_extra_count_fields(changes):
    with pytest.raises(ValidationError):
        SuppressedGroupSummary.model_validate({**suppressed(), **changes})


@pytest.mark.parametrize("changes", [
    {"attribute": "   "},
    {"reportable_groups": [], "suppressed_groups": []},
    {"suppressed_groups": [suppressed(), suppressed()]},
    {"suppressed_groups": [suppressed("A")]},
    {"unknown_field": 123},
])
def test_agent_attribute_rejects_inconsistent_groups(agent_payload, changes):
    with pytest.raises(ValidationError):
        AgentAttributeProfile.model_validate({
            **agent_payload["attributes"][0], **changes,
        })


@pytest.mark.parametrize("changes", [
    {"row_count": 0}, {"column_count": 0}, {"outcome_prevalence": 1.1},
    {"outcome_prevalence": float("nan")}, {"score_available": "false"},
    {"attributes": []}, {"data_quality_flags": ["   "]}, {"unknown_field": 123},
])
def test_agent_dataset_rejects_invalid_summary(agent_payload, changes):
    with pytest.raises(ValidationError):
        AgentDatasetProfile.model_validate({**agent_payload, **changes})


def test_agent_dataset_rejects_duplicate_attributes(agent_payload):
    agent_payload["attributes"] *= 2
    with pytest.raises(ValidationError, match="Attribute names must be unique"):
        AgentDatasetProfile.model_validate(agent_payload)


def test_agent_dataset_rejects_more_attributes_than_columns(agent_payload):
    second = deepcopy(agent_payload["attributes"][0])
    second["attribute"] = "other"
    agent_payload["attributes"].append(second)
    agent_payload["column_count"] = 1
    with pytest.raises(ValidationError, match="must not exceed the column count"):
        AgentDatasetProfile.model_validate(agent_payload)


@pytest.mark.parametrize("changes", [{"count": 49}, {"positive_outcomes": 9}])
def test_agent_dataset_rejects_ineligible_reported_groups(agent_payload, changes):
    agent_payload["attributes"][0]["reportable_groups"][0].update(changes)
    with pytest.raises(ValidationError, match="must meet the suppression policy"):
        AgentDatasetProfile.model_validate(agent_payload)


def test_reported_counts_cannot_exceed_total_rows(agent_payload):
    agent_payload["row_count"] = 49
    with pytest.raises(ValidationError, match="must not exceed the row count"):
        AgentDatasetProfile.model_validate(agent_payload)


def test_without_suppression_all_rows_must_be_accounted_for(agent_payload):
    agent_payload["attributes"][0]["suppressed_groups"] = []
    with pytest.raises(ValidationError, match="every row must be accounted for"):
        AgentDatasetProfile.model_validate(agent_payload)
