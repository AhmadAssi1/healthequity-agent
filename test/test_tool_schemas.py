"""Validation of internal profiling contracts and aggregate consistency."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.tool_schemas import AttributeProfile, DatasetProfile, GroupCounts


def group_counts(label="A", count=2, positives=1):
    return {
        "group_value": label, "count": count, "positive_outcomes": positives,
    }


@pytest.fixture
def profile_payload():
    return {
        "row_count": 4,
        "column_count": 3,
        "positive_outcomes": 2,
        "outcome_prevalence": 0.5,
        "score_available": False,
        "attributes": [{
            "attribute": "group",
            "groups": [group_counts("A"), group_counts("B")],
        }],
        "data_quality_flags": [],
    }


@pytest.mark.parametrize("positives", [0, 1, 2])
def test_group_counts_accepts_boundary_outcome_counts(positives):
    assert GroupCounts(**group_counts(positives=positives)).positive_outcomes == positives


@pytest.mark.parametrize("changes", [
    {"count": 0}, {"count": -1}, {"count": True}, {"count": 2.5},
    {"count": "2"}, {"positive_outcomes": -1}, {"positive_outcomes": 3},
    {"positive_outcomes": True}, {"positive_outcomes": 1.5},
    {"group_value": float("nan")}, {"group_value": float("inf")},
    {"unknown_field": 1},
])
def test_group_counts_rejects_invalid_values(changes):
    with pytest.raises(ValidationError):
        GroupCounts.model_validate({**group_counts(), **changes})


@pytest.mark.parametrize("labels", [(None, "Missing"), (1, "1"), (True, 1), (1, 1.0)])
def test_attribute_profile_preserves_distinct_category_types(labels):
    result = AttributeProfile(
        attribute="group", groups=[group_counts(label) for label in labels],
    )
    assert [type(item.group_value) for item in result.groups] == [
        type(label) for label in labels
    ]


@pytest.mark.parametrize("label", [None, "Missing", 1, True])
def test_attribute_profile_rejects_repeated_category(label):
    with pytest.raises(ValidationError, match="exactly once"):
        AttributeProfile(attribute="group", groups=[group_counts(label)] * 2)


@pytest.mark.parametrize("changes", [
    {"attribute": ""}, {"attribute": " \t"}, {"groups": []}, {"unknown": 1},
])
def test_attribute_profile_rejects_invalid_structure(changes):
    with pytest.raises(ValidationError):
        AttributeProfile.model_validate({
            "attribute": "group", "groups": [group_counts()], **changes,
        })


def test_dataset_profile_round_trips_through_json(profile_payload):
    result = DatasetProfile.model_validate(profile_payload)
    restored = DatasetProfile.model_validate_json(result.model_dump_json())
    assert restored.model_dump() == profile_payload


@pytest.mark.parametrize("changes", [
    {"row_count": 0}, {"column_count": 0}, {"positive_outcomes": -1},
    {"positive_outcomes": 5}, {"outcome_prevalence": 0.6},
    {"outcome_prevalence": float("nan")}, {"outcome_prevalence": float("inf")},
    {"score_available": "false"}, {"attributes": []},
    {"data_quality_flags": ["   "]}, {"unknown_field": 123},
])
def test_dataset_profile_rejects_invalid_summary(profile_payload, changes):
    with pytest.raises(ValidationError):
        DatasetProfile.model_validate({**profile_payload, **changes})


def test_dataset_profile_rejects_duplicate_attributes(profile_payload):
    profile_payload["attributes"] *= 2
    with pytest.raises(ValidationError, match="Attribute names must be unique"):
        DatasetProfile.model_validate(profile_payload)


def test_dataset_profile_rejects_too_many_attributes(profile_payload):
    second = deepcopy(profile_payload["attributes"][0])
    second["attribute"] = "other"
    profile_payload["attributes"].append(second)
    profile_payload["column_count"] = 1
    with pytest.raises(ValidationError, match="must not exceed the column count"):
        DatasetProfile.model_validate(profile_payload)


def test_dataset_profile_requires_every_row_per_attribute(profile_payload):
    profile_payload["attributes"][0]["groups"][0]["count"] = 3
    with pytest.raises(ValidationError, match="account for every row"):
        DatasetProfile.model_validate(profile_payload)


def test_dataset_profile_requires_every_positive_per_attribute(profile_payload):
    profile_payload["attributes"][0]["groups"][0]["positive_outcomes"] = 0
    with pytest.raises(ValidationError, match="account for every positive outcome"):
        DatasetProfile.model_validate(profile_payload)


def test_attribute_totals_are_not_summed_across_attributes(profile_payload):
    second = deepcopy(profile_payload["attributes"][0])
    second["attribute"] = "other"
    profile_payload["attributes"].append(second)
    assert len(DatasetProfile.model_validate(profile_payload).attributes) == 2


def test_prevalence_accepts_float_roundoff_but_not_display_rounding(profile_payload):
    profile_payload.update(row_count=3, positive_outcomes=1, outcome_prevalence=1 / 3)
    profile_payload["attributes"][0]["groups"] = [group_counts(count=3, positives=1)]
    assert DatasetProfile.model_validate(profile_payload).outcome_prevalence == 1 / 3
    profile_payload["outcome_prevalence"] = 0.33
    with pytest.raises(ValidationError, match="prevalence must match"):
        DatasetProfile.model_validate(profile_payload)
