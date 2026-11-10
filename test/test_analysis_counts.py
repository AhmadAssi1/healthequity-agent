"""Confusion-count correctness, policy integration, and schema invariants."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError
from sklearn.metrics import confusion_matrix

from app.analysis_counts import compute_analysis_counts
from app.schemas import ColumnMapping
from app.tool_schemas import (
    AttributeAnalysisCounts,
    ConfusionCounts,
    GroupAnalysisCounts,
)
from fairness.policy import SuppressionPolicy


@pytest.fixture
def mapping():
    return ColumnMapping(outcome="actual", prediction="predicted", groups=["group"])


@pytest.fixture
def permissive_policy():
    return SuppressionPolicy(min_group_size=1, min_positive_outcomes=0)


def frame_for_counts(specs):
    rows = []
    for label, counts in specs:
        for (outcome, prediction), count in zip(
            [(0, 0), (0, 1), (1, 0), (1, 1)], counts, strict=True,
        ):
            rows.extend([(outcome, prediction, label)] * count)
    return pd.DataFrame(rows, columns=["actual", "predicted", "group"])


def group_payload():
    return {
        "group_value": "A", "count": 50, "positive_outcomes": 10,
        "confusion": {"tn": 35, "fp": 5, "fn": 0, "tp": 10},
    }


def test_hand_counted_cells_are_not_swapped(mapping, permissive_policy):
    df = frame_for_counts([("A", [7, 3, 2, 5]), ("B", [1, 4, 6, 8])])
    result = compute_analysis_counts(df, mapping, permissive_policy)
    assert result[0].attribute == "group"
    assert [g.confusion.model_dump() for g in result[0].groups] == [
        {"tn": 7, "fp": 3, "fn": 2, "tp": 5},
        {"tn": 1, "fp": 4, "fn": 6, "tp": 8},
    ]
    assert [(g.count, g.positive_outcomes) for g in result[0].groups] == [(17, 7), (19, 14)]


@pytest.mark.parametrize("seed", [0, 19, 42])
def test_randomized_counts_match_sklearn_reference(mapping, permissive_policy, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "actual": rng.integers(0, 2, 200),
        "predicted": rng.integers(0, 2, 200),
        "group": rng.choice(["A", "B", "C"], 200),
    })
    result = compute_analysis_counts(df, mapping, permissive_policy)
    for group in result[0].groups:
        subset = df.loc[df["group"] == group.group_value]
        tn, fp, fn, tp = confusion_matrix(
            subset["actual"], subset["predicted"], labels=[0, 1],
        ).ravel()
        assert group.confusion.model_dump() == {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }


def test_suppression_filters_counts_at_policy_boundaries(mapping):
    df = frame_for_counts([
        ("eligible", [35, 5, 0, 10]),
        ("small", [35, 4, 0, 10]),
        ("few_positives", [36, 5, 0, 9]),
    ])
    result = compute_analysis_counts(df, mapping, SuppressionPolicy())
    assert [g.group_value for g in result[0].groups] == ["eligible"]
    assert result[0].groups[0].confusion.model_dump() == {
        "tn": 35, "fp": 5, "fn": 0, "tp": 10,
    }


def test_all_suppressed_attribute_is_retained_with_empty_groups(mapping):
    df = frame_for_counts([("A", [1, 1, 1, 1])])
    result = compute_analysis_counts(df, mapping, SuppressionPolicy())
    assert [item.model_dump() for item in result] == [{"attribute": "group", "groups": []}]


def test_default_policy_comes_from_configuration_loader(mapping, monkeypatch):
    monkeypatch.setattr(
        "app.group_checks.load_suppression_policy",
        lambda: SuppressionPolicy(min_group_size=1, min_positive_outcomes=0),
    )
    result = compute_analysis_counts(frame_for_counts([("A", [1, 0, 0, 0])]), mapping)
    assert result[0].groups[0].confusion.tn == 1


def test_missing_and_literal_missing_keep_separate_cells(mapping, permissive_policy):
    df = pd.DataFrame({
        "actual": [0, 1, 0, 1], "predicted": [0, 1, 1, 0],
        "group": pd.Series([None, np.nan, pd.NA, "Missing"], dtype=object),
    })
    groups = compute_analysis_counts(df, mapping, permissive_policy)[0].groups
    assert [g.group_value for g in groups] == [None, "Missing"]
    assert groups[0].confusion.model_dump() == {"tn": 1, "fp": 1, "fn": 0, "tp": 1}
    assert groups[1].confusion.model_dump() == {"tn": 0, "fp": 0, "fn": 1, "tp": 0}


def test_typed_categories_are_counted_separately(mapping, permissive_policy):
    df = pd.DataFrame({
        "actual": [0, 0, 1, 1], "predicted": [0, 1, 0, 1],
        "group": pd.Series([1, "1", True, 1.0], dtype=object),
    })
    groups = compute_analysis_counts(df, mapping, permissive_policy)[0].groups
    assert [type(g.group_value) for g in groups] == [int, str, bool, float]
    assert [g.confusion.model_dump() for g in groups] == [
        {"tn": 1, "fp": 0, "fn": 0, "tp": 0},
        {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
        {"tn": 0, "fp": 0, "fn": 1, "tp": 0},
        {"tn": 0, "fp": 0, "fn": 0, "tp": 1},
    ]


def test_numpy_labels_are_normalized_consistently(mapping, permissive_policy):
    df = frame_for_counts([("A", [1, 1, 1, 1])])
    df["group"] = pd.Series([np.int64(7)] * 4, dtype=object)
    group = compute_analysis_counts(df, mapping, permissive_policy)[0].groups[0]
    assert type(group.group_value) is int
    assert group.confusion.model_dump() == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_counts_preserve_duplicate_rows_index_and_input(mapping, permissive_policy):
    df = frame_for_counts([("A", [2, 3, 1, 4])])
    df.index = [7] * len(df)
    original = df.copy(deep=True)
    result = compute_analysis_counts(df, mapping, permissive_policy)
    assert_frame_equal(df, original)
    assert result[0].groups[0].count == 10
    assert result[0].groups[0].confusion.model_dump() == {"tn": 2, "fp": 3, "fn": 1, "tp": 4}
    assert AttributeAnalysisCounts.model_validate_json(result[0].model_dump_json()) == result[0]


def test_multiple_attributes_are_computed_independently(mapping, permissive_policy):
    df = frame_for_counts([("A", [1, 1, 1, 1])])
    df["other"] = ["X", "Y", "X", "Y"]
    mapping.groups = ["group", "other"]
    result = compute_analysis_counts(df, mapping, permissive_policy)
    assert [item.attribute for item in result] == ["group", "other"]
    assert [g.confusion.model_dump() for g in result[1].groups] == [
        {"tn": 1, "fp": 0, "fn": 1, "tp": 0},
        {"tn": 0, "fp": 1, "fn": 0, "tp": 1},
    ]


@pytest.mark.parametrize("labels", [[0, 0, 0, 0], [1, 1, 1, 1]])
def test_single_outcome_class_is_representable(mapping, permissive_policy, labels):
    df = pd.DataFrame({"actual": labels, "predicted": [0, 1, 0, 1], "group": ["A"] * 4})
    result = compute_analysis_counts(df, mapping, permissive_policy)[0].groups[0]
    assert result.positive_outcomes == sum(labels)
    assert sum(result.confusion.model_dump().values()) == 4


def test_missing_prediction_mapping_is_rejected(permissive_policy):
    df = frame_for_counts([("A", [1, 1, 1, 1])])
    with pytest.raises(ValueError, match="prediction column must be mapped"):
        compute_analysis_counts(
            df, ColumnMapping(outcome="actual", groups=["group"]), permissive_policy,
        )


@pytest.mark.parametrize("column, values", [
    ("actual", [1, 0, 2, 1]), ("predicted", [0, 1, None, 0]),
    ("predicted", ["0", "1", "0", "1"]),
    ("actual", [True, False, False, True]),
])
def test_invalid_data_stops_before_counting(mapping, permissive_policy, column, values):
    df = frame_for_counts([("A", [1, 1, 1, 1])])
    df[column] = values
    with pytest.raises(ValueError):
        compute_analysis_counts(df, mapping, permissive_policy)


@pytest.mark.parametrize("field", ["tn", "fp", "fn", "tp"])
@pytest.mark.parametrize("invalid", [-1, True, 1.5, "1"])
def test_confusion_counts_rejects_invalid_cell_types_and_values(field, invalid):
    with pytest.raises(ValidationError):
        ConfusionCounts(**{**group_payload()["confusion"], field: invalid})


def test_confusion_counts_rejects_empty_group():
    with pytest.raises(ValidationError, match="at least one observation"):
        ConfusionCounts(tn=0, fp=0, fn=0, tp=0)


def test_group_analysis_rejects_inconsistent_total():
    with pytest.raises(ValidationError, match="sum to the group count"):
        GroupAnalysisCounts(**{**group_payload(), "count": 51})


def test_group_analysis_rejects_inconsistent_positive_count():
    with pytest.raises(ValidationError, match="TP plus FN"):
        GroupAnalysisCounts(**{**group_payload(), "positive_outcomes": 9})


def test_attribute_analysis_rejects_duplicate_groups():
    with pytest.raises(ValidationError, match="exactly one set"):
        AttributeAnalysisCounts(attribute="group", groups=[group_payload()] * 2)


@pytest.mark.parametrize("model, payload", [
    (ConfusionCounts, {"tn": 1, "fp": 0, "fn": 0, "tp": 0}),
    (GroupAnalysisCounts, group_payload()),
    (AttributeAnalysisCounts, {"attribute": "group", "groups": []}),
])
def test_analysis_schemas_forbid_unknown_fields(model, payload):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate({**payload, "invented": 123})
