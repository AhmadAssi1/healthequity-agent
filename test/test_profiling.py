"""Profiling checks against hand-counted data and the existing data contract."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.profiling import profile_dataset
from app.schemas import ColumnMapping
from app.tool_schemas import DatasetProfile


@pytest.fixture
def sample():
    return pd.DataFrame({
        "actual": [1, 0, 0, 1],
        "predicted": [1, 0, 1, 0],
        "gender": ["Female", "Male", "Female", None],
    })


@pytest.fixture
def mapping():
    return ColumnMapping(outcome="actual", prediction="predicted", groups=["gender"])


def test_profile_matches_hand_counted_example(sample, mapping):
    result = profile_dataset(sample, mapping)
    assert result.model_dump() == {
        "row_count": 4,
        "column_count": 3,
        "positive_outcomes": 2,
        "outcome_prevalence": 0.5,
        "score_available": False,
        "attributes": [{
            "attribute": "gender",
            "groups": [
                {"group_value": "Female", "count": 2, "positive_outcomes": 1},
                {"group_value": "Male", "count": 1, "positive_outcomes": 0},
                {"group_value": None, "count": 1, "positive_outcomes": 1},
            ],
        }],
        "data_quality_flags": [
            "Missing values in 'gender' form a separate group.",
        ],
    }


def test_profile_preserves_input_and_handles_duplicate_index(sample, mapping):
    sample.index = [7, 7, 9, 9]
    original = sample.copy(deep=True)
    first = profile_dataset(sample, mapping)
    second = profile_dataset(sample, mapping)
    assert first == second
    assert_frame_equal(sample, original)
    assert DatasetProfile.model_validate_json(first.model_dump_json()) == first


def test_no_warning_for_clean_two_class_data(sample, mapping):
    sample.loc[3, "gender"] = "Male"
    assert profile_dataset(sample, mapping).data_quality_flags == []


def test_profile_accounts_for_all_rows_in_multiple_attributes(sample, mapping):
    sample["age_band"] = ["young", "old", "old", "young"]
    mapping.groups = ["gender", "age_band"]
    result = profile_dataset(sample, mapping)
    assert [item.attribute for item in result.attributes] == ["gender", "age_band"]
    assert [(g.group_value, g.count, g.positive_outcomes)
            for g in result.attributes[1].groups] == [("young", 2, 2), ("old", 2, 0)]


def test_missing_categories_are_merged_but_literal_missing_is_distinct(sample, mapping):
    sample["gender"] = pd.Series([None, np.nan, pd.NA, "Missing"], dtype=object)
    result = profile_dataset(sample, mapping)
    assert [(g.group_value, g.count, g.positive_outcomes)
            for g in result.attributes[0].groups] == [(None, 3, 1), ("Missing", 1, 1)]


def test_all_missing_attribute_is_retained(sample, mapping):
    sample["gender"] = None
    result = profile_dataset(sample, mapping)
    assert len(result.attributes[0].groups) == 1
    assert result.attributes[0].groups[0].count == 4
    assert result.attributes[0].groups[0].positive_outcomes == 2


def test_unused_categorical_levels_do_not_create_empty_groups(sample, mapping):
    sample["gender"] = pd.Categorical(
        ["A", "B", "A", None], categories=["A", "B", "unused"],
    )
    result = profile_dataset(sample, mapping)
    assert [g.group_value for g in result.attributes[0].groups] == ["A", "B", None]


def test_mixed_typed_labels_remain_separate(sample, mapping):
    sample["gender"] = pd.Series([1, "1", True, 1.0], dtype=object)
    result = profile_dataset(sample, mapping)
    groups = result.attributes[0].groups
    assert [type(g.group_value) for g in groups] == [int, str, bool, float]
    assert [g.count for g in groups] == [1, 1, 1, 1]
    assert [g.positive_outcomes for g in groups] == [1, 0, 0, 1]


def test_numpy_category_scalars_are_serializable(sample, mapping):
    sample["gender"] = pd.Series(
        [np.int64(1), np.float64(2.5), np.bool_(True), np.str_("A")], dtype=object,
    )
    result = profile_dataset(sample, mapping)
    assert [type(g.group_value) for g in result.attributes[0].groups] == [
        int, float, bool, str,
    ]
    assert DatasetProfile.model_validate_json(result.model_dump_json()) == result


def test_duplicate_audit_rows_warn_without_dropping_rows(sample, mapping):
    sample = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    sample["unmapped_id"] = range(len(sample))
    original = sample.copy(deep=True)
    result = profile_dataset(sample, mapping)
    assert result.row_count == 5
    assert result.positive_outcomes == 3
    assert any("Duplicate rows" in flag for flag in result.data_quality_flags)
    assert_frame_equal(sample, original)


def test_missing_predictions_can_be_profiled_but_are_flagged(sample):
    result = profile_dataset(
        sample.drop(columns="predicted"),
        ColumnMapping(outcome="actual", groups=["gender"]),
    )
    assert result.row_count == 4
    assert any("Predictions are not mapped" in flag for flag in result.data_quality_flags)


@pytest.mark.parametrize("positive_label", [0, 1])
def test_single_outcome_class_is_profiled_and_flagged(sample, mapping, positive_label):
    sample["actual"] = positive_label
    result = profile_dataset(sample, mapping)
    assert result.outcome_prevalence == positive_label
    assert result.positive_outcomes == positive_label * len(sample)
    assert any("only one class" in flag for flag in result.data_quality_flags)


def test_float_binary_labels_are_accepted(sample, mapping):
    sample[["actual", "predicted"]] = sample[["actual", "predicted"]].astype(float)
    result = profile_dataset(sample, mapping)
    assert result.positive_outcomes == 2
    assert all(type(g.positive_outcomes) is int for g in result.attributes[0].groups)


def test_mapped_score_is_validated_and_marked_available(sample, mapping):
    sample["risk"] = [0.9, 0.1, 0.8, 0.2]
    mapping.score = "risk"
    assert profile_dataset(sample, mapping).score_available is True


def test_unmapped_score_is_not_assumed_to_be_a_probability(sample, mapping):
    sample["score"] = "unmapped text"
    assert profile_dataset(sample, mapping).score_available is False


def test_raw_unmapped_values_are_not_copied_into_profile(sample, mapping):
    sample["note"] = "PRIVATE_EXAMPLE_DO_NOT_SEND"
    result = profile_dataset(sample, mapping)
    assert result.column_count == 4
    assert "PRIVATE_EXAMPLE_DO_NOT_SEND" not in result.model_dump_json()
    assert "predicted" not in result.model_dump()


@pytest.mark.parametrize("column, values", [
    ("actual", [1, 0, None, 1]), ("predicted", [1, 0, None, 1]),
    ("actual", [1, 0, 2, 1]), ("predicted", [1, 0, 0.5, 1]),
    ("actual", [True, False, False, True]),
    ("actual", ["1", "0", "0", "1"]),
])
def test_invalid_labels_stop_profiling(sample, mapping, column, values):
    sample[column] = values
    with pytest.raises(ValueError):
        profile_dataset(sample, mapping)


@pytest.mark.parametrize("bad_score", [None, -0.1, 1.1, float("inf"), "bad"])
def test_invalid_mapped_scores_stop_profiling(sample, mapping, bad_score):
    sample["risk"] = [bad_score] * 4
    mapping.score = "risk"
    with pytest.raises(ValueError):
        profile_dataset(sample, mapping)


def test_empty_data_stops_before_aggregation(sample, mapping):
    with pytest.raises(ValueError, match="at least one data row"):
        profile_dataset(sample.iloc[:0], mapping)


def test_missing_mapped_column_stops_profiling(sample, mapping):
    with pytest.raises(ValueError, match="Missing mapped columns"):
        profile_dataset(sample.drop(columns="gender"), mapping)


def test_duplicate_column_names_stop_profiling(sample, mapping):
    sample.columns = ["actual", "gender", "gender"]
    with pytest.raises(ValueError, match="Column names must be unique"):
        profile_dataset(sample, mapping)


@pytest.mark.parametrize("label", [("A", "B"), datetime(2026, 1, 1), float("inf")])
def test_unsupported_categories_are_rejected(sample, mapping, label):
    sample["gender"] = pd.Series([label] * 4, dtype=object)
    with pytest.raises(ValueError):
        profile_dataset(sample, mapping)
