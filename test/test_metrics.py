import numpy as np
import pandas as pd
import pytest
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
from pandas.testing import assert_frame_equal

from fairness.metrics import compute_fairness_gaps, compute_group_metrics


@pytest.fixture
def toy():
    return pd.DataFrame({
        "outcome": [1, 1, 0, 0, 1, 1, 0, 0],
        "prediction": [1, 0, 1, 0, 1, 1, 1, 0],
        "group": ["A"] * 4 + ["B"] * 4,
        "score": [0.9, 0.4, 0.7, 0.2, 0.8, 0.7, 0.6, 0.1],
    })


def test_metrics_match_hand_calculations(toy):
    result = compute_group_metrics(toy, "group", "score")
    expected = pd.DataFrame(
        {
            "count": [4.0, 4.0],
            "base_rate": [0.5, 0.5],
            "selection_rate": [0.5, 0.75],
            "tpr": [0.5, 1.0],
            "fpr": [0.5, 0.5],
            "ppv": [0.5, 2 / 3],
            "auroc": [0.75, 1.0],
        },
        index=pd.Index(["A", "B"], name="group"),
    )
    assert_frame_equal(result, expected, check_exact=False, atol=1e-6, rtol=0)


def test_score_is_optional(toy):
    with_score = compute_group_metrics(toy, "group", "score")
    without_score = compute_group_metrics(toy.drop(columns="score"), "group")
    assert "auroc" not in without_score
    assert_frame_equal(without_score, with_score.drop(columns="auroc"))


@pytest.mark.parametrize(
    ("outcomes", "predictions", "expected"),
    [
        ([1, 1, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0, np.nan, 1]),
        ([0, 0, 0, 0], [1, 0, 1, 0], [0, 0.5, np.nan, 0.5, 0, np.nan]),
        ([1, 1, 1, 1], [1, 0, 1, 0], [1, 0.5, 0.5, np.nan, 1, np.nan]),
    ],
)
def test_undefined_metrics_preserve_nan(outcomes, predictions, expected):
    df = pd.DataFrame({
        "outcome": outcomes,
        "prediction": predictions,
        "group": ["A"] * 4,
        "score": [0.4, 0.3, 0.2, 0.1],
    })
    result = compute_group_metrics(df, "group", "score")
    np.testing.assert_allclose(
        result.loc["A"].drop("count").to_numpy(),
        expected,
        atol=1e-6,
        rtol=0,
        equal_nan=True,
    )


def test_undefined_group_does_not_affect_other_groups(toy):
    toy.loc[toy["group"] == "A", "prediction"] = 0
    result = compute_group_metrics(toy, "group", "score")
    assert pd.isna(result.loc["A", "ppv"])
    assert result.loc["B", "ppv"] == pytest.approx(2 / 3)


def test_missing_group_is_not_dropped_or_merged_with_literal_missing(toy):
    toy["group"] = [None] * 4 + ["Missing"] * 4
    result = compute_group_metrics(toy, "group", "score")
    assert len(result) == 2
    assert result["count"].sum() == len(toy)
    assert result.loc[None, "selection_rate"] == 0.5
    assert result.loc["Missing", "selection_rate"] == 0.75


def test_unobserved_categories_are_excluded(toy):
    toy["group"] = pd.Categorical(toy["group"], categories=["A", "B", "C"])
    result = compute_group_metrics(toy, "group")
    assert result.index.tolist() == ["A", "B"]


def test_numeric_group_codes_are_supported(toy):
    toy["group"] = toy["group"].map({"A": 10, "B": 20})
    result = compute_group_metrics(toy, "group")
    assert result.loc[10, "tpr"] == 0.5
    assert result.loc[20, "tpr"] == 1.0


def test_input_is_not_modified(toy):
    before = toy.copy(deep=True)
    compute_group_metrics(toy, "group", "score")
    assert_frame_equal(toy, before)


@pytest.mark.parametrize("column", ["outcome", "prediction", "score", "group"])
def test_missing_columns_rejected(toy, column):
    with pytest.raises(ValueError, match="Missing mapped columns"):
        compute_group_metrics(toy.drop(columns=column), "group", "score")


@pytest.mark.parametrize("column", ["outcome", "prediction"])
def test_invalid_labels_rejected(toy, column):
    toy.loc[0, column] = 2
    with pytest.raises(ValueError, match="only 0 or 1"):
        compute_group_metrics(toy, "group", "score")


@pytest.mark.parametrize("invalid_score", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_scores_rejected(toy, invalid_score):
    toy.loc[0, "score"] = invalid_score
    with pytest.raises(ValueError, match="Score"):
        compute_group_metrics(toy, "group", "score")


def test_empty_dataset_rejected(toy):
    with pytest.raises(ValueError, match="at least one data row"):
        compute_group_metrics(toy.iloc[:0], "group", "score")


def test_fairness_gaps_match_hand_calculations_and_references(toy):
    result = compute_fairness_gaps(compute_group_metrics(toy, "group", "score"))
    gaps = result["gaps"]
    expected = {
        "demographic_parity_difference": 0.25,
        "tpr_gap": 0.5,
        "fpr_gap": 0.0,
        "equalized_odds_difference": 0.5,
        "ppv_difference": 2 / 3 - 1 / 2,
    }
    for metric, value in expected.items():
        assert gaps.loc[metric, "value"] == pytest.approx(value, abs=1e-6)
        assert gaps.loc[metric, "reason"] is None
    kwargs = {
        "y_true": toy["outcome"], "y_pred": toy["prediction"],
        "sensitive_features": toy["group"],
    }
    assert gaps.loc["demographic_parity_difference", "value"] == pytest.approx(
        demographic_parity_difference(**kwargs), abs=1e-6,
    )
    assert gaps.loc["equalized_odds_difference", "value"] == pytest.approx(
        equalized_odds_difference(**kwargs), abs=1e-6,
    )
    assert result["disparate_impact_ratio"]["value"] == pytest.approx(
        demographic_parity_ratio(**kwargs), abs=1e-6,
    )
    assert result["equalized_odds_component"] == "tpr"
    assert gaps.loc["demographic_parity_difference", "min_group"] == "A"
    assert gaps.loc["demographic_parity_difference", "max_group"] == "B"


def test_reference_changes_signed_differences_but_not_pairwise_gaps(toy):
    table = compute_group_metrics(toy, "group", "score")
    a = compute_fairness_gaps(table)
    b = compute_fairness_gaps(table, reference_group="B")
    assert a["reference_group"] == "A"
    assert a["reference_group_method"] == "largest_group"
    assert b["reference_group_method"] == "user_selected"
    assert_frame_equal(a["gaps"], b["gaps"])
    assert a["reference_differences"].loc["B", "selection_rate"] == 0.25
    assert b["reference_differences"].loc["A", "selection_rate"] == -0.25
    assert b["reference_differences"].loc["B", "selection_rate"] == 0.0


def test_reference_defaults_to_largest_group(toy):
    df = pd.concat([toy, toy[toy["group"] == "B"]], ignore_index=True)
    result = compute_fairness_gaps(compute_group_metrics(df, "group"))
    assert result["reference_group"] == "B"


def test_unknown_reference_rejected(toy):
    with pytest.raises(ValueError, match="Reference group"):
        compute_fairness_gaps(compute_group_metrics(toy, "group"), "absent")


@pytest.mark.parametrize("number_of_groups", [0, 1])
def test_fewer_than_two_groups_make_all_gaps_unavailable(toy, number_of_groups):
    table = compute_group_metrics(toy, "group").iloc[:number_of_groups]
    result = compute_fairness_gaps(table)
    assert result["gaps"]["value"].isna().all()
    assert result["gaps"]["reason"].notna().all()
    assert pd.isna(result["disparate_impact_ratio"]["value"])
    assert "two eligible groups" in result["disparate_impact_ratio"]["reason"]
    if number_of_groups == 0:
        assert result["reference_group_method"] == "unavailable"
        assert result["reference_differences"].empty


def test_one_undefined_rate_is_not_silently_skipped(toy):
    # Add a third group: A and B remain defined, but C has undefined PPV.
    extra = toy.iloc[:4].copy()
    extra["group"] = "C"
    extra["prediction"] = 0
    df = pd.concat([toy, extra], ignore_index=True)
    result = compute_fairness_gaps(compute_group_metrics(df, "group"))
    assert pd.isna(result["gaps"].loc["ppv_difference", "value"])
    assert "undefined" in result["gaps"].loc["ppv_difference", "reason"]
    assert result["gaps"].loc["demographic_parity_difference", "value"] == 0.75
    assert "No positive predictions" in result["undefined_metric_reasons"].loc["C", "ppv"]


@pytest.mark.parametrize("outcome, missing_metric", [(0, "tpr"), (1, "fpr")])
def test_eod_requires_both_component_gaps(toy, outcome, missing_metric):
    toy.loc[toy["group"] == "A", "outcome"] = outcome
    result = compute_fairness_gaps(compute_group_metrics(toy, "group", "score"))
    assert pd.isna(result["gaps"].loc[f"{missing_metric}_gap", "value"])
    assert pd.isna(result["gaps"].loc["equalized_odds_difference", "value"])
    assert result["equalized_odds_component"] is None
    assert result["undefined_metric_reasons"].loc["A", missing_metric]
    assert result["undefined_metric_reasons"].loc["A", "auroc"]


def test_maximum_pairwise_gap_can_exclude_reference_and_eod_use_fpr(toy):
    table = compute_group_metrics(toy, "group")
    table.loc["C"] = table.loc["A"]
    table["selection_rate"] = [0.5, 0.1, 0.9]
    table["tpr"] = [0.5, 0.5, 0.5]
    table["fpr"] = [0.5, 0.1, 0.9]
    result = compute_fairness_gaps(table, "A")
    assert result["gaps"].loc["demographic_parity_difference", "value"] == pytest.approx(0.8)
    assert result["gaps"].loc["demographic_parity_difference", "min_group"] == "B"
    assert result["gaps"].loc["demographic_parity_difference", "max_group"] == "C"
    assert result["equalized_odds_component"] == "fpr"
    assert result["gaps"].loc["equalized_odds_difference", "value"] == pytest.approx(0.8)


def test_equal_rates_have_zero_gaps_and_ratio_one(toy):
    toy.loc[toy["group"] == "B", "prediction"] = [1, 0, 1, 0]
    result = compute_fairness_gaps(compute_group_metrics(toy, "group"))
    assert (result["gaps"]["value"] == 0).all()
    assert result["disparate_impact_ratio"]["value"] == 1


def test_zero_selection_rates_do_not_produce_ratio_one(toy):
    toy["prediction"] = 0
    result = compute_fairness_gaps(compute_group_metrics(toy, "group"))
    assert result["gaps"].loc["demographic_parity_difference", "value"] == 0
    assert pd.isna(result["disparate_impact_ratio"]["value"])
    assert "0/0" in result["disparate_impact_ratio"]["reason"]


def test_missing_group_can_be_explicit_reference(toy):
    toy["group"] = [None] * 4 + ["Missing"] * 4
    result = compute_fairness_gaps(compute_group_metrics(toy, "group"), None)
    assert result["reference_group"] is None
    assert result["reference_group_method"] == "user_selected"
    assert result["reference_differences"].loc["Missing", "selection_rate"] == 0.25


def test_gap_computation_does_not_modify_input(toy):
    table = compute_group_metrics(toy, "group", "score")
    before = table.copy(deep=True)
    compute_fairness_gaps(table)
    assert_frame_equal(table, before)


def test_numeric_groups_keep_missing_reference_distinct(toy):
    toy["group"] = [None] * 4 + [10] * 4
    table = compute_group_metrics(toy, "group")
    result = compute_fairness_gaps(table, None)
    assert result["reference_group"] is None
    assert result["reference_differences"].loc[10, "selection_rate"] == 0.25


@pytest.mark.parametrize("invalid_value", [np.inf, -0.1, 1.1])
def test_invalid_aggregate_rates_rejected(toy, invalid_value):
    table = compute_group_metrics(toy, "group")
    table.loc["A", "tpr"] = invalid_value
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        compute_fairness_gaps(table)


def test_duplicate_aggregate_groups_rejected(toy):
    table = compute_group_metrics(toy, "group")
    table.index = ["A", "A"]
    with pytest.raises(ValueError, match="unique"):
        compute_fairness_gaps(table)


def test_missing_aggregate_column_rejected(toy):
    table = compute_group_metrics(toy, "group").drop(columns="ppv")
    with pytest.raises(ValueError, match="Missing metric columns"):
        compute_fairness_gaps(table)
