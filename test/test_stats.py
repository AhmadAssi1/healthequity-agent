import pandas as pd
import pytest

from fairness.metrics import compute_fairness_gaps, compute_group_metrics
from fairness.stats import (
    bootstrap_fairness_intervals,
    compare_proportions,
    two_proportion_z_test,
)


def test_two_proportion_z_matches_notebook():
    z, p_value = two_proportion_z_test(40, 100, 60, 100)
    assert z == pytest.approx(-2.8284271247)
    assert p_value == pytest.approx(0.004677734981)

    result = compare_proportions([40, 60], [100, 100])
    assert result.method == "two_proportion_z"
    assert result.statistic == z
    assert result.p_value == p_value


def test_chi_square_matches_notebook():
    result = compare_proportions([40, 60, 50], [100, 100, 100])
    assert result.method == "chi_square"
    assert result.statistic == pytest.approx(8)
    assert result.p_value == pytest.approx(0.01831563889)


def test_sparse_tpr_table_routes_to_exact_fisher():
    result = compare_proportions([6, 8], [14, 26])
    assert result.method == "fisher_exact"
    assert result.min_expected_count == pytest.approx(4.9)
    assert result.p_value == pytest.approx(0.5013, abs=0.00005)


def test_expected_count_of_five_does_not_trigger_fisher():
    result = compare_proportions([4, 6], [10, 10])
    assert result.min_expected_count == 5
    assert result.method == "two_proportion_z"


def test_sparse_three_group_fisher_is_seeded_and_uses_all_groups():
    result = compare_proportions([0, 0, 4], [5, 5, 5], seed=42)
    assert result.method == "fisher_monte_carlo"
    assert result.p_value == pytest.approx(15 / 1365, abs=0.005)
    assert result == compare_proportions([0, 0, 4], [5, 5, 5], seed=42)


def test_single_outcome_has_no_significance_result():
    result = compare_proportions([0, 0], [10, 20])
    assert result.method == "unavailable"
    assert result.p_value is None
    assert result.reason


@pytest.mark.parametrize(
    ("counts", "totals"),
    [([1], [5]), ([1, 2], [5]), ([2, 1], [0, 5]), ([6, 1], [5, 5]), ([1.5, 1], [5, 5])],
)
def test_invalid_count_tables_raise(counts, totals):
    with pytest.raises(ValueError):
        compare_proportions(counts, totals)


@pytest.fixture
def bootstrap_example():
    rows = []
    for group, counts in (("A", [20, 20, 20, 40]), ("B", [32, 8, 28, 32])):
        for (outcome, prediction), count in zip(
            [(1, 1), (1, 0), (0, 1), (0, 0)], counts,
        ):
            rows.extend([(group, outcome, prediction)] * count)
    return pd.DataFrame(rows, columns=["group", "outcome", "prediction"])


def test_bootstrap_points_match_metric_engine_and_intervals_repeat(bootstrap_example):
    points = compute_fairness_gaps(
        compute_group_metrics(bootstrap_example, "group"),
    )["gaps"]["value"]
    intervals = bootstrap_fairness_intervals(bootstrap_example, "group")
    assert intervals == bootstrap_fairness_intervals(bootstrap_example, "group")
    for metric, interval in intervals.items():
        assert interval.observed == pytest.approx(points.loc[metric])
        assert interval.valid_resamples == interval.total_resamples == 1000
        assert 0 <= interval.lower < interval.upper <= 1


def test_bootstrap_preserves_group_membership_and_paired_labels():
    df = pd.DataFrame({
        "group": ["A"] * 10 + [None] * 10,
        "outcome": [0, 1] * 10,
        "prediction": [0] * 10 + [0, 1] * 5,
    })
    intervals = bootstrap_fairness_intervals(df, "group")
    dpd = intervals["demographic_parity_difference"]
    assert dpd.observed == 0.5
    assert dpd.valid_resamples == 1000
    assert dpd.lower < 0.5 < dpd.upper
    eod = intervals["equalized_odds_difference"]
    assert eod.observed == eod.lower == eod.upper == 1.0
    ppv = intervals["ppv_difference"]
    assert ppv.observed is ppv.lower is ppv.upper is None
    assert ppv.valid_resamples == 0


def test_bootstrap_tracks_partially_undefined_resamples():
    df = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "outcome": [0, 1, 0, 1],
        "prediction": [0, 1, 0, 1],
    })
    intervals = bootstrap_fairness_intervals(df, "group")
    assert intervals["demographic_parity_difference"].valid_resamples == 1000
    eod = intervals["equalized_odds_difference"]
    assert 0 < eod.valid_resamples < 1000
    assert eod.lower == eod.upper == 0


def test_bootstrap_requires_two_groups(bootstrap_example):
    with pytest.raises(ValueError, match="two eligible groups"):
        bootstrap_fairness_intervals(bootstrap_example.iloc[:100], "group")
