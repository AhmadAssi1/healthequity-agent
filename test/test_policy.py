import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from fairness.policy import (
    SeverityPolicy,
    SuppressionPolicy,
    check_group_eligibility,
    check_group_sizes,
    classify_severity,
    filter_reportable_groups,
    load_severity_policy,
    load_suppression_policy,
)


@pytest.mark.parametrize(
    ("count", "positives", "expected_reportable", "expected_failures"),
    [
        (49, 10, False, 1),
        (50, 9, False, 1),
        (49, 9, False, 2),
        (50, 10, True, 0),
        (51, 11, True, 0),
    ],
)
def test_group_eligibility_boundaries(
    count,
    positives,
    expected_reportable,
    expected_failures,
):
    result = check_group_eligibility(
        count,
        positives,
        SuppressionPolicy(),
    )

    assert result["reportable"] is expected_reportable

    reason = result["suppression_reason"]

    if expected_reportable:
        assert pd.isna(reason)
    else:
        assert reason.count("below minimum") == expected_failures


def test_custom_thresholds():
    policy = SuppressionPolicy(
        min_group_size=4,
        min_positive_outcomes=2,
    )

    result = check_group_eligibility(4, 2, policy)

    assert result["reportable"] is True
    assert result["suppression_reason"] is None


def test_group_sizes_preserve_missing_groups_and_input():
    df = pd.DataFrame({
        "group": [None] * 50 + ["Missing"] * 49,
        "outcome": (
            [1] * 10 + [0] * 40
            + [1] * 9 + [0] * 40
        ),
    })
    original = df.copy(deep=True)

    result = check_group_sizes(df, "group")

    assert len(result) == 2

    assert result.loc[None, "count"] == 50
    assert result.loc[None, "positive_outcomes"] == 10
    assert bool(result.loc[None, "reportable"]) is True
    assert pd.isna(result.loc[None, "suppression_reason"])

    assert result.loc["Missing", "count"] == 49
    assert result.loc["Missing", "positive_outcomes"] == 9
    assert bool(result.loc["Missing", "reportable"]) is False
    assert (
        result.loc["Missing", "suppression_reason"]
        .count("below minimum") == 2
    )

    assert_frame_equal(df, original)


@pytest.mark.parametrize(
    ("count", "positives"),
    [
        (-1, 0),   # Negative group size
        (50, -1),  # Negative positive count
        (50, 51),  # More positives than records
    ],
)
def test_invalid_counts_rejected(count, positives):
    with pytest.raises(ValueError):
        check_group_eligibility(
            count,
            positives,
            SuppressionPolicy(),
        )


@pytest.mark.parametrize(
    ("gap", "lower", "upper", "expected"),
    [
        (0.15, 0.08, 0.22, "substantial"),
        (0.15, 0.00, 0.25, "watch"),
        (0.08, 0.02, 0.14, "watch"),
        (0.03, 0.00, 0.07, "no_evidence_found"),
        (0.10, 0.04, 0.16, "watch"),
        (0.05, 0.00, 0.10, "no_evidence_found"),
        (None, None, None, "unavailable"),
        (0.15, None, None, "unavailable"),
        (float("nan"), 0.00, 0.20, "unavailable"),
    ],
)
def test_severity_classification(gap, lower, upper, expected):
    result = classify_severity(
        gap,
        lower,
        upper,
        SeverityPolicy(),
    )

    assert result == expected


@pytest.mark.parametrize(
    ("watch", "substantial"),
    [
        (-0.01, 0.10),
        (0.05, 1.01),
        (0.10, 0.10),
        (0.20, 0.10),
    ],
)
def test_invalid_severity_thresholds(watch, substantial):
    with pytest.raises(ValidationError):
        SeverityPolicy(
            watch_gap=watch,
            substantial_gap=substantial,
        )


def test_custom_severity_thresholds():
    custom = SeverityPolicy(
        watch_gap=0.10,
        substantial_gap=0.20,
    )

    assert classify_severity(
        0.15, 0.08, 0.22, SeverityPolicy()
    ) == "substantial"

    assert classify_severity(
        0.15, 0.08, 0.22, custom
    ) == "watch"


@pytest.mark.parametrize(
    ("gap", "lower", "upper"),
    [
        (-0.10, 0.00, 0.20),        # Negative gap
        (1.10, 0.00, 1.00),         # Gap exceeds 1
        (0.15, -0.01, 0.25),        # Negative lower bound
        (0.15, 0.05, 1.01),         # Upper bound exceeds 1
        (0.15, 0.25, 0.05),         # Reversed bounds
        (float("inf"), 0.05, 0.25), # Infinite gap
    ],
)
def test_invalid_severity_inputs(gap, lower, upper):
    with pytest.raises(ValueError):
        classify_severity(
            gap,
            lower,
            upper,
            SeverityPolicy(),
        )


def test_load_policies_from_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "fairness:\n"
        "  min_group_size: 80\n"
        "  min_positive_outcomes: 15\n"
        "severity:\n"
        "  watch_gap: 0.08\n"
        "  substantial_gap: 0.20\n",
        encoding="utf-8",
    )

    suppression = load_suppression_policy(config_path)
    severity = load_severity_policy(config_path)

    assert suppression.min_group_size == 80
    assert suppression.min_positive_outcomes == 15
    assert severity.watch_gap == 0.08
    assert severity.substantial_gap == 0.20

def test_filter_keeps_only_reportable_groups():
    df = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 49 + [None] * 50,
        "outcome": (
            [1] * 10 + [0] * 40
            + [1] * 10 + [0] * 39
            + [1] * 10 + [0] * 40
        ),
    })
    original = df.copy(deep=True)

    filtered, eligibility = filter_reportable_groups(df, "group")

    expected = df.loc[
        df["group"].eq("A") | df["group"].isna()
    ]

    assert_frame_equal(filtered, expected)
    assert_frame_equal(df, original)

    assert len(filtered) == 100
    assert not filtered["group"].eq("B").any()
    assert bool(eligibility.loc["B", "reportable"]) is False
    assert pd.notna(eligibility.loc["B", "suppression_reason"])

def test_filter_when_all_groups_are_suppressed():
    df = pd.DataFrame({
        "group": ["A"] * 4 + ["B"] * 4,
        "outcome": [1, 1, 0, 0] * 2,
    })

    filtered, eligibility = filter_reportable_groups(df, "group")

    assert filtered.empty
    assert filtered.columns.equals(df.columns)
    assert len(eligibility) == 2
    assert not eligibility["reportable"].any()
    assert eligibility["suppression_reason"].notna().all()


@pytest.mark.parametrize(
    ("count", "positives"),
    [
        (50.5, 10),
        (50, 10.5),
        (True, 10),
        (50, False),
        (np.bool_(True), 10),
        (50, np.bool_(False)),
        ("50", 10),
        (50, "10"),
    ],
)
def test_non_integer_counts_rejected(count, positives):
    with pytest.raises(TypeError, match="must be integers"):
        check_group_eligibility(
            count,
            positives,
            SuppressionPolicy(),
        )


def test_numpy_integer_counts_accepted():
    result = check_group_eligibility(
        np.int64(50),
        np.int64(10),
        SuppressionPolicy(),
    )

    assert result["reportable"] is True