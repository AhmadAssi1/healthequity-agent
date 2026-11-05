import pytest
from pandas.testing import assert_frame_equal

from fairness.metrics import compute_group_metrics
from fairness.scenarios import make_gap_scenario, make_group, make_scenario


@pytest.mark.parametrize("metric", ["selection_rate", "tpr", "ppv"])
@pytest.mark.parametrize("gap", [0.00, 0.01, 0.20, 0.50])
def test_planted_gap_matches_requested_value(metric, gap):
    df = make_gap_scenario(metric, gap)
    metrics = compute_group_metrics(df, "group")

    actual_gap = metrics.loc["B", metric] - metrics.loc["A", metric]

    assert actual_gap == pytest.approx(gap, abs=1e-6)
    assert len(df) == 400
    assert metrics["count"].to_dict() == {"A": 200, "B": 200}


def test_scenario_is_reproducible():
    first = make_gap_scenario("tpr", 0.20, seed=42)
    second = make_gap_scenario("tpr", 0.20, seed=42)

    assert_frame_equal(first, second)


@pytest.mark.parametrize("gap", [-0.01, 0.51, float("nan"), 0.015])
def test_invalid_gap_rejected(gap):
    with pytest.raises(ValueError):
        make_gap_scenario("tpr", gap)


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        make_gap_scenario("accuracy", 0.20)

def test_group_matches_requested_confusion_counts():
    df = make_group("A", tn=30, fp=20, fn=10, tp=40)

    counts = df.groupby(["outcome", "prediction"]).size().to_dict()

    assert counts == {
        (0, 0): 30,
        (0, 1): 20,
        (1, 0): 10,
        (1, 1): 40,
    }
    assert df["group"].eq("A").all()


@pytest.mark.parametrize(
    ("tn", "fp", "fn", "tp", "error"),
    [
        (-1, 20, 10, 40, ValueError),
        (0, 0, 0, 0, ValueError),
        (30.5, 20, 10, 40, TypeError),
        (True, 20, 10, 40, TypeError),
    ],
)
def test_invalid_confusion_counts(tn, fp, fn, tp, error):
    with pytest.raises(error):
        make_group("A", tn=tn, fp=fp, fn=fn, tp=tp)


def test_empty_scenario_rejected():
    with pytest.raises(ValueError):
        make_scenario({})