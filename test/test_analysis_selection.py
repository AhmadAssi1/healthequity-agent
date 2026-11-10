"""Analysis-selection rules, reference alignment, and plan validation."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.analysis_selection import (
    _rate_counts,
    plan_attribute_analyses,
    plan_fairness_metrics,
    plan_rate_test,
    select_analyses,
)
from app.tool_schemas import (
    AnalysisPlan,
    AttributeAnalysisCounts,
    AttributeAnalysisPlan,
    FairnessMetricPlan,
    GroupAnalysisCounts,
    RateTestPlan,
)
from fairness.stats import compare_proportions

RATES = ("selection_rate", "tpr", "fpr", "ppv")


def group(label="A", *, tn=30, fp=10, fn=20, tp=40):
    return GroupAnalysisCounts(
        group_value=label,
        count=tn + fp + fn + tp,
        positive_outcomes=fn + tp,
        confusion={"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    )


def selection_groups(specs):
    return [
        group(str(i), tn=total - selected, fp=0, fn=0, tp=selected)
        for i, (selected, total) in enumerate(specs)
    ]


def attribute(groups=None, name="group"):
    return AttributeAnalysisCounts(
        attribute=name,
        groups=[group(), group("B", tn=30, fp=20, fn=10, tp=40)]
        if groups is None else groups,
    )


@pytest.fixture
def plan_payload():
    return select_analyses([attribute()], score_available=True).model_dump()


@pytest.mark.parametrize("rate, expected", [
    ("selection_rate", (50, 100)), ("tpr", (40, 60)),
    ("fpr", (10, 40)), ("ppv", (40, 50)),
])
def test_each_rate_uses_its_correct_numerator_and_denominator(rate, expected):
    assert _rate_counts(group(), rate) == expected


def test_unsupported_rate_is_rejected():
    with pytest.raises(ValueError, match="Unsupported rate"):
        _rate_counts(group(), "accuracy")


@pytest.mark.parametrize("specs, expected_method, expected_minimum", [
    ([(5, 10), (5, 10)], "two_proportion_z", 5.0),
    ([(4, 10), (5, 10)], "fisher_exact", 4.5),
    ([(5, 10), (5, 10), (5, 10)], "chi_square", 5.0),
    ([(4, 10), (5, 10), (5, 10)], "fisher_monte_carlo", 14 / 3),
    # Observed zero cells do not imply small EXPECTED counts.
    ([(0, 10), (10, 10)], "two_proportion_z", 5.0),
    ([(0, 10), (10, 10), (5, 10)], "chi_square", 5.0),
])
def test_group_count_and_expected_count_choose_correct_method(
    specs, expected_method, expected_minimum,
):
    plan = plan_rate_test(selection_groups(specs), "selection_rate")
    assert plan.method == expected_method
    assert plan.min_expected_count == pytest.approx(expected_minimum)
    assert plan.reason.strip()


@pytest.mark.parametrize("size", [0, 1])
@pytest.mark.parametrize("rate", RATES)
def test_fewer_than_two_groups_excludes_statistical_tests(size, rate):
    plan = plan_rate_test([group()] * size, rate)
    assert plan.method == "unavailable"
    assert plan.min_expected_count is None


@pytest.mark.parametrize("rate, undefined_group", [
    ("tpr", group(tn=20, fp=10, fn=0, tp=0)),
    ("fpr", group(tn=0, fp=0, fn=20, tp=10)),
    ("ppv", group(tn=20, fp=0, fn=10, tp=0)),
])
def test_zero_denominator_excludes_test_without_dropping_the_group(rate, undefined_group):
    plan = plan_rate_test([undefined_group, group("B")], rate)
    assert plan.method == "unavailable"
    assert plan.min_expected_count is None
    assert "denominator is zero" in plan.reason


@pytest.mark.parametrize("selected", [0, 10])
def test_single_success_failure_category_has_no_test(selected):
    plan = plan_rate_test(selection_groups([(selected, 10)] * 2), "selection_rate")
    assert plan.method == "unavailable"
    assert plan.min_expected_count == 0.0


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("groups", [
    [group(), group("B", tn=30, fp=20, fn=10, tp=40)],
    [group(tn=2, fp=1, fn=1, tp=2), group("B", tn=1, fp=1, fn=1, tp=1)],
    [group(str(i), tn=2, fp=1, fn=1, tp=2) for i in range(3)],
    [group(str(i)) for i in range(3)],
])
def test_planned_test_matches_existing_engine_for_each_rate(rate, groups):
    numerators, totals = zip(*[_rate_counts(item, rate) for item in groups], strict=True)
    reference = compare_proportions(numerators, totals, n_resamples=99, seed=42)
    plan = plan_rate_test(groups, rate)
    assert plan.method == reference.method
    assert plan.min_expected_count == pytest.approx(reference.min_expected_count)


@pytest.mark.parametrize("size", [0, 1])
def test_fewer_than_two_groups_excludes_all_fairness_gaps_and_bootstraps(size):
    plans = plan_fairness_metrics([group()] * size)
    assert len(plans) == 3
    assert all(not item.available and not item.bootstrap for item in plans)


def test_defined_rates_include_all_headline_metrics_and_bootstraps():
    plans = plan_fairness_metrics(attribute().groups)
    assert len({item.metric for item in plans}) == 3
    assert all(item.available and item.bootstrap for item in plans)


@pytest.mark.parametrize("undefined_group, unavailable_metrics", [
    (group(tn=20, fp=10, fn=0, tp=0), {"equalized_odds_difference"}),
    (group(tn=0, fp=0, fn=20, tp=10), {"equalized_odds_difference"}),
    (group(tn=20, fp=0, fn=10, tp=0), {"ppv_difference"}),
])
def test_undefined_component_excludes_only_affected_metrics(undefined_group, unavailable_metrics):
    plans = plan_fairness_metrics([undefined_group, group("B")])
    assert {item.metric for item in plans if not item.available} == unavailable_metrics
    assert all(item.bootstrap == item.available for item in plans)


def test_gap_availability_is_distinct_from_significance_test_availability():
    groups = [group("A", tn=20, fp=0, fn=10, tp=0), group("B", tn=30, fp=0, fn=10, tp=0)]
    metrics = {item.metric: item for item in plan_fairness_metrics(groups)}
    assert metrics["demographic_parity_difference"].available is True
    assert plan_rate_test(groups, "selection_rate").method == "unavailable"


@pytest.mark.parametrize("groups, scores, expected", [
    ([], False, False), ([], True, False),
    ([group()], False, False), ([group()], True, True),
    ([group(tn=10, fp=10, fn=0, tp=0)], True, False),
    ([group(tn=0, fp=0, fn=10, tp=10)], True, False),
    ([group(tn=10, fp=10, fn=0, tp=0), group("B")], True, True),
])
def test_auroc_requires_scores_and_at_least_one_two_class_group(groups, scores, expected):
    plan = plan_attribute_analyses(attribute(groups), scores)
    assert plan.compute_auroc is expected
    assert plan.auroc_reason.strip()
    assert plan.eligible_group_count == len(groups)


def test_selection_does_not_execute_tests_or_bootstrap(monkeypatch):
    def unexpected_execution(*args, **kwargs):
        pytest.fail("Planning must not execute statistical tests or bootstrap.")

    monkeypatch.setattr("fairness.stats.compare_proportions", unexpected_execution)
    monkeypatch.setattr("fairness.stats.bootstrap_fairness_intervals", unexpected_execution)
    assert len(select_analyses([attribute()], True).attributes) == 1


def test_complete_plan_round_trip_and_multiple_attributes():
    inputs = [attribute(), attribute([], "all_suppressed")]
    original = [item.model_dump() for item in inputs]
    result = select_analyses(inputs, score_available=True)
    assert [item.attribute for item in result.attributes] == ["group", "all_suppressed"]
    assert result.attributes[1].eligible_group_count == 0
    assert all(not item.available for item in result.attributes[1].fairness_metrics)
    assert [item.model_dump() for item in inputs] == original
    assert AnalysisPlan.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("changes", [
    {"rate": "accuracy"}, {"method": "unknown"},
    {"min_expected_count": None}, {"min_expected_count": -1},
    {"min_expected_count": float("nan")}, {"reason": "   "}, {"extra": 1},
])
def test_rate_plan_rejects_invalid_contract(changes):
    with pytest.raises(ValidationError):
        RateTestPlan.model_validate({
            "rate": "tpr", "method": "two_proportion_z",
            "min_expected_count": 5.0, "reason": "Expected counts are adequate.",
            **changes,
        })


def test_fairness_metric_plan_rejects_bootstrap_for_unavailable_metric():
    with pytest.raises(ValidationError, match="Bootstrap cannot be planned"):
        FairnessMetricPlan(
            metric="ppv_difference", available=False, bootstrap=True,
            reason="Undefined PPV.",
        )


@pytest.mark.parametrize("change", ["duplicate_metric", "duplicate_rate", "zero_auroc"])
def test_attribute_plan_rejects_inconsistent_structure(plan_payload, change):
    payload = deepcopy(plan_payload["attributes"][0])
    if change == "duplicate_metric":
        payload["fairness_metrics"][1] = deepcopy(payload["fairness_metrics"][0])
    elif change == "duplicate_rate":
        payload["statistical_tests"][1] = deepcopy(payload["statistical_tests"][0])
    else:
        payload = plan_attribute_analyses(attribute([]), True).model_dump()
        payload["compute_auroc"] = True
    with pytest.raises(ValidationError):
        AttributeAnalysisPlan.model_validate(payload)


def test_single_group_plan_rejects_available_fairness_comparison():
    payload = plan_attribute_analyses(attribute([group()]), False).model_dump()
    payload["fairness_metrics"][0]["available"] = True
    with pytest.raises(ValidationError, match="Fairness comparisons require"):
        AttributeAnalysisPlan.model_validate(payload)


def test_single_group_plan_rejects_available_statistical_comparison():
    payload = plan_attribute_analyses(attribute([group()]), False).model_dump()
    payload["statistical_tests"][0].update(method="two_proportion_z", min_expected_count=5.0)
    with pytest.raises(ValidationError, match="Statistical comparisons require"):
        AttributeAnalysisPlan.model_validate(payload)


def test_plan_rejects_auroc_without_scores(plan_payload):
    plan_payload["score_available"] = False
    with pytest.raises(ValidationError, match="without a score column"):
        AnalysisPlan.model_validate(plan_payload)


def test_plan_rejects_duplicate_attributes(plan_payload):
    plan_payload["attributes"] *= 2
    with pytest.raises(ValidationError, match="exactly one analysis plan"):
        AnalysisPlan.model_validate(plan_payload)


def test_plan_rejects_empty_attribute_list():
    with pytest.raises(ValidationError):
        select_analyses([], score_available=False)
