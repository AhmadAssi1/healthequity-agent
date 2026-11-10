"""Guard plan drift using trusted evidence, with complete correction records."""

from copy import deepcopy

import pandas as pd
import pytest
from pydantic import ValidationError

from app.analysis_counts import compute_analysis_counts
from app.analysis_selection import select_analyses
from app.plan_guard import guard_analysis_plan
from app.schemas import ColumnMapping
from app.tool_schemas import (
    AttributeAnalysisCounts,
    GuardedAnalysisPlan,
    PlanCorrection,
)
from fairness.policy import SuppressionPolicy


@pytest.fixture
def counts():
    return [AttributeAnalysisCounts(
        attribute="group",
        groups=[{
            "group_value": label,
            "count": 100,
            "positive_outcomes": 50,
            "confusion": {"tn": 30, "fp": 20, "fn": 20, "tp": 30},
        } for label in ["A", "B"]],
    )]


@pytest.fixture
def canonical(counts):
    return select_analyses(counts, score_available=False)


def correction_at(result, path):
    return next(item for item in result.corrections if item.field_path == path)


@pytest.mark.parametrize("as_model", [True, False])
def test_matching_proposal_is_preserved_without_corrections(counts, canonical, as_model):
    proposal = canonical if as_model else canonical.model_dump()
    result = guard_analysis_plan(proposal, counts, False)
    assert result.plan == canonical
    assert result.corrections == []


def test_corrects_score_and_auroc_claims_using_trusted_context(counts, canonical):
    proposed = canonical.model_dump()
    proposed["score_available"] = True
    proposed["attributes"][0]["compute_auroc"] = True
    result = guard_analysis_plan(proposed, counts, False)
    assert result.plan == canonical
    score = correction_at(result, "$['score_available']")
    assert score.proposed_value is True
    assert score.corrected_value is False
    auroc = correction_at(result, "$['attributes'][0]['compute_auroc']")
    assert auroc.operation == "replace"
    assert auroc.reason == "No score column was provided."


@pytest.mark.parametrize("key, value", [
    ("method", "fisher_exact"),
    ("min_expected_count", 0.1),
    ("reason", "Unsupported explanation from the proposal."),
])
def test_corrects_test_selection_expected_counts_and_explanations(counts, canonical, key, value):
    proposed = canonical.model_dump()
    proposed["attributes"][0]["statistical_tests"][0][key] = value
    result = guard_analysis_plan(proposed, counts, False)
    assert result.plan == canonical
    assert len(result.corrections) == 1
    item = result.corrections[0]
    assert item.field_path == f"$['attributes'][0]['statistical_tests'][0]['{key}']"
    assert item.proposed_value == value
    assert item.reason.strip()


@pytest.mark.parametrize("key", ["available", "bootstrap"])
def test_restores_required_available_metrics_and_bootstrap(counts, canonical, key):
    proposed = canonical.model_dump()
    proposed["attributes"][0]["fairness_metrics"][0][key] = False
    result = guard_analysis_plan(proposed, counts, False)
    assert result.plan == canonical
    assert result.corrections[0].corrected_value is True


def test_group_count_is_taken_from_trusted_counts(counts, canonical):
    proposed = canonical.model_dump()
    proposed["attributes"][0]["eligible_group_count"] = 999
    result = guard_analysis_plan(proposed, counts, False)
    assert result.plan.attributes[0].eligible_group_count == 2
    assert "trusted analysis counts" in result.corrections[0].reason


def test_unknown_and_missing_fields_are_recorded(counts, canonical):
    proposed = canonical.model_dump()
    del proposed["score_available"]
    proposed["invented"] = {"execute": "unrecognized_tool"}
    result = guard_analysis_plan(proposed, counts, False)
    added = correction_at(result, "$['score_available']")
    removed = correction_at(result, "$['invented']")
    assert added.operation == "add"
    assert added.proposed_value is None
    assert added.corrected_value is False
    assert removed.operation == "remove"
    assert removed.proposed_value == proposed["invented"]
    assert removed.corrected_value is None
    assert result.plan == canonical


def test_null_is_a_replacement_not_a_missing_field(counts, canonical):
    proposed = canonical.model_dump()
    proposed["score_available"] = None
    result = guard_analysis_plan(proposed, counts, False)
    assert result.corrections[0].operation == "replace"
    assert result.corrections[0].proposed_value is None


@pytest.mark.parametrize("wrong_value", [False, None, "not a list", {"bad": 1}])
def test_wrong_subtree_type_is_replaced_and_logged(counts, canonical, wrong_value):
    proposed = canonical.model_dump()
    proposed["attributes"] = wrong_value
    result = guard_analysis_plan(proposed, counts, False)
    assert result.plan == canonical
    assert len(result.corrections) == 1
    assert result.corrections[0].field_path == "$['attributes']"
    assert result.corrections[0].proposed_value == wrong_value
    assert result.corrections[0].corrected_value == canonical.model_dump()["attributes"]


def test_empty_object_is_completed_with_logged_additions(counts, canonical):
    result = guard_analysis_plan({}, counts, False)
    assert result.plan == canonical
    assert len(result.corrections) == 2
    assert {item.operation for item in result.corrections} == {"add"}


@pytest.mark.parametrize("section", ["fairness_metrics", "statistical_tests"])
def test_missing_and_extra_list_items_are_recorded(counts, canonical, section):
    proposed = canonical.model_dump()
    items = proposed["attributes"][0][section]
    removed = items.pop()
    result = guard_analysis_plan(proposed, counts, False)
    assert result.plan == canonical
    assert len(result.corrections) == 1
    assert result.corrections[0].operation == "add"
    assert result.corrections[0].corrected_value == removed

    proposed = canonical.model_dump()
    items = proposed["attributes"][0][section]
    items.extend([deepcopy(items[0]), deepcopy(items[0])])
    result = guard_analysis_plan(proposed, counts, False)
    assert result.plan == canonical
    assert [item.operation for item in result.corrections] == ["remove", "remove"]


def test_unselected_attribute_is_removed(counts, canonical):
    proposed = canonical.model_dump()
    extra = deepcopy(proposed["attributes"][0])
    extra["attribute"] = "unselected_attribute"
    proposed["attributes"].append(extra)
    result = guard_analysis_plan(proposed, counts, False)
    assert result.plan == canonical
    assert result.corrections[0].operation == "remove"
    assert result.corrections[0].field_path == "$['attributes'][1]"


def test_reordered_items_are_normalized_and_recorded(counts, canonical):
    proposed = canonical.model_dump()
    proposed["attributes"][0]["statistical_tests"].reverse()
    result = guard_analysis_plan(proposed, counts, False)
    assert result.plan == canonical
    assert result.corrections
    assert any(item.field_path.endswith("['rate']") for item in result.corrections)


def test_strict_type_drift_is_not_hidden_by_python_boolean_equality(counts, canonical):
    proposed = canonical.model_dump()
    proposed["score_available"] = 0
    result = guard_analysis_plan(proposed, counts, False)
    assert len(result.corrections) == 1
    assert type(result.corrections[0].proposed_value) is int
    assert result.corrections[0].corrected_value is False


def test_no_input_mutation_or_aliased_trace_values(counts, canonical):
    proposed = canonical.model_dump()
    proposed["attributes"] = {"unexpected": [1, 2]}
    original = deepcopy(proposed)
    original_counts = [item.model_dump() for item in counts]
    result = guard_analysis_plan(proposed, counts, False)
    assert proposed == original
    assert [item.model_dump() for item in counts] == original_counts
    proposed["attributes"]["unexpected"].append(3)
    assert result.corrections[0].proposed_value == {"unexpected": [1, 2]}


def test_result_round_trips_to_trace_json(counts, canonical):
    result = guard_analysis_plan({}, counts, False)
    restored = GuardedAnalysisPlan.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.plan == canonical


@pytest.mark.parametrize("proposal", [None, [], "{}", True, 1])
def test_non_object_proposals_fail_clearly(counts, proposal):
    with pytest.raises(TypeError, match="JSON object"):
        guard_analysis_plan(proposal, counts, False)


@pytest.mark.parametrize("proposal", [
    {"bad": float("nan")}, {"bad": float("inf")}, {"bad": float("-inf")},
    {"bad": object()}, {"bad": (1, 2)}, {1: "non-string key"},
    {"nested": [{"bad": float("nan")}]},
])
def test_non_json_and_non_finite_values_are_rejected(counts, proposal):
    with pytest.raises(ValidationError):
        guard_analysis_plan(proposal, counts, False)


def test_invalid_trusted_inputs_fail_instead_of_returning_a_plan():
    with pytest.raises(ValidationError):
        guard_analysis_plan({}, [], False)


def test_policy_filtered_pipeline_corrects_invented_eligibility_and_score_metadata():
    df = pd.DataFrame({
        "actual": [1] * 20 + [0] * 80 + [1] * 20 + [0] * 80 + [1, 0, 0],
        "predicted": ([1, 0] * 50) * 2 + [1, 0, 0],
        "group": ["A"] * 100 + ["B"] * 100 + ["small"] * 3,
    })
    mapping = ColumnMapping(outcome="actual", prediction="predicted", groups=["group"])
    counts = compute_analysis_counts(df, mapping, SuppressionPolicy())
    proposed = select_analyses(counts, False).model_dump()
    proposed["attributes"][0]["eligible_group_count"] = 3
    proposed["attributes"][0]["compute_auroc"] = True
    proposed["score_available"] = True
    result = guard_analysis_plan(proposed, counts, mapping.score is not None)
    assert result.plan.attributes[0].eligible_group_count == 2
    assert result.plan.attributes[0].compute_auroc is False
    assert result.plan.score_available is False
    assert len(result.corrections) == 3


def test_all_suppressed_groups_cannot_be_made_comparable(counts, canonical):
    empty = AttributeAnalysisCounts(attribute="group", groups=[])
    result = guard_analysis_plan(canonical, [empty], False)
    plan = result.plan.attributes[0]
    assert plan.eligible_group_count == 0
    assert all(not item.available for item in plan.fairness_metrics)
    assert all(item.method == "unavailable" for item in plan.statistical_tests)
    assert result.corrections


def test_guard_does_not_execute_proposed_or_corrected_analyses(monkeypatch, counts):
    def unexpected(*args, **kwargs):
        pytest.fail("Guard must not execute analyses.")

    monkeypatch.setattr("fairness.stats.compare_proportions", unexpected)
    monkeypatch.setattr("fairness.stats.bootstrap_fairness_intervals", unexpected)
    assert guard_analysis_plan({}, counts, False).corrections


@pytest.mark.parametrize("changes", [
    {"operation": "execute"}, {"field_path": " "}, {"reason": ""},
    {"proposed_value": float("nan")}, {"corrected_value": float("inf")},
    {"unexpected": 1},
])
def test_correction_contract_rejects_invalid_records(changes):
    with pytest.raises(ValidationError):
        PlanCorrection.model_validate({
            "operation": "replace", "field_path": "$['score_available']",
            "proposed_value": True, "corrected_value": False,
            "reason": "The dataset has no mapped score column.", **changes,
        })
