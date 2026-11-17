"""Failure-sensitive scoring and offline runner checks; no external API calls."""

import json
from copy import deepcopy

import pytest

from app.agent import AuditAgent
from app.llm_config import LLMConfig
from app.tools import AuditToolSession
from evals.offline_provider import OfflineProvider
from evals.runner import main, run_suite, schema_failure_count, usage_totals
from evals.scenarios import build_scenarios
from evals.scoring import _plan_matches, score_findings, summarize


@pytest.fixture(scope="module")
def cases():
    return {case.name: case for case in build_scenarios()}


@pytest.fixture(scope="module")
def completed(cases):
    case = cases["no_gap"]
    session = AuditToolSession(case.data, case.mapping)
    result = AuditAgent(
        provider=OfflineProvider(session), session=session,
        llm_config=LLMConfig(model_name="offline-scripted"),
    ).run()
    return result.findings.model_dump(mode="json"), result.trace.model_dump(mode="json")


def test_oracle_accepts_complete_protocol_but_does_not_certify_human_claims(cases, completed):
    result = score_findings(cases["no_gap"], *completed)
    for field in ("numeric_pass", "schema_valid", "suppression_correct",
                  "tool_selection_correct", "automated_claims_pass"):
        assert result[field] is True
    assert result["human_review"] == "pending"
    assert result["model_proposals_total"] == 0


@pytest.mark.parametrize("mutation", ["count", "rate", "gap", "wrong_attribute", "wrong_shape"])
def test_structurally_valid_wrong_answers_fail_scoring(cases, completed, mutation):
    findings, trace = deepcopy(completed)
    if mutation == "count":
        findings["attributes"][0]["groups"][0]["count"] -= 1
    elif mutation == "rate":
        findings["attributes"][0]["groups"][0]["tpr"]["value"] = 0.123
    elif mutation == "gap":
        findings["attributes"][0]["fairness_findings"][0]["value"] = 0.123
    elif mutation == "wrong_attribute":
        findings["attributes"][0]["attribute"] = "wrong"
        findings["dataset"]["group_columns"] = ["wrong"]
    else:
        findings["dataset"]["column_count"] += 1
    assert not score_findings(cases["no_gap"], findings, trace)["numeric_pass"]


@pytest.mark.parametrize("mutation", ["missing_stage", "failed_tool", "bootstrap_size", "bootstrap_seed", "missing_rate"])
def test_bad_or_missing_tool_choices_fail_even_with_correct_findings(cases, completed, mutation):
    findings, trace = deepcopy(completed)
    records = [r for t in trace["turns"] for r in t["tool_executions"]]
    if mutation == "missing_stage":
        trace["turns"][1]["tool_executions"] = []
    elif mutation == "failed_tool":
        records[1]["result"]["is_error"] = True
    elif mutation.startswith("bootstrap"):
        field = "n_resamples" if mutation == "bootstrap_size" else "seed"
        records[-1]["call"]["arguments"][field] = 17
    else:
        tests = next(r for r in records if r["call"]["name"] == "run_significance_test")
        tests["result"]["output"]["results"].pop()
    assert not score_findings(cases["no_gap"], findings, trace)["tool_selection_correct"]


def test_bad_proposed_plan_is_not_credited_as_correct_after_guardrail(cases, completed):
    findings, trace = deepcopy(completed)
    selected = next(r for t in trace["turns"] for r in t["tool_executions"]
                    if r["call"]["name"] == "select_analyses")
    selected["call"]["arguments"]["proposed_plan"] = {"attributes": []}
    result = score_findings(cases["no_gap"], findings, trace)
    assert result["tool_selection_correct"] is True
    assert result["model_proposals_total"] == 1
    assert result["model_proposals_correct"] == 0


@pytest.mark.parametrize("bad_plan", [None, [], {}, {"attributes": [None]}, {"attributes": ["wrong"]}])
def test_malformed_proposals_fail_without_crashing(cases, bad_plan):
    assert not _plan_matches(bad_plan, cases["no_gap"])


def test_failed_and_not_run_cases_remain_in_denominators():
    rows = [
        {"expected_status": "audit", "status": "completed", "numeric_pass": True},
        {"expected_status": "audit", "status": "failed"},
        {"expected_status": "audit", "status": "not_run"},
        {"expected_status": "refuse", "status": "refused", "refusal_correct": True},
        {"expected_status": "refuse", "status": "failed", "refusal_correct": False},
    ]
    summary = summarize(rows, "live")
    assert summary["numeric_accuracy"] == 1 / 3
    assert summary["refusal_correctness"] == 0.5
    assert summary["unsupported_claim_rate"] is None
    assert summary["live_benchmark_passed"] is False


def test_refusals_never_construct_provider(cases, tmp_path):
    def forbidden(session):
        pytest.fail("Invalid input must not construct a provider")
    result = run_suite(
        [cases["missing_outcome"], cases["mislabelled_prediction"]],
        output_dir=tmp_path / "refusals", mode="live",
        llm_config=LLMConfig(model_name="test-model"), provider_factory=forbidden,
        progress=lambda *a, **k: None,
    )
    assert result["refusal_correctness"] == 1
    assert all(row["provider_calls"] == 0 for row in result["cases"])


def test_provider_failure_is_saved_and_secret_exception_text_is_not(cases, tmp_path):
    def broken(session):
        raise RuntimeError("private-api-token-must-not-appear")
    output = tmp_path / "failure"
    result = run_suite(
        [cases["no_gap"]], output_dir=output, mode="live",
        llm_config=LLMConfig(model_name="test-model"), provider_factory=broken,
        progress=lambda *a, **k: None,
    )
    assert result["numeric_accuracy"] == 0
    assert result["cases"][0]["error_type"] == "RuntimeError"
    assert "private-api-token" not in (output / "scorecard.json").read_text()


def test_offline_scorecard_persists_trace_and_finishes_all_twelve(cases, tmp_path):
    output = tmp_path / "offline"
    result = run_suite(
        list(cases.values()), output_dir=output, mode="offline",
        llm_config=LLMConfig(model_name="offline-scripted"), provider_factory=OfflineProvider,
        progress=lambda *a, **k: None,
    )
    assert result["completed_valid_audits"] == 10
    assert result["numeric_accuracy"] == result["tool_selection_accuracy"] == 1
    assert result["refusal_correctness"] == result["suppression_correctness"] == 1
    assert len(list(output.glob("*_trace.json"))) == 10
    assert result["benchmark_status"] == "offline_smoke_only"
    assert all(row["estimated_cost_usd"] == 0 for row in result["cases"])
    stored = json.loads((output / "scorecard.json").read_text())
    assert stored == result
    assert "not established" in (output / "scorecard.md").read_text()


def test_unknown_usage_is_not_counted_as_free():
    trace = {"turns": [{"response": {"usage": {"input_tokens": 12, "output_tokens": 5}}},
                       {"response": None}]}
    assert usage_totals(trace) == {"input_tokens": None, "output_tokens": None}


def test_claim_rejections_and_transport_errors_are_not_schema_failures():
    errors = ["Final findings failed claim validation.",
              "Model turn failed (APIConnectionError).",
              "Structured output parsing failed.",
              "AuditFindings schema validation failed.", None]
    trace = {"turns": [{"structured_response_requested": True, "error": e} for e in errors]}
    assert schema_failure_count(trace) == 2


@pytest.mark.parametrize("prices", [(float("nan"), 1), (1, float("inf")), (-1, 1), (1, None)])
def test_invalid_prices_rejected_before_audit(tmp_path, prices):
    with pytest.raises(ValueError):
        run_suite([], output_dir=tmp_path / "bad", mode="offline",
                  llm_config=LLMConfig(model_name="offline-scripted"),
                  provider_factory=OfflineProvider, input_price=prices[0], output_price=prices[1])


def test_cli_unknown_scenario_rejected_before_provider():
    with pytest.raises(SystemExit) as error:
        main(["--scenario", "absent"])
    assert error.value.code == 2
