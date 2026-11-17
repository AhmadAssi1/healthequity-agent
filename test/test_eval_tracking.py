"""MLflow export, failure handling, receipts, and CI event boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from evals.ci_summary import main as publish_summary
from evals.tracking import log_evaluation


def bundle(tmp_path, *, failed=False):
    directory = tmp_path / "batch"
    directory.mkdir()
    row = {
        "scenario": "no_gap", "expected_status": "audit",
        "status": "failed" if failed else "completed", "provider_calls": 8,
        "elapsed_seconds": 12.5, "input_tokens": 100, "output_tokens": 20,
        "estimated_cost_usd": None, "schema_valid": not failed, "numeric_pass": not failed,
        "suppression_correct": not failed, "tool_selection_correct": not failed,
        "automated_claims_pass": not failed,
    }
    scorecard = {
        "mode": "live", "model": "test-model", "benchmark_status": "pending_human_claim_review",
        "numeric_accuracy_scope": "fixture checks only", "numeric_accuracy": 0.0 if failed else 1.0,
        "unsupported_claim_rate": None, "live_benchmark_passed": False,
        "prices_usd_per_million": {"input": None, "output": None}, "cases": [row],
    }
    (directory / "scorecard.json").write_text(json.dumps(scorecard), encoding="utf-8")
    (directory / "scorecard.md").write_text("# Test scorecard\n", encoding="utf-8")
    (directory / "no_gap_trace.json").write_text('{"aggregate_only":true}', encoding="utf-8")
    (directory / "no_gap_findings.json").write_text('{"summary":"fixture"}', encoding="utf-8")
    (directory / "private.env").write_text("do-not-upload", encoding="utf-8")
    return directory


def mock_client():
    client = Mock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="123")
    client.create_run.side_effect = [SimpleNamespace(info=SimpleNamespace(run_id=n))
                                   for n in ("parent", "child", "parent2", "child2")]
    return client


def test_export_logs_parent_children_and_allowlisted_aggregates(tmp_path):
    directory = bundle(tmp_path)
    before = (directory / "scorecard.json").read_bytes()
    client = mock_client()
    receipt = log_evaluation(directory, tracking_uri="sqlite:///example.db", client=client)
    assert receipt["run_id"] == "parent" and receipt["case_run_ids"] == {"no_gap": "child"}
    assert client.create_run.call_args_list[1].kwargs["tags"]["mlflow.parentRunId"] == "parent"
    uploaded = [Path(c.args[1]).name for c in client.log_artifact.call_args_list]
    assert set(uploaded) == {"scorecard.json", "scorecard.md", "no_gap_trace.json", "no_gap_findings.json"}
    assert (directory / "scorecard.json").read_bytes() == before
    metric_names = [c.args[1] for c in client.log_metric.call_args_list]
    assert "numeric_accuracy" in metric_names
    assert "unsupported_claim_rate" not in metric_names
    assert "estimated_cost_usd" not in metric_names
    assert client.create_run.call_args_list[0].kwargs["tags"]["evaluation.human_review"] == "pending"


def test_receipt_prevents_duplicate_export_but_trace_change_has_new_identity(tmp_path):
    directory = bundle(tmp_path)
    client = mock_client()
    first = log_evaluation(directory, tracking_uri="sqlite:///example.db", client=client)
    again = log_evaluation(directory, tracking_uri="sqlite:///example.db", client=client)
    assert again["reused"] and again["run_id"] == first["run_id"]
    assert client.create_run.call_count == 2
    (directory / "no_gap_trace.json").write_text('{"aggregate_only":true,"changed":true}', encoding="utf-8")
    changed = log_evaluation(directory, tracking_uri="sqlite:///example.db", client=client)
    assert not changed["reused"] and changed["run_id"] == "parent2"


def test_failed_evaluation_stays_failed_in_mlflow(tmp_path):
    directory = bundle(tmp_path, failed=True)
    client = mock_client()
    log_evaluation(directory, tracking_uri="sqlite:///example.db", client=client)
    assert all(c.kwargs["status"] == "FAILED" for c in client.set_terminated.call_args_list)


def test_export_storage_failure_is_surfaced_and_no_success_receipt_written(tmp_path):
    directory = bundle(tmp_path)
    client = mock_client()
    client.log_artifact.side_effect = OSError("unavailable")
    with pytest.raises(OSError):
        log_evaluation(directory, tracking_uri="sqlite:///example.db", client=client)
    assert not (directory / "mlflow-export.json").exists()
    assert client.set_terminated.call_args.kwargs["status"] == "FAILED"


def test_case_paths_cannot_escape_source_directory(tmp_path):
    directory = bundle(tmp_path)
    path = directory / "scorecard.json"
    scorecard = json.loads(path.read_text())
    scorecard["cases"][0]["scenario"] = "../private"
    path.write_text(json.dumps(scorecard))
    client = mock_client()
    with pytest.raises(ValueError):
        log_evaluation(directory, client=client)
    client.create_run.assert_not_called()


def test_ci_summary_prints_and_publishes_saved_markdown(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "outputs/evaluations/batch"
    path.mkdir(parents=True)
    (path / "scorecard.md").write_text("# Evaluation\n", encoding="utf-8")
    target = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
    publish_summary()
    assert "# Evaluation" in capsys.readouterr().out
    assert "# Evaluation" in target.read_text()


def test_workflow_has_no_live_execution_or_api_secret_on_pull_requests():
    path = Path(__file__).resolve().parents[1] / ".github/workflows/eval.yml"
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"pull_request", "push"}
    assert workflow["on"]["push"]["branches"] == ["main"]
    offline, live = workflow["jobs"]["deterministic"], workflow["jobs"]["live"]
    assert "OPENAI_API_KEY" not in offline["env"]
    assert not any("--live" in step.get("run", "") for step in offline["steps"])
    assert live["needs"] == "deterministic"
    assert live["if"] == "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    runs = [s.get("run", "") for s in live["steps"]]
    assert runs.index("python -m evals.probe") < runs.index("python -m evals --live --mlflow")
    assert all(s["if"] == "always()" for s in live["steps"] if "upload-artifact" in s.get("uses", ""))


def test_export_to_real_sqlite_store_and_retrieve_artifacts(tmp_path, monkeypatch):
    from mlflow import MlflowClient

    monkeypatch.chdir(tmp_path)
    directory = bundle(tmp_path)
    uri = "sqlite:///tracking.db"
    receipt = log_evaluation(directory, tracking_uri=uri)
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(receipt["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.metrics["numeric_accuracy"] == 1
    assert "unsupported_claim_rate" not in run.data.metrics
    assert client.get_run(receipt["case_run_ids"]["no_gap"]).info.status == "FINISHED"
    downloaded = client.download_artifacts(receipt["run_id"], "evaluation/scorecard.json")
    assert json.loads(Path(downloaded).read_text())["model"] == "test-model"
