"""CLI integration checks with actual tools and an offline scripted provider."""

import json
from unittest.mock import Mock

import pandas as pd
import pytest
import test_agent as agent_fixtures

from app.agent import AuditAgent
from app.agent_schemas import AgentRunTrace
from app.audit_schemas import AuditFindings
from app.llm_config import LLMConfig
from healthequity import cli


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "demo.csv"
    pd.DataFrame({
        "outcome": [0, 0, 1, 1] * 2,
        "prediction": [0, 1, 0, 1] * 2,
        "gender": ["A"] * 4 + ["B"] * 4,
    }).to_csv(path, index=False)
    return path


@pytest.fixture
def offline(monkeypatch):
    create_provider = Mock(return_value=None)
    monkeypatch.setattr(cli, "load_llm_config", lambda: LLMConfig(model_name="test-model"))
    monkeypatch.setattr(cli, "OpenAIProvider", create_provider)

    class Provider(agent_fixtures.ScriptedProvider):
        def generate(self, **kwargs):
            result = super().generate(**kwargs)
            if result.structured_output is not None:
                result.structured_output["dataset"]["score_available"] = self.session.get_evidence().profile.score_available
            return result

    def agent(**kwargs):
        kwargs["provider"] = Provider(kwargs["session"])
        return AuditAgent(**kwargs)

    monkeypatch.setattr(cli, "AuditAgent", agent)
    return create_provider


def files_at(root):
    directory, = root.iterdir()
    return directory, json.loads((directory / "run.json").read_text(encoding="utf-8"))


def test_cli_runs_actual_tools_and_persists_validated_artifacts(csv_file, tmp_path, offline, capsys):
    root = tmp_path / "runs"
    assert cli.main(["audit", str(csv_file), "--output-dir", str(root)]) == 0
    directory, record = files_at(root)
    assert record["status"] == "completed"
    assert record["mapping"]["groups"] == ["gender"]
    assert record["fingerprint"]["row_count"] == 8
    trace = AgentRunTrace.model_validate_json((directory / "trace.json").read_text(encoding="utf-8"))
    findings = AuditFindings.model_validate_json((directory / "findings.json").read_text(encoding="utf-8"))
    verdict = json.loads((directory / "claims_validation.json").read_text(encoding="utf-8"))
    assert trace.status == "completed" and trace.claims_validation.passed and verdict["passed"]
    assert findings.dataset.row_count == 8
    assert record["trace_id"] == trace.run_id
    assert "Status: completed" in capsys.readouterr().out
    assert offline.call_count == 1
    assert not list(directory.glob("*.tmp"))


@pytest.mark.parametrize("problem", ["missing_file", "missing_outcome", "missing_prediction", "no_groups",
                                    "nonbinary", "duplicate_headers", "empty", "bad_budget", "xlsx"])
def test_cli_refuses_bad_inputs_before_provider_setup(csv_file, tmp_path, offline, problem):
    frame = pd.read_csv(csv_file)
    extra = []
    if problem == "missing_file":
        csv_file = tmp_path / "absent.csv"
    elif problem == "xlsx":
        csv_file = tmp_path / "file.xlsx"
    elif problem.startswith("missing_"):
        frame.drop(columns=problem.removeprefix("missing_"), inplace=True)
        frame.to_csv(csv_file, index=False)
    elif problem == "no_groups":
        frame.drop(columns="gender").to_csv(csv_file, index=False)
    elif problem == "nonbinary":
        frame.loc[0, "outcome"] = 3
        frame.to_csv(csv_file, index=False)
    elif problem == "duplicate_headers":
        csv_file.write_text("outcome,prediction,gender,gender\n0,1,A,B\n", encoding="utf-8")
    elif problem == "empty":
        frame.iloc[:0].to_csv(csv_file, index=False)
    else:
        extra = ["--max-seconds", "181"]
    root = tmp_path / "runs"
    assert cli.main(["audit", str(csv_file), "--output-dir", str(root), *extra]) == 2
    offline.assert_not_called()
    directory, record = files_at(root)
    assert record["status"] == "failed" and record["phase"] == "input_validation"
    assert record["error"] and record["trace_id"] is None
    assert not (directory / "trace.json").exists()
    assert not (directory / "findings.json").exists()


@pytest.mark.parametrize("no_score", [False, True])
def test_cli_custom_mapping_and_score_selection(csv_file, tmp_path, offline, no_score):
    frame = pd.read_csv(csv_file).rename(columns={"outcome": "actual", "prediction": "label", "gender": "cohort"})
    frame["risk"] = 0.5
    frame.to_csv(csv_file, index=False)
    root = tmp_path / "runs"
    options = ["--no-score"] if no_score else ["--score", "risk"]
    assert cli.main(["audit", str(csv_file), "--outcome", "actual", "--prediction", "label",
                     "--groups", "cohort", "--output-dir", str(root), *options]) == 0
    _, record = files_at(root)
    assert record["mapping"] == {"outcome": "actual", "prediction": "label", "groups": ["cohort"],
                                 "score": None if no_score else "risk"}


def test_cli_detects_canonical_score_column(csv_file, tmp_path, offline):
    frame = pd.read_csv(csv_file)
    frame["score"] = 0.5
    frame.to_csv(csv_file, index=False)
    root = tmp_path / "runs"
    assert cli.main(["audit", str(csv_file), "--output-dir", str(root)]) == 0
    assert files_at(root)[1]["mapping"]["score"] == "score"


def test_cli_failed_agent_preserves_trace_and_verdict(csv_file, tmp_path, offline, monkeypatch):
    class BadProvider(agent_fixtures.ScriptedProvider):
        def generate(self, **kwargs):
            result = super().generate(**kwargs)
            if result.structured_output is not None:
                result.structured_output["summary"] = "The audit used 999999999 rows."
            return result

    def agent(**kwargs):
        kwargs["provider"] = BadProvider(kwargs["session"])
        return AuditAgent(**kwargs)

    monkeypatch.setattr(cli, "AuditAgent", agent)
    root = tmp_path / "runs"
    assert cli.main(["audit", str(csv_file), "--output-dir", str(root)]) == 1
    directory, record = files_at(root)
    trace = AgentRunTrace.model_validate_json((directory / "trace.json").read_text(encoding="utf-8"))
    assert trace.status == "failed" and trace.claims_validation.passed is False
    assert record["status"] == "failed" and record["trace_id"] == trace.run_id
    assert (directory / "claims_validation.json").exists()
    assert not (directory / "findings.json").exists()


def test_cli_provider_failure_does_not_expose_exception_secrets(csv_file, tmp_path, offline, monkeypatch, capsys):
    secret = "credential-must-not-appear"
    monkeypatch.setattr(cli, "OpenAIProvider", Mock(side_effect=RuntimeError(secret)))
    root = tmp_path / "runs"
    assert cli.main(["audit", str(csv_file), "--output-dir", str(root)]) == 2
    directory, record = files_at(root)
    assert record["phase"] == "provider_setup" and record["status"] == "failed"
    assert secret not in capsys.readouterr().err
    assert secret not in (directory / "run.json").read_text(encoding="utf-8")


def test_cli_runs_have_separate_directories(csv_file, tmp_path, offline):
    root = tmp_path / "runs"
    for _ in range(2):
        assert cli.main(["audit", str(csv_file), "--output-dir", str(root)]) == 0
    assert len(list(root.glob("*/trace.json"))) == 2


def test_cli_checks_storage_before_provider(csv_file, tmp_path, offline):
    root = tmp_path / "not-a-directory"
    root.write_text("existing", encoding="utf-8")
    assert cli.main(["audit", str(csv_file), "--output-dir", str(root)]) == 3
    offline.assert_not_called()
    assert root.read_text(encoding="utf-8") == "existing"


def test_atomic_writer_cleans_temporary_file_on_failed_replace(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    target.write_text('{"old":true}', encoding="utf-8")
    monkeypatch.setattr(cli.os, "replace", Mock(side_effect=OSError("blocked")))
    with pytest.raises(OSError):
        cli._write_json(target, {"new": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["record.json"]


def test_help_is_available_without_credentials(offline, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["audit", "--help"])
    assert caught.value.code == 0
    assert "--groups" in capsys.readouterr().out
    offline.assert_not_called()
