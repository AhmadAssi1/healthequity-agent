"""Offline saved-output review must preserve original outcomes and files."""

import json

import pandas as pd
import pytest
import test_agent as helpers

from app.agent_errors import AgentRunError
from app.schemas import ColumnMapping
from app.tools import AuditToolSession
from healthequity.review_saved import main, review


@pytest.fixture
def saved_run(tmp_path):
    frame = pd.DataFrame({"actual": [0, 0, 1, 1] * 2, "predicted": [0, 1, 0, 1] * 2,
                          "group": ["A"] * 4 + ["B"] * 4})
    session = AuditToolSession(frame, ColumnMapping(outcome="actual", prediction="predicted", groups=["group"]))
    result = helpers.agent_for(session).run()
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (directory / "trace.json").write_text(result.trace.model_dump_json(), encoding="utf-8")
    return directory, session


def test_review_preserves_files_and_uses_saved_model_candidate(saved_run):
    directory, _ = saved_run
    (directory / "findings.json").write_text('{"unrelated":"ignored"}', encoding="utf-8")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    report = review(directory)
    assert report["original_status"] == "completed" and report["recheck"] == "passed"
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_failed_live_run_can_pass_offline_without_being_relabelled(saved_run):
    directory, _ = saved_run
    (directory / "run.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    report = review(directory)
    assert report["original_status"] == "failed" and report["recheck"] == "passed"
    assert json.loads((directory / "run.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_review_reports_unsupported_candidate(saved_run):
    directory, session = saved_run

    class Provider(helpers.ScriptedProvider):
        def generate(self, **kwargs):
            result = super().generate(**kwargs)
            if result.structured_output is not None:
                result.structured_output["summary"] = "The audit used 999999999 rows."
            return result

    with pytest.raises(AgentRunError) as caught:
        helpers.agent_for(session, Provider(session)).run()
    (directory / "run.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    (directory / "trace.json").write_text(caught.value.trace.model_dump_json(), encoding="utf-8")
    report = review(directory)
    assert report["recheck"] == "failed"
    assert any(issue["code"] == "unsupported_number" for issue in report["issues"])


def test_review_without_agent_trace_is_explicitly_unavailable(tmp_path):
    (tmp_path / "run.json").write_text('{"status":"failed"}', encoding="utf-8")
    assert review(tmp_path)["recheck"] == "unavailable"


def test_review_command_has_no_results_or_invalid_limit(tmp_path):
    assert main(["--root", str(tmp_path)]) == 2
    with pytest.raises(SystemExit) as caught:
        main(["--last", "0"])
    assert caught.value.code == 2


def test_review_command_reports_local_success(saved_run, capsys):
    directory, _ = saved_run
    assert main(["--root", str(directory.parent), "--last", "1"]) == 0
    assert "Original live outcomes remain unchanged" in capsys.readouterr().out


def test_saved_summary_grounding_preview_is_read_only(saved_run):
    directory, _ = saved_run
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    report = review(directory, ground=True, include_summary=True)
    assert report["recheck"] == "passed" and report["summary_grounded"]
    assert "This audit covers 8 rows" in report["summary"]
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_show_summary_requires_explicit_grounding_mode():
    with pytest.raises(SystemExit) as caught:
        main(["--show-summary"])
    assert caught.value.code == 2
