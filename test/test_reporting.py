"""PDF/HTML reports stay evidence-bound, suppress small groups and retain provenance."""

import hashlib
import json
from copy import deepcopy

import pytest
from pypdf import PdfReader

from app.agent import AuditAgent
from app.llm_config import LLMConfig
from app.reporting import METHODOLOGY, build_context, render_report
from app.tools import AuditToolSession
from evals.offline_provider import OfflineProvider
from evals.scenarios import build_scenarios
from healthequity.report import load_saved, main


@pytest.fixture(scope="module")
def report_case():
    case = next(c for c in build_scenarios() if c.name == "tiny_group")
    session = AuditToolSession(case.data, case.mapping)
    result = AuditAgent(provider=OfflineProvider(session), session=session,
                        llm_config=LLMConfig(model_name="offline-scripted")).run()
    return result, session.get_evidence()


def test_context_preserves_numerical_inputs_and_excludes_suppressed_rows(report_case):
    result, evidence = report_case
    before = deepcopy((result.findings.model_dump(), evidence.model_dump()))
    context = build_context(result.findings, evidence, run_id="audit", trace_id="trace")
    section = next(s for s in context["sections"] if s["title"] == "Attribute detail: group")
    group_table = next(b for b in section["blocks"] if b["kind"] == "table")
    assert {r[0] for r in group_table["rows"][1:]} == {"A", "B"}
    suppression = [b["text"] for b in section["blocks"] if b["kind"] == "bullet"]
    assert any("Tiny" in s for s in suppression) and any("LowPositive" in s for s in suppression)
    assert context["human_review"] == "pending"
    assert context["original_limitations"] == result.findings.limitations
    assert (result.findings.model_dump(), evidence.model_dump()) == before


@pytest.mark.parametrize("defect", ["count", "unsupported_summary", "unsupported_limitation"])
def test_report_refuses_bad_findings_before_writing(report_case, tmp_path, defect):
    result, evidence = report_case
    findings = result.findings.model_copy(deep=True)
    if defect == "count":
        findings.attributes[0].groups[0].count -= 1
    elif defect == "unsupported_summary":
        findings.summary = "There are 999999999 rows."
    else:
        findings.limitations = ["There are 999999999 rows."]
    destination = tmp_path / "bad.pdf"
    with pytest.raises(ValueError, match="validation"):
        render_report(findings, evidence, destination, run_id="audit", trace_id="trace")
    assert not destination.exists()


def test_report_outputs_contain_provenance_and_escaped_html(report_case, tmp_path):
    result, evidence = report_case
    destination = tmp_path / "draft.pdf"
    receipt = render_report(result.findings, evidence, destination,
                            run_id="audit<script>bad()</script>", trace_id="trace-123",
                            source_provenance={"human_review": "approved", "source_trace_sha256": "a" * 64})
    assert receipt.trace_id == "trace-123"
    reader = PdfReader(destination)
    text = "\n".join(p.extract_text() for p in reader.pages)
    for required in ("Executive summary", "Fairness Results by Group Attribute", "Technical Appendix",
                     "Report Provenance", "Interpretation limits", "trace-123", "Tiny", "LowPositive"):
        assert required in text
    html = destination.with_suffix(".html").read_text(encoding="utf-8")
    for removed in ("HUMAN REVIEW PENDING", "human narrative review pending", "Management report /",
                    "This report summarizes statistical evidence."):
        assert removed not in text and removed not in html
    assert '<p class="notice">' not in html
    assert "<script>bad()</script>" not in html
    assert "&lt;script&gt;" in html
    assert 'name="viewport"' in html and 'alt="' in html
    review = json.loads(destination.with_suffix(".review.json").read_text())
    assert review["human_review"] == "pending"
    assert review["source_trace_sha256"] == "a" * 64
    assert review["methodology_sha256"] == hashlib.sha256(METHODOLOGY.read_bytes()).hexdigest()
    assert review["pdf_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_existing_reports_are_never_overwritten(report_case, tmp_path):
    result, evidence = report_case
    destination = tmp_path / "draft.pdf"
    destination.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        render_report(result.findings, evidence, destination, run_id="audit", trace_id="trace")
    assert destination.read_bytes() == b"original"


@pytest.mark.parametrize("metadata", [{"row_count": 1}, {"row_count": 480, "decision_threshold": 2},
                                      {"row_count": 480, "decision_threshold": True}])
def test_mismatched_provenance_or_invalid_threshold_rejected(report_case, metadata):
    result, evidence = report_case
    with pytest.raises(ValueError):
        build_context(result.findings, evidence, run_id="audit", trace_id="trace", model_metadata=metadata)


def test_missing_operating_point_stays_unknown(report_case):
    result, evidence = report_case
    context = build_context(result.findings, evidence, run_id="audit", trace_id="trace")
    text = json.dumps(context)
    assert "Not supplied; cannot infer from predictions" in text
    assert "Calibration curves cannot be reconstructed" in text


def test_render_failure_removes_incomplete_outputs(report_case, tmp_path, monkeypatch):
    from app import reporting
    result, evidence = report_case
    destination = tmp_path / "draft.pdf"
    def fail(context, path):
        path.write_bytes(b"partial")
        raise OSError("disk failure")
    monkeypatch.setattr(reporting, "_write_pdf", fail)
    with pytest.raises(OSError):
        render_report(result.findings, evidence, destination, run_id="audit", trace_id="trace")
    assert not destination.exists() and not destination.with_suffix(".html").exists()


def test_companion_created_during_render_is_preserved(report_case, tmp_path, monkeypatch):
    from app import reporting
    result, evidence = report_case
    destination = tmp_path / "audit.pdf"
    companion = destination.with_suffix(".html")
    def competing_writer(context, staged_pdf):
        companion.write_text("Existing report")
        staged_pdf.write_bytes(b"staged PDF")
    monkeypatch.setattr(reporting, "_write_pdf", competing_writer)
    with pytest.raises(FileExistsError):
        render_report(result.findings, evidence, destination, run_id="audit", trace_id="trace")
    assert companion.read_text() == "Existing report"
    assert not destination.exists() and not destination.with_suffix(".review.json").exists()


def test_saved_report_cli_needs_no_provider_and_retains_source_hashes(report_case, tmp_path):
    result, _ = report_case
    source = tmp_path / "saved"
    source.mkdir()
    (source / "trace.json").write_text(result.trace.model_dump_json(), encoding="utf-8")
    (source / "findings.json").write_text(result.findings.model_dump_json(), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    destination = tmp_path / "audit.pdf"
    assert main([str(source), "--output", str(destination)]) == 0
    review = json.loads(destination.with_suffix(".review.json").read_text())
    assert review["source_trace_sha256"] == hashlib.sha256(before["trace.json"]).hexdigest()
    assert review["render_seconds"] < 60
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_failed_saved_audit_cannot_be_presented_as_completed(report_case, tmp_path):
    result, _ = report_case
    trace = result.trace.model_dump(mode="json")
    trace["status"] = "failed"
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="completed"):
        load_saved(tmp_path)


def test_completed_approval_requires_reviewed_edition(report_case, tmp_path):
    result, evidence = report_case
    destination = tmp_path / "approved.pdf"
    with pytest.raises(ValueError, match="reviewed report edition"):
        render_report(result.findings, evidence, destination, run_id="audit", trace_id="trace",
                      review_complete=True)
    assert not destination.exists()


def test_completed_approval_is_bound_to_source_hashes(report_case):
    from app.reporting import _apply_final_review
    result, evidence = report_case
    context = build_context(result.findings, evidence,
                            run_id="714405af53d549fba5a0a105c13a966d", trace_id="trace",
                            source_provenance={"source_trace_sha256": "changed",
                                               "source_findings_sha256": "changed"})
    with pytest.raises(ValueError, match="reviewed source artifacts"):
        _apply_final_review(context, evidence)
    assert context["human_review"] == "pending"
