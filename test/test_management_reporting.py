"""Management reports adapt to tasks and preserve unsupported/unavailable cases."""

import json
from copy import deepcopy

import pytest
from pypdf import PdfReader

from app.agent import AuditAgent
from app.llm_config import LLMConfig
from app.management_reporting import (
    apply_management_presentation,
    validate_report_metadata,
)
from app.reporting import build_context, render_report
from app.schemas import ColumnMapping
from app.tools import AuditToolSession
from evals.offline_provider import OfflineProvider
from fairness.scenarios import make_scenario


def offline_case(counts, *, score=False):
    frame = make_scenario(counts).rename(columns={"group": "region"})
    if score:
        frame["score"] = 0.2 + frame.prediction * 0.6
    mapping = ColumnMapping(outcome="outcome", prediction="prediction", groups=["region"],
                            score="score" if score else None)
    session = AuditToolSession(frame, mapping)
    result = AuditAgent(provider=OfflineProvider(session), session=session,
                        llm_config=LLMConfig(model_name="offline-scripted")).run()
    return result.findings, session.get_evidence()


@pytest.fixture(scope="module")
def case():
    return offline_case({"North": {"tn": 40, "fp": 10, "fn": 5, "tp": 45},
                         "South": {"tn": 45, "fp": 5, "fn": 25, "tp": 25}})


def context_for(case, metadata=None):
    findings, evidence = case
    context = build_context(findings, evidence, run_id="new-audit", trace_id="new-trace",
                            model_metadata=metadata)
    apply_management_presentation(context, findings, evidence, metadata or {})
    return context


@pytest.mark.parametrize("task,unit", [("Loan Default Prediction", "applications"),
                                      ("Equipment Failure Detection", "sensor observations")])
def test_new_task_pdf_uses_supplied_context_and_has_no_readmission_prose(case, tmp_path, task, unit):
    findings, evidence = case
    before = deepcopy((findings.model_dump(), evidence.model_dump()))
    metadata = {"row_count": 200, "task_name": task, "observation_unit": unit,
                "positive_class": "event recorded", "decision_threshold": 0.4,
                "model_name": "Example classifier", "model_run_id": "model-run-789",
                "group_display_names": {"region": "Operating region"}}
    path = tmp_path / "task.pdf"
    render_report(findings, evidence, path, run_id="new-audit", trace_id="new-trace", model_metadata=metadata)
    reader = PdfReader(path)
    text = " ".join(" ".join(page.extract_text().split()) for page in reader.pages)
    assert task in text and unit in text and "Operating region" in text and "40.00%" in text
    assert "Example classifier" in text and "model-run-789" in text
    assert all(term not in text.lower() for term in ("readmission", "hospital", "patient", "clinical", "ahmad assi"))
    review = json.loads(path.with_suffix(".review.json").read_text())
    assert review["human_review"] == "pending" and "review_approval" not in review
    assert review["supplied_metadata"]["task_name"] == task
    assert review["editorial_sections"][0]["title"] == "Executive summary"
    assert reader.metadata.producer.startswith("fpdf2 ")
    assert len(reader.outline) == len(review["editorial_sections"])
    headline = next(s for s in review["editorial_sections"] if s["title"] == "Fairness Results by Group Attribute")
    table = next(b for b in headline["blocks"] if b["kind"] == "table")
    for attribute, measure, gap, interval, priority in table["rows"][1:]:
        assert f"{attribute} / {measure} Gap (percentage points) {gap}" in text
        assert f"95% confidence interval {interval} Review priority {priority}" in text

    # Check the actual PDF's result text at phone width, including the numeric values.
    gap_values = {row[2] for row in table["rows"][1:]}
    sizes = []
    def collect_sizes(value, cm, tm, font, size):
        if value.strip() in gap_values:
            sizes.append(size * 360 / float(reader.pages[0].mediabox.width))
    for page in reader.pages:
        if "Gap (percentage points)" in page.extract_text():
            page.extract_text(visitor_text=collect_sizes)
    assert sizes and min(sizes) >= 13
    assert (findings.model_dump(), evidence.model_dump()) == before


def test_without_metadata_report_does_not_guess_task_or_positive_class(case):
    context = context_for(case)
    assert context["title"] == "Binary Classification Model - Fairness Audit"
    text = json.dumps(context["sections"])
    assert "task name was not supplied" in text
    assert "Not supplied; outcome 1 is the positive class" in text
    assert "Not supplied; cannot infer from binary predictions" in text
    assert "No probability-score column was supplied" in text


@pytest.mark.parametrize("updates", [
    {"task_type": "regression"}, {"task_type": "multiclass_classification"},
    {"task_name": " "}, {"positive_class": 1}, {"decision_threshold": float("nan")},
    {"decision_threshold": float("inf")}, {"row_count": True},
    {"group_display_names": {"unknown_column": "Region"}},
    {"group_display_names": {"region": ""}},
])
def test_incompatible_or_malformed_metadata_fails_before_render(case, tmp_path, updates):
    findings, evidence = case
    metadata = {"row_count": 200, **updates}
    with pytest.raises(ValueError):
        render_report(findings, evidence, tmp_path / "bad.pdf", run_id="new", trace_id="new",
                      model_metadata=metadata)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("metadata", [[], "task", 1])
def test_metadata_must_be_an_object(metadata):
    with pytest.raises(ValueError, match="JSON object"):
        validate_report_metadata(metadata, 200, ["region"])


def test_all_suppressed_is_unavailable_not_fair():
    case = offline_case({"Small A": {"tn": 4, "fp": 1, "fn": 1, "tp": 4},
                         "Small B": {"tn": 4, "fp": 1, "fn": 1, "tp": 4}})
    context = context_for(case)
    text = json.dumps(context["sections"])
    assert "0 eligible groups and 2 excluded groups" in text
    assert "3 as unavailable" in text
    assert "requires at least two available group rates" in text
    assert not any(b["kind"] == "chart" for s in context["sections"] for b in s["blocks"])


def test_undefined_ppv_is_not_zero_and_group_labels_remain_distinct():
    case = offline_case({"": {"tn": 50, "fp": 0, "fn": 50, "tp": 0},
                         "Missing": {"tn": 50, "fp": 0, "fn": 50, "tp": 0}})
    context = context_for(case)
    section = next(s for s in context["sections"] if s["title"] == "Model Performance by Region")
    table = next(b for b in section["blocks"] if b["kind"] == "table")
    assert {r[0] for r in table["rows"][1:]} == {"Empty (blank)", "Missing"}
    assert all(r[6] == "Unavailable" for r in table["rows"][1:])


def test_larger_group_sets_are_paginated_in_charts_without_hiding_groups():
    case = offline_case({f"Region {i}": {"tn": 40, "fp": 10, "fn": 10, "tp": 40} for i in range(11)})
    context = context_for(case)
    section = next(s for s in context["sections"] if s["title"] == "Model Performance by Region")
    charts = [b for b in section["blocks"] if b["kind"] == "chart"]
    assert [c["text"].split(".")[0] for c in charts] == [
        "Groups 1-4 of 11", "Groups 5-8 of 11", "Groups 9-11 of 11",
    ]
    table = next(b for b in section["blocks"] if b["kind"] == "table")
    assert len(table["rows"]) == 12


def test_display_metadata_is_escaped_in_html(case, tmp_path):
    findings, evidence = case
    path = tmp_path / "escaped.pdf"
    render_report(findings, evidence, path, run_id="new", trace_id="new",
                  model_metadata={"row_count": 200, "task_name": "<script>alert(1)</script>"})
    html = path.with_suffix(".html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_single_outcome_class_keeps_fpr_and_equalized_odds_unavailable():
    case = offline_case({"A": {"tn": 0, "fp": 0, "fn": 50, "tp": 50},
                         "B": {"tn": 0, "fp": 0, "fn": 25, "tp": 75}})
    context = context_for(case)
    section = next(s for s in context["sections"] if s["title"] == "Model Performance by Region")
    table = next(b for b in section["blocks"] if b["kind"] == "table")
    assert all(row[5] == "Unavailable" for row in table["rows"][1:])
    assert any("FPR: unavailable" in b.get("text", "") for b in section["blocks"])
    headline = context["sections"][1]["blocks"][1]["rows"]
    eod = next(row for row in headline[1:] if row[1] == "Equalized odds")
    assert eod[2] == eod[3] == eod[4] == "Unavailable"
