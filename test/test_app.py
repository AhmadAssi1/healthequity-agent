from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from lightgbm import LGBMClassifier
from openai import APIConnectionError
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from streamlit.testing.v1 import AppTest

from app import app as web
from app.agent import AuditAgent
from app.llm_config import LLMConfig
from evals.offline_provider import OfflineProvider

ENTRY = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture
def offline_audit(monkeypatch):
    def create_agent(*, provider, session, llm_config):
        return AuditAgent(provider=OfflineProvider(session), session=session, llm_config=llm_config)

    monkeypatch.setattr(web, "load_llm_config", lambda: LLMConfig(model_name="offline-scripted"))
    monkeypatch.setattr(web, "OpenAIProvider", lambda config: None)
    monkeypatch.setattr(web, "AuditAgent", create_agent)


def test_uploaded_csv_requires_attestation_and_keeps_results_for_the_current_mapping(offline_audit):
    app = AppTest.from_file(str(ENTRY), default_timeout=30).run()
    assert not app.exception
    assert app.header[0].value == "Audit"
    assert not app.file_uploader
    app.checkbox[0].check().run()
    rows = ("0,0,A\n1,1,A\n0,1,B\n1,0,B\n" * 30).encode()
    app.file_uploader[0].upload("audit.csv", b"outcome,prediction,gender\n" + rows).run()
    assert [field.value for field in app.selectbox] == ["outcome", "prediction", None]
    assert app.multiselect[0].value == ["gender"]
    app.button[0].click().run()
    assert not app.exception
    assert app.download_button[0].label == "Download draft PDF (A4)"
    findings, pdf, local, html = app.session_state["uploaded_audit_result"][2]
    assert not local
    assert findings.dataset.row_count == 120
    reader = PdfReader(BytesIO(pdf))
    assert all((float(page.mediabox.width), float(page.mediabox.height)) == pytest.approx(A4)
               for page in reader.pages)
    assert "Fairness Results by Group Attribute" in reader.pages[1].extract_text()
    assert 'class="he-audit-report management embedded"' in html
    assert 'data-label="95% confidence interval"' in html
    assert any(panel.label == "Read full report" for panel in app.expander)
    details = next(panel for panel in app.expander if panel.label == "Detailed audit summary")
    assert not details.proto.expanded
    assert details.text[0].value == findings.summary
    app.run()
    assert app.download_button
    app.selectbox[1].set_value("outcome").run()
    assert not app.download_button
    assert app.error[0].value == "Use a different CSV column for each role."
    app.checkbox[0].uncheck().run()
    assert not app.file_uploader


def test_unreadable_csv_is_rejected_before_showing_the_mapping_controls():
    app = AppTest.from_file(str(ENTRY), default_timeout=30).run()
    app.checkbox[0].check().run()
    app.file_uploader[0].upload("invalid.csv", b"outcome,outcome\n0,1\n").run()
    assert not app.exception
    assert app.error[0].value.startswith("Upload a readable UTF-8 CSV")
    assert not app.selectbox


def test_upload_without_predictions_trains_and_reports_on_held_out_rows(monkeypatch):
    fit = LGBMClassifier.fit
    predict = LGBMClassifier.predict_proba
    rows_seen = {}

    def record_fit(model, frame, outcome):
        rows_seen["train"] = set(frame.index)
        return fit(model, frame, outcome)

    def record_predict(model, frame):
        rows_seen["test"] = set(frame.index)
        return predict(model, frame)

    monkeypatch.setattr(LGBMClassifier, "fit", record_fit)
    monkeypatch.setattr(LGBMClassifier, "predict_proba", record_predict)
    app = AppTest.from_file(str(ENTRY), default_timeout=60).run()
    app.checkbox[0].check().run()
    rows = ("0,A,1,low\n1,A,8,high\n0,B,2,low\n1,B,9,high\n" * 160).encode()
    app.file_uploader[0].upload("features.csv", b"outcome,gender,measurement,category\n" + rows).run()
    features = next(field for field in app.multiselect if field.label == "Demo model features")
    assert features.options == ["measurement", "category"]
    features.set_value(["measurement", "category"]).run()
    app.checkbox[1].check().run()
    button = next(item for item in app.button if item.label == "Train demo model and audit")
    button.click().run()
    assert not app.exception and not app.error
    findings, pdf, local, html = app.session_state["uploaded_audit_result"][2]
    assert local and findings.dataset.score_available
    assert len(rows_seen["train"]) == 480 and len(rows_seen["test"]) == findings.dataset.row_count == 160
    assert rows_seen["train"].isdisjoint(rows_seen["test"])
    label = "Demo model trained on this upload."
    assert any(label in item.value for item in app.info)
    assert label in PdfReader(BytesIO(pdf)).pages[0].extract_text()
    assert label in html


def test_demo_uses_the_bundled_sample_and_renders_a_pdf(offline_audit):
    app = AppTest.from_string("from app.app import show_demo\nshow_demo()", default_timeout=60).run()
    assert app.header[0].value == "Demo"
    app.button[0].click().run()
    assert not app.exception
    assert app.download_button[0].label == "Download draft PDF (A4)"
    findings, pdf, local, _ = app.session_state["demo_audit_result"]
    assert not local
    assert findings.dataset.row_count == 14_998
    assert pdf.startswith(b"%PDF-")


@pytest.mark.parametrize("upload", [False, True], ids=["bundled", "upload"])
def test_local_only_deployment_audits_without_a_model_provider(monkeypatch, upload):
    monkeypatch.setenv("HEALTHEQUITY_LOCAL_ONLY", "1")
    monkeypatch.setattr(web, "OpenAIProvider", Mock(side_effect=AssertionError("Unexpected model provider")))
    if upload:
        app = AppTest.from_file(str(ENTRY), default_timeout=60).run()
        app.checkbox[0].check().run()
        rows = ("0,0,A\n1,1,A\n0,1,B\n1,0,B\n" * 30).encode()
        app.file_uploader[0].upload("audit.csv", b"outcome,prediction,gender\n" + rows).run()
    else:
        app = AppTest.from_string("from app.app import show_demo\nshow_demo()", default_timeout=60).run()

    assert all(item.label != "Run locally without OpenAI" for item in app.checkbox)
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.download_button[0].label == "Download draft PDF (A4)"
    assert any("Local statistical audit" in item.value for item in app.info)
    web.OpenAIProvider.assert_not_called()


@pytest.mark.parametrize("setup_fails", [False, True])
def test_connection_failure_still_produces_labeled_results_and_pdf(monkeypatch, setup_fails):
    class UnreachableProvider:
        model_name = "test-model"

        def generate(self, **kwargs):
            raise APIConnectionError(request=Mock())

    def create_provider(config):
        if setup_fails:
            raise ImportError("Using SOCKS proxy, but the 'socksio' package is not installed.")
        return UnreachableProvider()

    monkeypatch.setattr(web, "load_llm_config", lambda: LLMConfig(model_name="test-model"))
    monkeypatch.setattr(web, "OpenAIProvider", create_provider)
    app = AppTest.from_string("from app.app import show_demo\nshow_demo()", default_timeout=60).run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    expected_reason = "requires the socksio package" if setup_fails else "network connection to OpenAI"
    assert any(expected_reason in item.value for item in app.markdown)
    assert "Local statistical audit" in app.info[0].value
    assert app.download_button[0].label == "Download draft PDF (A4)"
    findings, pdf, local, html = app.session_state["demo_audit_result"]
    assert local
    assert findings.dataset.row_count == 14_998
    assert len(findings.attributes) == 4
    assert all(gap.confidence_interval for gap in findings.attributes[0].fairness_findings)
    assert "Local statistical audit" in PdfReader(BytesIO(pdf)).pages[0].extract_text()
    assert "Local statistical audit" in html
