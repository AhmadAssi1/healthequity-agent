"""Streamlit entry point for uploaded and demonstration fairness audits."""

import csv
import hashlib
import json
import os
from io import StringIO
from pathlib import Path
from uuid import uuid4

import streamlit as st
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAIError
from pydantic import ValidationError

from app.agent import AuditAgent
from app.agent_errors import AgentRunError
from app.data import load_data
from app.demo_model import (
    DEMO_MODEL_NAME,
    DEMO_THRESHOLD,
    describe_demo_model,
    train_demo_predictions,
)
from app.llm_config import load_llm_config
from app.local_audit import LOCAL_AUDIT_LABEL, run_local_audit
from app.openai_provider import OpenAIProvider
from app.reporting import METHODOLOGY, METRIC_NAMES, SEVERITY, render_web_report
from app.schemas import ColumnMapping
from app.tools import AuditToolSession

DEMO_SAMPLE = Path(__file__).resolve().parents[1] / "data/samples/diabetes_audit_sample.csv"
DEMO_GROUPS = ("race", "gender", "age", "payer_code")


def select_columns(columns, *, file_key):
    """Suggest common names while leaving every assignment editable."""
    normalized = {column: column.strip().casefold().replace(" ", "_") for column in columns}
    fields = (
        ("Actual outcome", "outcome", ("outcome", "y_true", "actual_outcome")),
        ("Predicted label", "prediction", ("prediction", "y_pred", "predicted_label")),
        ("Probability score (optional)", "score", ("score", "y_score", "predicted_probability")),
    )
    mapping = {}
    for label, role, aliases in fields:
        matches = [index for index, column in enumerate(columns) if normalized[column] in aliases]
        mapping[role] = st.selectbox(
            label, columns, index=matches[0] if len(matches) == 1 else None,
            placeholder="Choose a column", key=f"{file_key}_{role}",
        )
    mapping["groups"] = st.multiselect(
        "Group attributes", columns,
        default=[column for column in columns if normalized[column] in (
            *DEMO_GROUPS, "age_group", "ethnicity", "sex"
        )],
        key=f"{file_key}_groups",
    )
    return mapping


def run_audit(frame, mapping, *, metadata=None, local_only=False, demo_features=None):
    """Run the existing audit agent and prepare the validated draft PDF."""
    status = st.status("Validating data and column assignments…", expanded=True)
    demo_training_rows = None
    if demo_features is not None:
        status.update(label="Training LightGBM demo model…")
        uploaded_rows = len(frame)
        try:
            frame, mapping = train_demo_predictions(frame, mapping, demo_features)
        except ValueError as error:
            status.update(label="Demo training could not start or complete", state="error")
            st.text(str(error))
            return None
        except (ImportError, OSError):
            status.update(label="Demo training unavailable", state="error")
            st.error("LightGBM could not load in this environment.")
            return None
        demo_training_rows = uploaded_rows - len(frame)
        metadata = {"row_count": len(frame), "model_name": DEMO_MODEL_NAME,
                    "decision_threshold": DEMO_THRESHOLD}
        status.write(describe_demo_model(demo_training_rows, len(frame)))
    try:
        session = AuditToolSession(frame, mapping)
    except ValidationError:
        status.update(label="Dataset validation failed", state="error")
        st.error("The dataset or column assignments do not meet the audit requirements.")
        return None
    except ValueError as error:
        status.update(label="Dataset validation failed", state="error")
        st.error("Review the dataset and column assignments.")
        st.text(str(error))
        return None

    status.write("Data and column assignments validated.")
    connection_failure = None
    if not local_only:
        status.update(label="Running the fairness audit…")
        try:
            config = load_llm_config()
            provider = OpenAIProvider(config)
            result = AuditAgent(provider=provider, session=session, llm_config=config).run()
        except ImportError as error:
            connection_failure = (
                "The configured SOCKS proxy requires the socksio package."
                if "socksio" in str(error)
                else "The OpenAI client could not load a required dependency."
            )
        except APIConnectionError:
            connection_failure = "A network connection to OpenAI could not be established."
        except AgentRunError as error:
            cause = error.__cause__
            if isinstance(cause, APITimeoutError):
                final_request = error.trace.turns and error.trace.turns[-1].structured_response_requested
                request = "final report request" if final_request else "request"
                connection_failure = f"The OpenAI {request} exceeded the {config.timeout_seconds:g}-second timeout."
            elif isinstance(cause, APIConnectionError):
                connection_failure = "A network connection to OpenAI could not be established."
            else:
                status.update(label="Audit could not be completed", state="error")
                if isinstance(cause, APIStatusError):
                    st.error(f"The model service returned HTTP {cause.status_code}.")
                else:
                    st.error(f"The audit stopped during analysis or findings validation ({type(cause).__name__}).")
                st.caption(f"Audit reference: {error.trace.run_id}")
                return None
        except (ValueError, OpenAIError):
            status.update(label="Audit service unavailable", state="error")
            st.error("The audit service could not start. Check the server's model configuration.")
            return None

    local = local_only or connection_failure is not None
    if local:
        if connection_failure is not None:
            status.write(f"{connection_failure} Continuing with the local statistical audit.")
        status.update(label="Running the local statistical audit…")
        try:
            findings = run_local_audit(session)
        except ValueError:
            status.update(label="Local audit could not be completed", state="error")
            st.error("The audit could not produce validated local findings for these inputs.")
            return None
        trace_id = None
        run_id = uuid4().hex
    else:
        findings = result.findings
        trace_id = run_id = result.trace.run_id
        status.text(f"OpenAI agent completed the audit using {config.model_name}.")

    status.write("Fairness metrics and findings validated.")
    status.update(label="Preparing the draft PDF…")
    try:
        pdf, html = render_web_report(
            findings, session.get_evidence(), run_id=run_id, trace_id=trace_id,
            model_metadata=metadata, demo_training_rows=demo_training_rows,
        )
    except Exception:  # noqa: BLE001 - rendering is optional once validated findings exist.
        status.update(label="Audit complete. Report generation failed.", state="error")
        return findings, None, local, None
    label = "Audit and draft PDF ready" if pdf is not None else "Audit and online report ready"
    status.update(label=label, state="complete", expanded=False)
    return findings, pdf, local, html


def show_results(result, *, demo=False, demo_training_rows=None):
    findings, pdf, local, *report = result
    html = report[0] if report else None
    st.subheader("Audit results")
    if local:
        st.info(f"{LOCAL_AUDIT_LABEL}.")
    if demo_training_rows is not None:
        st.info(describe_demo_model(demo_training_rows, findings.dataset.row_count))
    st.caption(f"{findings.dataset.row_count:,} observations · {len(findings.attributes)} group attributes")
    rows = []
    for attribute in findings.attributes:
        for gap in attribute.fairness_findings:
            interval = gap.confidence_interval
            rows.append({
                "Attribute": attribute.attribute,
                "Metric": METRIC_NAMES[gap.metric],
                "Gap (pp)": "Unavailable" if gap.value is None else f"{gap.value * 100:.2f}",
                "95% CI (pp)": (
                    "Unavailable" if interval is None
                    else f"{interval.lower * 100:.2f} to {interval.upper * 100:.2f}"
                ),
                "Review priority": SEVERITY[gap.severity],
            })
    st.dataframe(rows, hide_index=True, width="stretch")
    with st.expander("Group coverage and limitations"):
        for attribute in findings.attributes:
            st.text(f"{attribute.attribute}: {len(attribute.groups)} reportable, "
                    f"{len(attribute.suppressed_groups)} suppressed")
        for item in [*findings.warnings, *findings.limitations]:
            st.text(f"• {item}")
    if pdf is None:
        st.warning("The draft PDF could not be generated. Your validated results are available above.")
    else:
        st.download_button(
            "Download draft PDF (A4)" if report else "Download draft PDF", pdf,
            mime="application/pdf", on_click="ignore",
            file_name="healthequity-demo-audit.pdf" if demo else "healthequity-audit.pdf",
        )
    if html:
        with st.expander("Read full report"):
            st.html(html)
    with st.expander("Detailed audit summary", expanded=False):
        st.text(findings.summary)


def use_local_audit(key):
    return os.getenv("HEALTHEQUITY_LOCAL_ONLY") == "1" or st.checkbox(
        "Run locally without OpenAI", key=key,
    )


def show_audit():
    st.header("Audit")
    st.caption("Use a CSV with binary outcomes (0 or 1) and at least one group attribute. "
               "Supply predictions or select features to train a demo model.")
    attested = st.checkbox("I confirm this file contains only synthetic or de-identified data.")
    if not attested:
        st.session_state.pop("uploaded_audit_result", None)
        return
    uploaded_file = st.file_uploader("Upload a CSV", type=["csv"], max_upload_size=25)
    if uploaded_file is None:
        st.session_state.pop("uploaded_audit_result", None)
        return

    contents = uploaded_file.getvalue()
    try:
        with StringIO(contents.decode("utf-8-sig"), newline="") as stream:
            frame = load_data(stream)
    except (csv.Error, ValueError):
        st.session_state.pop("uploaded_audit_result", None)
        st.error("Upload a readable UTF-8 CSV with a header, unique column names and data rows.")
        return
    if frame.empty:
        st.session_state.pop("uploaded_audit_result", None)
        st.error("The CSV must contain at least one data row.")
        return

    file_key = hashlib.sha256(contents).hexdigest()
    st.caption(f"{len(frame):,} rows · {len(frame.columns)} columns")
    st.subheader("Map columns")
    mapping = select_columns(frame.columns.tolist(), file_key=file_key)
    outcome, prediction, score, groups = (mapping[key] for key in ("outcome", "prediction", "score", "groups"))
    assigned = [column for column in (outcome, prediction, score, *groups) if column is not None]
    demo_features = None
    if prediction is None and outcome and groups:
        demo_features = st.multiselect(
            "Demo model features", [column for column in frame.columns if column not in assigned],
            help="Choose inputs available before the outcome. Leave out identifiers and outcome-derived fields.",
            key=f"{file_key}_demo_features",
        )
        st.caption("The demo trains on about 75% of rows and audits only the remaining 25%. "
                   "Use one row per subject to keep subjects separate.")
    selection = mapping if demo_features is None else mapping | {"demo_features": demo_features}
    existing = st.session_state.get("uploaded_audit_result")
    if existing and (existing[0], existing[1]) != (file_key, selection):
        st.session_state.pop("uploaded_audit_result")
        existing = None

    if not (outcome and groups):
        st.info("Choose the actual outcome and at least one group attribute.")
        return
    if len(assigned) != len(set(assigned)):
        st.error("Use a different CSV column for each role.")
        return
    if demo_features == []:
        st.info("Select at least one demo model feature, or map a predicted label to audit existing predictions.")
        return
    try:
        selected_columns = ColumnMapping(**mapping)
    except ValidationError:
        st.error("Each mapped CSV column must have a nonblank name.")
        return
    local_only = use_local_audit("upload_local_only")
    button_label = "Train demo model and audit" if demo_features is not None else "Run audit"
    if st.button(button_label, type="primary"):
        result = run_audit(frame, selected_columns, local_only=local_only, demo_features=demo_features)
        if result is None:
            st.session_state.pop("uploaded_audit_result", None)
            return
        existing = (file_key, selection, result)
        st.session_state["uploaded_audit_result"] = existing
    if existing:
        result = existing[2]
        training_rows = len(frame) - result[0].dataset.row_count if demo_features is not None else None
        show_results(result, demo_training_rows=training_rows)


def show_demo():
    st.header("Demo")
    st.write("Try the audit on a bundled hospital readmission sample with precomputed predictions.")
    local_only = use_local_audit("demo_local_only")
    if st.button("Run demo audit", type="primary"):
        st.session_state.pop("demo_audit_result", None)
        try:
            frame = load_data(DEMO_SAMPLE)
            metadata = json.loads(DEMO_SAMPLE.with_name(f"{DEMO_SAMPLE.stem}_metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st.error("The demonstration dataset or its metadata is unavailable.")
            return
        mapping = ColumnMapping(outcome="outcome", prediction="prediction", score="score", groups=list(DEMO_GROUPS))
        st.session_state["demo_audit_result"] = run_audit(frame, mapping, metadata=metadata, local_only=local_only)
    result = st.session_state.get("demo_audit_result")
    if result is not None:
        show_results(result, demo=True)


def show_monitoring():
    st.header("Monitoring")
    st.info("Audit history and drift monitoring will be available here in a future release.")


def show_methodology():
    st.header("Methodology")
    content = METHODOLOGY.read_text(encoding="utf-8")
    st.markdown(content.partition("\n")[2].lstrip())


def main():
    st.set_page_config(page_title="HealthEquity Agent", layout="wide")
    pages = [
        st.Page(show_audit, title="Audit", default=True),
        st.Page(show_demo, title="Demo", url_path="demo"),
        st.Page(show_monitoring, title="Monitoring", url_path="monitoring"),
        st.Page(show_methodology, title="Methodology", url_path="methodology"),
    ]
    page = st.navigation(pages)
    st.title("HealthEquity Agent")
    page.run()
