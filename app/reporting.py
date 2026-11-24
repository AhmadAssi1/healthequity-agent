"""Evidence-bound draft reports from validated aggregates; no provider calls."""

import base64
import hashlib
import io
import json
import logging
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from xml.sax.saxutils import escape

from app.audit_schemas import AuditFindings
from app.claims_validation import validate_claims
from app.demo_model import describe_demo_model
from app.summary_grounding import ground_summary
from app.tool_contracts import AuditEvidence, RenderReportOutput

ROOT = Path(__file__).resolve().parents[1]
METHODOLOGY = ROOT / "docs/methodology.md"
TEMPLATE = Path(__file__).with_name("templates") / "audit_report.html.j2"
_WEB_PDF_LOCK = Lock()
METRIC_NAMES = {
    "demographic_parity_difference": "Demographic parity",
    "equalized_odds_difference": "Equalized odds",
    "ppv_difference": "Predictive parity (PPV)",
}
SEVERITY = {
    "substantial": "Substantial", "watch": "Watch",
    "no_evidence_found": "Below policy thresholds", "unavailable": "Unavailable",
}
DISCLAIMER = (
    "DRAFT - HUMAN REVIEW PENDING. This report summarizes statistical evidence. "
    "It is not a legal determination of discrimination or compliance and does not "
    "establish causation, clinical suitability, safety, or the absence of harm."
)


def label(value):
    if value is None:
        return "(missing value)"
    if value == "":
        return "(empty string)"
    return str(value) if isinstance(value, str) else json.dumps(value)


def rate(value):
    return "Unavailable" if value is None else f"{100 * value:.2f}%"


def pvalue(value):
    return "Unavailable" if value is None else f"{value:.4g}"


def methodology_blocks(text):
    """Parse the project's deliberately simple Markdown into shared blocks."""
    blocks, paragraph = [], []
    lines = text.splitlines()

    def flush():
        if paragraph:
            blocks.append({"kind": "p", "text": " ".join(paragraph)})
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush()
        elif line.startswith("#"):
            flush()
            if not line.startswith("# "):
                blocks.append({"kind": "heading", "text": line.lstrip("# ")})
        elif line.startswith("|"):
            flush()
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [c.strip() for c in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c) for c in cells):
                    rows.append(cells)
                index += 1
            blocks.append({"kind": "table", "rows": rows})
            continue
        elif line.startswith("- "):
            flush()
            bullet = line[2:]
            while index + 1 < len(lines) and lines[index + 1].startswith("  "):
                index += 1
                bullet += " " + lines[index].strip()
            blocks.append({"kind": "bullet", "text": bullet})
        else:
            paragraph.append(line)
        index += 1
    flush()
    return blocks


def revised_limitations(evidence):
    limitations = [
        ("Results describe the supplied observations and mapped attributes only. "
        "They do not establish fairness for other populations, unmeasured attributes, "
        "intersectional subgroups, or future time periods."),
        ("Observed outcomes, binary predictions, probability scores where supplied, and group "
        "labels must be accurate, aligned, and relevant to the same positive class."),
        ("Suppressed groups have insufficient evidence under the reporting policy. "
        "Their exclusion does not establish fairness or the absence of disparities."),
        ("Rate-test p-values are unadjusted for multiple testing. Omnibus tests do not establish "
        "which group pairs differ; no post-hoc pairwise comparisons are supplied."),
        ("Equalized odds has separate TPR and FPR tests, with no combined EOD p-value. "
        "A nonsignificant test does not establish equivalence or prove fairness."),
        ("Percentile bootstrap intervals for nonnegative maximum gaps may have positive "
        "lower bounds even when an observed gap is zero. They are not substitutes for "
        "rate tests; this report does not independently establish interval coverage."),
        ("The bootstrap treats observations as independent. Repeated stays for one patient "
        "may violate this assumption; patient-cluster-aware uncertainty is not implemented."),
        ("Severity is a configurable reporting convention, separate from statistical significance. "
        "The policy label no_evidence_found means below the configured gap rules, not proof of fairness."),
        ("A fixed decision threshold shapes the observed gaps. This report does not establish "
        "that the supplied threshold is clinically appropriate or optimal."),
    ]
    if any("Duplicate rows" in flag for flag in evidence.profile.data_quality_flags):
        limitations.append(
            "Repeated combinations in mapped audit columns were retained. Those combinations "
            "alone do not establish duplicate source encounters or identify repeated patients."
        )
    limitations.append(
        "Calibration curves cannot be reconstructed from this saved aggregate evidence. "
        "AUROC, when available, measures ranking and does not establish calibration."
        if evidence.profile.score_available else
        "No probability-score column was supplied. AUROC and score-based calibration analyses are unavailable."
    )
    return limitations


def _rates_chart(attribute, management=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    groups = attribute.groups
    if not groups:
        return None
    if management:
        return _management_rates_chart(attribute)
    figure, ax = plt.subplots(figsize=(7.2, 3.1 if management else 2.5))
    colors = [("tpr", "Readmissions detected (TPR)" if management else "True positive rate", "#147d92"),
              ("fpr", "False alarms (FPR)" if management else "False positive rate", "#c47a22")]
    for offset, (key, name, color) in enumerate(colors):
        values = [getattr(g, key).value for g in groups]
        positions = [i + (offset - 0.5) * 0.34 for i in range(len(groups))]
        labelled = False
        for pos, value in zip(positions, values, strict=True):
            if value is not None:
                ax.barh(pos, value, height=0.3, color=color, label=None if labelled else name)
                labelled = True
                if management:
                    ax.text(value + 0.005, pos, f"{value:.2%}", va="center", fontsize=9)
    names = [label(g.group_value) for g in groups]
    if management:
        names = [{"AfricanAmerican": "African American", "(empty string)": "Not recorded"}.get(n, n) for n in names]
    ax.set_yticks(range(len(groups)), names, fontsize=10 if management else 8)
    maximum = max((getattr(g, k).value or 0 for g in groups for k in ("tpr", "fpr")), default=0)
    limit = 0.5 if management and maximum <= 0.45 else 1
    ax.set_xlim(0, limit)
    if management:
        ax.set_xticks([i / 10 for i in range(int(limit * 10) + 1)])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis="x", labelsize=10 if management else 8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=8, frameon=False)
    figure.tight_layout(pad=1.0)
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=190, facecolor="white")
    plt.close(figure)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _management_rates_chart(attribute):
    """Aligned small multiples separate detection from false alarms."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    groups = attribute.groups
    names = [{"AfricanAmerican": "African American", "(empty string)": "Not recorded"}
             .get(label(g.group_value), label(g.group_value)) for g in groups]
    if attribute.attribute == "age":
        names = [f"{int(n[1:-1].split('-')[0])}-{int(n[1:-1].split('-')[1]) - 1}" for n in names]
    maximum = max((getattr(g, k).value or 0 for g in groups for k in ("tpr", "fpr")), default=0)
    limit = 0.5 if maximum <= 0.45 else 1
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, (key, title, color) in zip(axes, [
        ("tpr", "Readmissions detected (TPR)", "#327d94"),
        ("fpr", "False alarms (FPR)", "#c68b4d"),
    ], strict=True):
        for index, group in enumerate(groups):
            value = getattr(group, key).value
            if value is not None:
                ax.barh(index, value, height=0.44, color=color, zorder=3)
                ax.text(value + 0.009, index, f"{value:.2%}", va="center", fontsize=8,
                        color="#253747")
        ax.set_xlim(0, limit)
        step = 0.1 if limit == 0.5 else 0.2
        ax.set_xticks([i * step for i in range(round(limit / step) + 1)])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", color="#253747", pad=17)
        ax.set_yticks(range(len(groups)), names, fontsize=9)
        ax.tick_params(axis="both", length=0, labelsize=8, pad=7, colors="#496273")
        ax.grid(axis="x", color="#e5ebef", linewidth=0.6, zorder=0)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    axes[0].invert_yaxis()
    figure.subplots_adjust(left=0.22, right=0.985, bottom=0.16, top=0.83, wspace=0.29)
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=220, facecolor="white")
    plt.close(figure)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def build_context(findings, evidence, *, run_id, trace_id, model_metadata=None,
                  methodology_path=METHODOLOGY, source_provenance=None):
    findings = AuditFindings.model_validate(findings.model_dump())
    evidence = AuditEvidence.model_validate(evidence.model_dump())
    verdict = validate_claims(findings, evidence)
    if not verdict.passed:
        raise ValueError("Report input must pass evidence/claims validation.")
    grounded, _ = ground_summary(findings, evidence)
    source_provenance = {key: value for key, value in (source_provenance or {}).items()
                         if key in {"source_trace_sha256", "source_findings_sha256", "source_audit_started_at"}}
    from app.management_reporting import validate_report_metadata
    metadata = validate_report_metadata(model_metadata, findings.dataset.row_count, findings.dataset.group_columns)
    threshold = metadata.get("decision_threshold")
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                  or not 0 <= threshold <= 1):
        raise ValueError("Invalid supplied model decision threshold.")
    severity_counts = Counter(g.severity for a in findings.attributes for g in a.fairness_findings)
    blocks = [{"kind": "p", "text": (
        f"The audit covers {findings.dataset.row_count:,} observations, "
        f"{len(findings.attributes)} mapped attributes and "
        f"{sum(severity_counts.values())} attribute-metric comparisons. "
        f"Policy labels: {severity_counts['substantial']} substantial, "
        f"{severity_counts['watch']} watch, "
        f"{severity_counts['no_evidence_found']} below policy thresholds, "
        f"{severity_counts['unavailable']} unavailable."
    )}, {"kind": "p", "text": (
        "Use substantial and watch labels to prioritize further evidence review. "
        "The analysis does not determine why differences exist or recommend clinical decisions."
    )}, {"kind": "heading", "text": "Dataset profile and operating point"},
        {"kind": "table", "rows": [
            ["Item", "Recorded value"],
            ["Observations / input columns", f"{findings.dataset.row_count:,} / {findings.dataset.column_count}"],
            ["Mapped attributes", ", ".join(findings.dataset.group_columns)],
            ["Observed positive outcome rate", rate(evidence.profile.outcome_prevalence)],
            ["Probability score supplied", "Yes" if findings.dataset.score_available else "No"],
            ["Supplied decision threshold", "Not supplied; cannot infer from predictions" if threshold is None else f"{threshold:.6g}"],
            ["Supplied positive class", str(metadata.get("positive_class", "Not supplied"))],
            ["Supplied model run", str(metadata.get("model_run_id", "Not supplied"))],
            ["Groups eligible when", f"N >= {evidence.profile.suppression_policy.min_group_size}; actual positives >= {evidence.profile.suppression_policy.min_positive_outcomes}"],
        ]}, {"kind": "p", "text": (
            "Operating-point details are supplied provenance, not inferred or independently "
            "verified from the aggregate findings. Counts and fractions describe this audit only."
        )}]
    sections = [{"title": "Audit overview", "blocks": blocks}]
    rows = [["Attribute", "Fairness difference", "Gap (pp)", "95% CI (pp)", "Policy label"]]
    for attribute in findings.attributes:
        for gap in attribute.fairness_findings:
            ci = gap.confidence_interval
            rows.append([attribute.attribute, METRIC_NAMES[gap.metric],
                         "Unavailable" if gap.value is None else f"{gap.value * 100:.2f}",
                         "Unavailable" if ci is None else f"{ci.lower * 100:.2f} to {ci.upper * 100:.2f}",
                         SEVERITY[gap.severity]])
    sections.append({"title": "Headline fairness results", "blocks": [
        {"kind": "p", "text": "Gaps are maximum pairwise differences among eligible groups. Values below are percentage points (pp), not relative percentage changes."},
        {"kind": "table", "rows": rows},
        {"kind": "p", "text": (
            f"Actual policy: substantial when gap > {evidence.severity_policy.substantial_gap:.2f} and CI lower > 0; "
            f"watch when gap > {evidence.severity_policy.watch_gap:.2f} without meeting substantial. "
            "Labels do not establish statistical significance. Below policy thresholds is the display "
            "label for the source value no_evidence_found. Rounded displays do not change classifications."
        )},
        {"kind": "p", "text": "No combined EOD p-value is reported. Component tests appear on the attribute pages. Intervals and p-values must be interpreted separately."},
    ]})
    for attribute in findings.attributes:
        metrics = [["Group", "N", "Base rate", "Selected", "TPR", "FPR", "PPV", "AUROC"]]
        for group in attribute.groups:
            metrics.append([label(group.group_value), f"{group.count:,}",
                            *[rate(getattr(group, key).value) for key in ("base_rate", "selection_rate", "tpr", "fpr", "ppv")],
                            "Unavailable" if group.auroc.value is None else f"{group.auroc.value:.4f}"])
        attribute_blocks = [{"kind": "p", "text": (
            f"{len(attribute.groups)} reportable groups; {len(attribute.suppressed_groups)} suppressed groups. "
            "Rates are descriptive. AUROC measures ranking, not probability calibration."
        )}, {"kind": "table", "rows": metrics}]
        chart = _rates_chart(attribute)
        if chart:
            attribute_blocks.append({"kind": "chart", "data": chart, "text": (
                "True positive and false positive rates among reportable groups. "
                "Group-level confidence intervals were not estimated; bars are descriptive."
            )})
        tests = [["Rate", "Test", "Statistic", "p-value", "Min. expected cell"]]
        for test in attribute.statistical_tests:
            tests.append([test.rate, test.method.replace("_", " "),
                          "Unavailable" if test.statistic is None else f"{test.statistic:.4g}",
                          pvalue(test.p_value), "Unavailable" if test.min_expected_count is None else f"{test.min_expected_count:.2f}"])
        attribute_blocks += [{"kind": "heading", "text": "Component rate tests"}, {"kind": "table", "rows": tests}]
        trusted_attribute = next(a for a in evidence.metrics.results if a.attribute == attribute.attribute)
        if trusted_attribute.suppressed_groups:
            reasons = {"below_min_group_size": "below the minimum group size",
                       "below_min_positive_outcomes": "below the minimum positive-outcome count"}
            attribute_blocks.append({"kind": "heading", "text": "Suppressed groups: insufficient evidence"})
            grouped_reasons = {}
            for group in trusted_attribute.suppressed_groups:
                grouped_reasons.setdefault(tuple(group.reason_codes), []).append(label(group.group_value))
            for codes, names in grouped_reasons.items():
                attribute_blocks.append({"kind": "bullet", "text": ", ".join(names) + ": "
                                         + "; ".join(reasons.get(r, r) for r in codes) + "."})
        else:
            attribute_blocks.append({"kind": "p", "text": "No groups were suppressed for this attribute."})
        sections.append({"title": f"Attribute detail: {attribute.attribute}", "blocks": attribute_blocks})
    limitations = revised_limitations(evidence)
    sections.append({"title": "Data quality and interpretation limits", "blocks": [
        {"kind": "heading", "text": "Recorded data-quality warnings"},
        *[{"kind": "bullet", "text": flag} for flag in findings.warnings],
        *([] if findings.warnings else [{"kind": "p", "text": "No configured quality warnings were raised. This does not establish that the data are error-free."}]),
        {"kind": "heading", "text": "Limitations for human review"},
        *[{"kind": "bullet", "text": item} for item in limitations],
    ]})
    method_bytes = Path(methodology_path).read_bytes()
    sections.append({"title": "Methodology appendix", "blocks": [
        {"kind": "p", "text": "The following appendix is generated from the project's methodology.md. It documents defaults; the actual run policies and interval settings are recorded separately in this report."},
        *methodology_blocks(method_bytes.decode("utf-8")),
    ]})
    boot_settings = sorted({(b.total_resamples, b.seed, b.confidence_level) for b in evidence.bootstrap.results})
    sections.append({"title": "Provenance and review status", "blocks": [
        {"kind": "table", "rows": [
            ["Field", "Recorded value"], ["Run ID", run_id], ["Trace ID", trace_id],
            *[[key.replace("_", " ").capitalize(), str(value)]
              for key, value in (source_provenance or {}).items()],
            ["Methodology SHA-256", hashlib.sha256(method_bytes).hexdigest()],
            ["Bootstrap settings (resamples / seed / confidence)", "; ".join(f"{n:,} / {seed} / {level:.0%}" for n, seed, level in boot_settings)],
            ["Renderer", "ReportLab (portable PDF fallback); Jinja2 HTML companion"],
            ["Review status", "Draft; human narrative review pending"],
        ]},
        {"kind": "p", "text": "The quantitative summary is rendered from trusted evidence. Limitations are revised editorial text aligned with the documented method. Source model narratives are retained in the companion review record; no historical audit has been relabelled or overwritten."},
        {"kind": "p", "text": DISCLAIMER},
    ]})
    return {
        "title": "Healthcare algorithm fairness audit", "subtitle": "Binary classification / evidence-bound statistical review",
        "run_id": run_id, "trace_id": trace_id, "disclaimer": DISCLAIMER,
        "generated_at": datetime.now(UTC).strftime("%d %b %Y, %H:%M UTC"),
        "sections": sections, "original_summary": findings.summary,
        "grounded_summary": grounded.summary, "original_limitations": findings.limitations,
        "revised_limitations": limitations, "human_review": "pending",
        "source_provenance": source_provenance,
    }


def _apply_page_one_review(context, findings, evidence, metadata):
    """Apply the user's approved first-page edits; keep remaining review unfinished."""
    names = {"race": "race", "gender": "gender", "age": "age", "payer_code": "payer category"}
    # This editorial approval belongs to this saved readmission audit only.
    if context["run_id"] != "714405af53d549fba5a0a105c13a966d":
        raise ValueError("First-page editorial approval does not apply to this audit.")
    context.update(
        title="Readmission Prediction Model - Fairness Audit",
        subtitle="Hospital management report / Readmission within 30 days",
        review_banner="",
        disclaimer=("Scope: This statistical review does not establish the causes of observed "
                    "differences or certify clinical safety or regulatory compliance."),
        human_review="in_progress; page_1_approved; remaining_pages_not_yet_approved",
    )
    context["sections"][0] = {"title": "Executive summary", "blocks": [
        {"kind": "p", "text": (
            f"This review assessed a hospital readmission prediction model using {findings.dataset.row_count:,} "
            "hospital stays, comparing results across race, gender, age, and payer categories."
        )},
        {"kind": "p", "text": (
            "Differences exceeding the audit's configured thresholds were observed across race, age, "
            "and payer categories. The largest measured gap concerned differences in prediction error "
            "rates across age groups. Gender differences remained below the configured thresholds; "
            "this does not establish that the model is fair in all settings."
        )},
        {"kind": "p", "text": (
            "These findings identify priorities for further investigation. Before using this assessment "
            "to support operational decisions, the hospital should examine data quality, subgroup sample "
            "sizes, and how predictions would affect patient care."
        )},
        {"kind": "heading", "text": "Audit at a glance"},
        {"kind": "table", "rows": [
            ["Item", "Details"], ["Hospital stays assessed", f"{findings.dataset.row_count:,}"],
            ["Categories compared", ", ".join(names.get(c, c) for c in findings.dataset.group_columns)],
            ["Observed readmission rate (less than 30 days)", rate(evidence.profile.outcome_prevalence)],
            ["Prediction scores available", "Yes" if findings.dataset.score_available else "No"],
            ["Model flagging threshold", "Not supplied" if metadata.get("decision_threshold") is None else
             f"A stay is flagged when predicted readmission risk is {100 * metadata['decision_threshold']:.1f}% or higher."],
            ["Minimum data for group reporting", "At least 50 stays and 10 observed readmissions per group."],
        ]},
        {"kind": "p", "text": "The threshold and model information were supplied with the audit sample; they were not independently verified in this report."},
    ]}
    provenance = context["sections"][-1]
    rows = provenance["blocks"][0]["rows"]
    rows.insert(3, ["Supplied model run", str(metadata.get("model_run_id", "Not supplied"))])
    for row in rows:
        if row[0] == "Review status":
            row[1] = "Review in progress; page 1 approved; remaining pages not yet approved"
    for section in context["sections"]:
        for block in section["blocks"]:
            if block.get("text") == DISCLAIMER:
                block["text"] = context["disclaimer"]


def _apply_page_two_review(context, evidence):
    section = context["sections"][1]
    original = section["blocks"]
    rows = next(b["rows"] for b in original if b["kind"] == "table")
    rows[0] = ["Patient category", "Measure", "Gap (percentage points)", "95% confidence interval", "Review priority"]
    for row in rows[1:]:
        row[0] = {"race": "Race", "gender": "Gender", "age": "Age", "payer_code": "Payer category"}.get(row[0], row[0])
    # Preserve detailed statistical qualifications in the technical appendix.
    context["sections"][-2]["blocks"].extend([
        {"kind": "heading", "text": "Fairness result interpretation details"},
        *original[2:],
    ])
    section.update(title="Fairness Results by Patient Category", blocks=[
        {"kind": "p", "text": "Gaps compare eligible groups. Both gaps and confidence intervals are shown in percentage points, not relative percentage changes."},
        {"kind": "table", "rows": rows},
        {"kind": "p", "text": "Demographic parity: difference in the proportion of stays flagged by the model. Equalized odds: the larger of the gaps in true positive rates and false positive rates. Predictive parity (PPV): difference in the proportion of flagged stays that had the outcome."},
        {"kind": "p", "text": (
            f"Substantial: gap greater than {100 * evidence.severity_policy.substantial_gap:g} percentage points "
            "and the confidence interval's lower bound above zero. "
            f"Watch: gap greater than {100 * evidence.severity_policy.watch_gap:g} percentage points "
            "without meeting the substantial rule. Remaining available results are below policy thresholds."
        )},
        {"kind": "p", "text": "Priority labels follow configured audit rules; they do not establish clinical harm or statistical significance. Results below these thresholds do not prove fairness."},
        {"kind": "p", "text": "The age equalized-odds gap has a wide confidence interval (27.80 to 64.94 percentage points). Its estimated magnitude is uncertain; 43.38 is a point estimate, not an exact population difference."},
    ])
    context["human_review"] = "in_progress; pages_1_2_approved; remaining_pages_not_yet_approved"
    for row in context["sections"][-1]["blocks"][0]["rows"]:
        if row[0] == "Review status":
            row[1] = "Review in progress; pages 1-2 approved; remaining pages not yet approved"


def _apply_page_three_review(context, findings):
    attribute = next(a for a in findings.attributes if a.attribute == "race")
    section = next(s for s in context["sections"] if s["title"] == "Attribute detail: race")
    old = section["blocks"]
    rows = next(b["rows"] for b in old if b["kind"] == "table")
    rows[0] = ["Race category", "Stays (N)", "Readmitted", "Flagged", "TPR", "FPR", "PPV", "AUROC"]
    for row in rows[1:]:
        row[0] = {"AfricanAmerican": "African American", "(empty string)": "Not recorded"}.get(row[0], row[0])
    test_table = [b for b in old if b["kind"] == "table"][1]
    context["sections"][-2]["blocks"].extend([
        {"kind": "heading", "text": "Race: component rate tests"}, test_table,
        {"kind": "p", "text": "These are unadjusted multi-group tests. The 0.05 threshold is nominal; multiple comparisons limit interpretation."},
    ])
    section.update(title="Model Performance by Race", blocks=[
        {"kind": "p", "text": "Six categories met the reporting requirements; none were excluded. Not recorded denotes an empty race field in the source, retained as a separate category."},
        {"kind": "table", "rows": rows},
        {"kind": "chart", "data": _rates_chart(attribute, management=True), "height_ratio": 3.1 / 7.2,
         "text": "Readmissions detected: share of actual readmissions flagged. False alarms: share of stays without readmission flagged. Bars are descriptive; group-level confidence intervals were not estimated."},
        {"kind": "p", "text": "Readmitted: observed readmission rate. Flagged: share flagged by the model. PPV: share of flagged stays with actual readmission. AUROC measures ranking, not probability calibration."},
        {"kind": "p", "text": "Observed detection rates ranged from 20.00% to 43.81% across race categories, while false positive rates ranged from 11.34% to 26.53%. These descriptive differences warrant investigation; estimates from smaller groups should be interpreted cautiously."},
        {"kind": "p", "text": "At a nominal 0.05 level, the unadjusted multi-group tests indicate differences in flagging and false positive rates. TPR and PPV tests did not meet this threshold; this does not establish equality. Full test results are in the appendix."},
    ])
    context["human_review"] = "in_progress; pages_1_2_3_approved; remaining_pages_not_yet_approved"
    for row in context["sections"][-1]["blocks"][0]["rows"]:
        if row[0] == "Review status":
            row[1] = "Review in progress; pages 1-3 approved; remaining pages not yet approved"


def _apply_page_four_review(context, findings):
    attribute = next(a for a in findings.attributes if a.attribute == "gender")
    section = next(s for s in context["sections"] if s["title"] == "Attribute detail: gender")
    old = section["blocks"]
    tables = [b for b in old if b["kind"] == "table"]
    tables[0]["rows"][0] = ["Gender category", "Stays (N)", "Readmitted", "Flagged", "TPR", "FPR", "PPV", "AUROC"]
    context["sections"][-2]["blocks"].extend([
        {"kind": "heading", "text": "Gender: component rate tests"}, tables[1],
        {"kind": "p", "text": "These are unadjusted two-proportion tests. The 0.05 significance threshold is nominal; failure to meet it does not establish equivalence."},
    ])
    section.update(title="Model Performance by Gender", blocks=[
        {"kind": "p", "text": "Two categories met the reporting requirements. One category was excluded because it had insufficient data."},
        tables[0],
        {"kind": "chart", "data": _rates_chart(attribute, management=True), "height_ratio": 3.1 / 7.2,
         "text": "Readmissions detected: share of actual readmissions flagged. False alarms: share of stays without readmission flagged. Bars are descriptive; group-level confidence intervals were not estimated."},
        {"kind": "p", "text": "Readmitted: observed readmission rate. Flagged: share flagged by the model. PPV: share of flagged stays with actual readmission. AUROC measures ranking, not probability calibration."},
        {"kind": "p", "text": "Detection rates were 42.65% for female stays and 41.69% for male stays. False positive rates were 21.41% and 20.59%, respectively. The measured fairness gaps remained below the configured audit thresholds. This does not establish fairness in other populations or settings."},
        {"kind": "p", "text": "None of the four rate comparisons met the nominal 0.05 significance threshold. This does not prove that the groups perform equally. Full test results are in the appendix."},
        {"kind": "heading", "text": "Category excluded from comparisons"},
        {"kind": "p", "text": "The Unknown/Invalid category was excluded from comparisons because it did not meet the minimum sample and positive-outcome requirements. Findings therefore do not cover this category."},
    ])
    context["human_review"] = "in_progress; pages_1_2_3_4_approved; remaining_pages_not_yet_approved"
    for row in context["sections"][-1]["blocks"][0]["rows"]:
        if row[0] == "Review status":
            row[1] = "Review in progress; pages 1-4 approved; remaining pages not yet approved"


def _apply_page_five_review(context, findings):
    attribute = next(a for a in findings.attributes if a.attribute == "age")
    attribute = attribute.model_copy(update={"groups": sorted(
        attribute.groups, key=lambda g: int(g.group_value[1:].split("-")[0]))})
    section = next(s for s in context["sections"] if s["title"] == "Attribute detail: age")
    old = section["blocks"]
    tables = [b for b in old if b["kind"] == "table"]
    rows = tables[0]["rows"]
    ordered_rows = sorted(rows[1:], key=lambda row: int(row[0][1:].split("-")[0]))
    for row in ordered_rows:
        low, high = map(int, row[0][1:-1].split("-"))
        row[0] = f"{low}-{high - 1}"
    rows[:] = [["Age (years)", "Stays (N)", "Readmitted", "Flagged", "TPR", "FPR", "PPV", "AUROC"], *ordered_rows]
    context["sections"][-2]["blocks"].extend([
        {"kind": "heading", "text": "Age: component rate tests"}, tables[1],
        {"kind": "p", "text": "These are unadjusted multi-group tests. The 0.05 threshold is nominal; multiple comparisons limit interpretation."},
    ])
    section.update(title="Model Performance by Age Group", blocks=[
        {"kind": "p", "text": "Eight age categories met the reporting requirements; two were excluded. Age bands are displayed in ascending order and denote completed years (20-29 corresponds to the source interval [20-30))."},
        tables[0],
        {"kind": "chart", "data": _rates_chart(attribute, management=True), "height_ratio": 3.1 / 7.2,
         "text": "Readmissions detected: share of actual readmissions flagged. False alarms: share of stays without readmission flagged. Both panels use a 0-100% scale. Bars are descriptive; group-level confidence intervals were not estimated."},
        {"kind": "p", "text": "Readmitted: observed readmission rate. Flagged: share flagged by the model. PPV: share of flagged stays with actual readmission. AUROC measures ranking, not probability calibration."},
        {"kind": "p", "text": "Observed detection rates ranged from 34.88% among stays aged 90-99 to 78.26% among those aged 20-29. This large difference warrants investigation. The younger category has a smaller sample, and group-level confidence intervals were not estimated; the observed rates should therefore be interpreted cautiously."},
        {"kind": "p", "text": "All four unadjusted rate comparisons met the nominal 0.05 significance threshold. These results indicate differences across age categories but do not explain their causes or establish clinical harm. Full test results are in the appendix."},
        {"kind": "heading", "text": "Categories excluded from comparisons"},
        {"kind": "p", "text": "The 0-9 category did not meet the minimum sample or positive-outcome requirements. The 10-19 category did not meet the positive-outcome requirement. Both were excluded from comparisons; findings do not cover these categories."},
    ])
    context["human_review"] = "in_progress; pages_1_2_3_4_5_approved; remaining_pages_not_yet_approved"
    for row in context["sections"][-1]["blocks"][0]["rows"]:
        if row[0] == "Review status":
            row[1] = "Review in progress; pages 1-5 approved; remaining pages not yet approved"


def _apply_page_six_review(context, findings):
    attribute = next(a for a in findings.attributes if a.attribute == "payer_code")
    section = next(s for s in context["sections"] if s["title"] == "Attribute detail: payer_code")
    old = section["blocks"]
    tables = [b for b in old if b["kind"] == "table"]
    tables[0]["rows"][0] = ["Payer category", "Stays (N)", "Readmitted", "Flagged", "TPR", "FPR", "PPV", "AUROC"]
    for row in tables[0]["rows"][1:]:
        if row[0] == "(empty string)":
            row[0] = "Not recorded"
    context["sections"][-2]["blocks"].extend([
        {"kind": "heading", "text": "Payer category: component rate tests"}, tables[1],
        {"kind": "p", "text": "These are unadjusted multi-group tests at a nominal 0.05 threshold. Payer codes are retained because no authoritative code-to-label mapping was verified. Source documentation: docs/data_card.md; UCI dataset 296; Strack et al. (2014), DOI 10.1155/2014/781670."},
    ])
    missing = next(g.count for g in attribute.groups if g.group_value == "")
    section.update(title="Model Performance by Payer Category", blocks=[
        {"kind": "p", "text": "Ten categories met the reporting requirements; seven were excluded. Codes are retained as recorded; their full meanings have not been verified. Not recorded denotes an empty source field."},
        tables[0],
        {"kind": "chart", "data": _rates_chart(attribute, management=True), "height_ratio": 2.75 / 7.2,
         "text": "Both panels use a 0-100% scale. Bars are descriptive; group-level confidence intervals were not estimated."},
        {"kind": "p", "text": "TPR: share of actual readmissions flagged. FPR: share of stays without readmission flagged. Readmitted: observed rate. Flagged: share flagged. PPV: share of flagged stays with readmission. AUROC measures ranking, not calibration."},
        {"kind": "p", "text": "Detection rates ranged from 31.48% to 51.39%; false positive rates ranged from 10.30% to 26.38%. Differences warrant investigation but do not establish that payer category caused them."},
        {"kind": "p", "text": f"Data quality: payer information was not recorded for {missing:,} stays ({100 * missing / findings.dataset.row_count:.1f}% of this audit sample), limiting interpretation of payer comparisons."},
        {"kind": "p", "text": "At the nominal 0.05 level, unadjusted tests indicated differences in flagging and false positive rates. TPR and PPV tests did not meet this threshold; this does not establish equality. Details are in the appendix."},
        {"kind": "heading", "text": "Categories excluded from comparisons"},
        *[b for b in old if b["kind"] == "bullet"],
    ])
    context["human_review"] = "in_progress; pages_1_2_3_4_5_6_approved; remaining_pages_not_yet_approved"
    for row in context["sections"][-1]["blocks"][0]["rows"]:
        if row[0] == "Review status":
            row[1] = "Review in progress; pages 1-6 approved; remaining pages not yet approved"


def _apply_page_seven_review(context, findings):
    section = next(s for s in context["sections"] if s["title"] == "Data quality and interpretation limits")
    context["sections"][-2]["blocks"].extend([
        {"kind": "heading", "text": "Full data-quality warnings and interpretation limits"},
        *[{**b, "text": "Detailed interpretation limits"} if b.get("text") == "Limitations for human review" else b
          for b in section["blocks"]],
    ])
    payer = next(a for a in findings.attributes if a.attribute == "payer_code")
    missing = next(g.count for g in payer.groups if g.group_value == "")
    section.update(title="Data Quality and Limits of Interpretation", blocks=[
        {"kind": "heading", "text": "Data quality"},
        {"kind": "bullet", "text": "Repeated combinations were found in the mapped audit columns and retained. Matching audit values do not by themselves establish duplicate hospital stays or identify repeated patients."},
        {"kind": "bullet", "text": f"Payer information was not recorded for {missing:,} stays ({100 * missing / findings.dataset.row_count:.1f}% of the sample). This limits interpretation of comparisons by payer category."},
        {"kind": "bullet", "text": "Interpretation depends on accurate outcomes, predictions, probability scores, and category labels, aligned to the same hospital stays and target outcome."},
        {"kind": "heading", "text": "What the findings cover"},
        {"kind": "bullet", "text": "Results describe this sample and the categories assessed. They do not establish fairness in other populations, future periods, or combinations of patient characteristics."},
        {"kind": "bullet", "text": "Categories excluded for insufficient data remain unassessed. Their exclusion does not establish fairness or the absence of differences."},
        {"kind": "bullet", "text": "Observed differences do not establish their causes or clinical harm. Priority labels follow configured audit rules; results below those thresholds do not prove fairness."},
        {"kind": "heading", "text": "What needs further validation"},
        {"kind": "bullet", "text": "Probability calibration: saved aggregate results cannot establish whether predicted risks match observed readmission frequencies. AUROC measures ranking, not calibration."},
        {"kind": "bullet", "text": "Clinical use: the supplied decision threshold has not been established as clinically appropriate or optimal by this audit."},
        {"kind": "bullet", "text": "Statistical results require cautious interpretation: tests were not adjusted for multiple comparisons, and uncertainty estimates do not account for repeated stays by the same patient."},
        {"kind": "p", "text": "Detailed statistical qualifications and the original recorded data-quality warnings are retained in the methodology appendix."},
    ])
    context["human_review"] = "in_progress; pages_1_2_3_4_5_6_7_approved; remaining_pages_not_yet_approved"
    for row in context["sections"][-1]["blocks"][0]["rows"]:
        if row[0] == "Review status":
            row[1] = "Review in progress; pages 1-7 approved; remaining pages not yet approved"


def _apply_final_review(context, evidence):
    expected_sources = {
        "source_trace_sha256": "4570a82443bed110de0f39b38bcf8dc2b8237fc7d8ae06f46052046e76fbd2d2",
        "source_findings_sha256": "2242d05c3996dad632afa8ef7ae912d59f9739517c35837a8620fbd074c6f043",
    }
    if (context["run_id"] != "714405af53d549fba5a0a105c13a966d"
            or any(context["source_provenance"].get(k) != v for k, v in expected_sources.items())):
        raise ValueError("Completed editorial approval applies only to the reviewed source artifacts.")
    context["human_review"] = "completed; author_editorial_and_visual_review; not_clinical_certification"
    context["review_approval"] = {
        "reviewer": "Ahmad Assi", "approval_date": "2026-11-17",
        "scope": "Author visual and editorial approval of this report, including the technical appendix.",
        "verification": "Metrics, 12 bootstrap intervals and 16 rate tests reproduced from the audit CSV; methodology checked against implementation.",
        "exclusions": "Not independent clinical validation, hospital authorization, legal review or regulatory certification.",
    }
    appendix = context["sections"][-2]
    appendix["title"] = "Technical Appendix - Methodology and Detailed Results"
    policy = evidence.profile.suppression_policy
    settings = sorted({(b.total_resamples, b.seed, b.confidence_level) for b in evidence.bootstrap.results})
    appendix["blocks"][:1] = [
        {"kind": "p", "text": "This appendix documents the implemented method and detailed results. Sections describing defaults are general documentation; the table below records the settings actually used for this audit."},
        {"kind": "heading", "text": "Actual audit settings"},
        {"kind": "table", "rows": [
            ["Setting", "Value used"],
            ["Minimum group size / actual positive outcomes", f"{policy.min_group_size} / {policy.min_positive_outcomes}"],
            ["Watch / substantial gap thresholds", f"> {100 * evidence.severity_policy.watch_gap:g} / > {100 * evidence.severity_policy.substantial_gap:g} percentage points; substantial also requires CI lower bound > 0"],
            ["Bootstrap resamples / seed / confidence", "; ".join(f"{n:,} / {seed} / {level:.0%}" for n, seed, level in settings)],
            ["Rate tests", "Automatic choice from eligible counts; two-proportion z-test or chi-square used in this audit. P-values unadjusted for multiple testing."],
        ]},
    ]
    provenance = context["sections"][-1]
    provenance["title"] = "Report Provenance and Review Record"
    rows = provenance["blocks"][0]["rows"]
    for row in rows:
        if row[0] == "Review status":
            row[1] = "Author visual and editorial review completed; technical appendix included"
    rows.extend([
        ["Reviewed by / date", "Ahmad Assi / 17 Nov 2026"],
        ["Verification", context["review_approval"]["verification"]],
        ["Approval scope", context["review_approval"]["exclusions"]],
    ])
    context["appendix_headers"] = True


def _write_pdf(context, output):
    if context.get("presentation_version"):
        from app.compact_pdf import write_compact_pdf
        write_compact_pdf(context, output)
        return
    _write_a4_pdf(context, output)


def _write_a4_pdf(context, output):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="AuditBody", fontName="Helvetica", fontSize=9, leading=13,
                              spaceAfter=7, textColor=colors.HexColor("#253747")))
    styles.add(ParagraphStyle(name="AuditCell", fontName="Helvetica", fontSize=7.5, leading=10,
                              wordWrap="CJK", alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="AuditSmall", parent=styles["AuditBody"], fontSize=7.5, leading=10))
    styles["Title"].fontName = "Helvetica-Bold"
    styles["Title"].fontSize = 25
    styles["Title"].leading = 29
    styles["Title"].textColor = colors.HexColor("#11394e")
    styles["Heading1"].textColor = colors.HexColor("#11394e")
    styles["Heading1"].fontSize = 18
    styles["Heading1"].leading = 23
    styles["Heading2"].fontSize = 11
    styles["Heading2"].leading = 15
    styles["Heading2"].spaceBefore = 10
    styles["Heading2"].keepWithNext = True
    width, height = A4
    content_width = width - 88
    class ReportDocument(SimpleDocTemplate):
        def afterFlowable(self, flowable):
            if hasattr(flowable, "report_section"):
                self.report_section = flowable.report_section
                self.section_start_page = self.page

        def afterPage(self):
            page(self.canv, self)

    target = str(output) if isinstance(output, Path) else output
    document = ReportDocument(target, pagesize=A4, leftMargin=44, rightMargin=44,
                                 topMargin=53, bottomMargin=48,
                                 title=context["title"], author="HealthEquity Agent / Ahmad Assi")

    def para(text, style="AuditBody"):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    def table(rows):
        columns = len(rows[0])
        weights = {2: [0.33, 0.67], 3: [0.24, 0.34, 0.42],
                   5: [0.16, 0.26, 0.14, 0.21, 0.23],
                   8: [0.22, 0.09, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115]}.get(columns)
        if columns == 8:
            total = sum(weights)
            weights = [w / total for w in weights]
        widths = [content_width * w for w in weights] if weights else [content_width / columns] * columns
        result = Table([[para(c, "AuditCell") for c in row] for row in rows], colWidths=widths,
                       repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcebf0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f9")]),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#a6c2cd")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        return result

    story = [para(context["title"], "Title")]
    for key, style in (("subtitle", "AuditBody"), ("review_banner", "Heading2"),
                       ("disclaimer", "AuditSmall")):
        if context[key]:
            story.append(para(context[key], style))
    story.append(Spacer(1, 10))
    for index, section in enumerate(context["sections"]):
        if index:
            story.append(PageBreak())
        section_heading = para(section["title"], "Heading1")
        section_heading.report_section = section["title"]
        story.append(section_heading)
        for block in section["blocks"]:
            kind = block["kind"]
            if kind == "heading":
                story.append(para(block["text"], "Heading2"))
            elif kind == "table":
                story.extend([table(block["rows"]), Spacer(1, 9)])
            elif kind == "chart":
                ratio = block.get("height_ratio", 2.5 / 7.2)
                chart_width = min(content_width, 360 / ratio) if context.get("presentation_version") else content_width
                image = Image(io.BytesIO(base64.b64decode(block["data"])), width=chart_width,
                              height=chart_width * ratio)
                story.append(KeepTogether([image, para(block["text"], "AuditSmall")]))
            else:
                story.append(para(("- " if kind == "bullet" else "") + block["text"]))

    def page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#bdd0d9"))
        canvas.setLineWidth(0.6)
        canvas.line(44, height - 34, width - 44, height - 34)
        canvas.setFillColor(colors.HexColor("#31556a"))
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(44, height - 25, "HEALTHEQUITY AGENT")
        canvas.setFont("Helvetica", 7)
        header = context["review_banner"]
        section = getattr(doc, "report_section", "")
        if section and doc.page > doc.section_start_page:
            if context.get("appendix_headers") and section.startswith("Technical Appendix"):
                header = "Technical Appendix - continued"
            elif context.get("presentation_version") and section.startswith("Model Performance by "):
                attribute = section.removeprefix("Model Performance by ")
                header = (attribute if len(attribute) <= 65 else attribute[:62] + "...") + " - continued"
        canvas.drawRightString(width - 44, height - 25, header)
        canvas.drawString(44, 28, "Statistical evidence only / " + context["generated_at"])
        canvas.drawRightString(width - 44, 28, f"Page {doc.page}")
        canvas.restoreState()
    document.build(story)


def _apply_draft_presentation(context, findings, evidence, model_metadata):
    from app.management_reporting import apply_management_presentation

    apply_management_presentation(context, findings, evidence, model_metadata or {})


def _render_report_html(context, *, embedded=False):
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    environment = Environment(loader=FileSystemLoader(str(TEMPLATE.parent)),
                              autoescape=select_autoescape(default=True))
    return environment.get_template(TEMPLATE.name).render(**context, embedded=embedded)


def render_web_report(findings, evidence, *, run_id, trace_id, model_metadata=None, demo_training_rows=None):
    """Build the responsive report and its A4 download from the same evidence."""
    # Matplotlib uses shared state across concurrent Streamlit sessions.
    with _WEB_PDF_LOCK, io.BytesIO() as output:
        context = build_context(findings, evidence, run_id=run_id, trace_id=trace_id or "Unavailable",
                                model_metadata=model_metadata)
        context["review_banner"] = "DRAFT / HUMAN REVIEW PENDING"
        _apply_draft_presentation(context, findings, evidence, model_metadata)
        provenance = context["sections"][-1]["blocks"][0]["rows"]
        for row in provenance:
            if row[0] == "Renderer":
                row[1] = "ReportLab (A4 PDF) and responsive HTML"
        if trace_id is None:
            from app.local_audit import LOCAL_AUDIT_LABEL

            context["sections"][0]["blocks"].insert(0, {"kind": "p", "text": LOCAL_AUDIT_LABEL})
        if demo_training_rows is not None:
            description = describe_demo_model(demo_training_rows, findings.dataset.row_count)
            context["sections"][0]["blocks"].insert(0, {"kind": "p", "text": description})
        html = _render_report_html(context, embedded=True)
        try:
            _write_a4_pdf(context, output)
        except Exception as error:  # noqa: BLE001 - keep the report readable if export fails.
            logging.getLogger(__name__).warning("A4 export failed (%s)", type(error).__name__)
            return None, html
        return output.getvalue(), html


def render_report(findings, evidence, output, *, run_id, trace_id, model_metadata=None,
                  methodology_path=METHODOLOGY, source_provenance=None, page_one_reviewed=False,
                  page_two_reviewed=False, page_three_reviewed=False, page_four_reviewed=False,
                  page_five_reviewed=False, page_six_reviewed=False, page_seven_reviewed=False,
                  review_complete=False):
    """Write PDF, responsive HTML and review/provenance JSON from one context."""
    output = Path(output).resolve()
    if output.suffix.lower() != ".pdf":
        raise ValueError("PDF output must end in .pdf.")
    targets = [output, output.with_suffix(".html"), output.with_suffix(".review.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Report output already exists; choose a new destination.")
    context = build_context(findings, evidence, run_id=run_id, trace_id=trace_id,
                            model_metadata=model_metadata, methodology_path=methodology_path,
                            source_provenance=source_provenance)
    context["review_banner"] = "DRAFT / HUMAN REVIEW PENDING"
    if not any((page_one_reviewed, page_two_reviewed, page_three_reviewed, page_four_reviewed,
                page_five_reviewed, page_six_reviewed, page_seven_reviewed, review_complete)):
        _apply_draft_presentation(context, findings, evidence, model_metadata)
    if page_one_reviewed:
        _apply_page_one_review(context, findings, evidence, model_metadata or {})
    if page_two_reviewed:
        if not page_one_reviewed:
            raise ValueError("Second-page approval requires the reviewed first-page edition.")
        _apply_page_two_review(context, evidence)
    if page_three_reviewed:
        if not page_two_reviewed:
            raise ValueError("Third-page approval requires the reviewed second-page edition.")
        _apply_page_three_review(context, findings)
    if page_four_reviewed:
        if not page_three_reviewed:
            raise ValueError("Fourth-page approval requires the reviewed third-page edition.")
        _apply_page_four_review(context, findings)
    if page_five_reviewed:
        if not page_four_reviewed:
            raise ValueError("Fifth-page approval requires the reviewed fourth-page edition.")
        _apply_page_five_review(context, findings)
    if page_six_reviewed:
        if not page_five_reviewed:
            raise ValueError("Sixth-page approval requires the reviewed fifth-page edition.")
        _apply_page_six_review(context, findings)
    if page_seven_reviewed:
        if not page_six_reviewed:
            raise ValueError("Seventh-page approval requires the reviewed sixth-page edition.")
        _apply_page_seven_review(context, findings)
    if review_complete:
        if not page_seven_reviewed:
            raise ValueError("Completed approval requires the reviewed report edition.")
        _apply_final_review(context, evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    html = _render_report_html(context)
    with TemporaryDirectory(dir=output.parent) as staging_directory:
        staged_pdf = Path(staging_directory) / output.name
        _write_pdf(context, staged_pdf)
        staged_pdf.with_suffix(".html").write_text(html, encoding="utf-8")
        review = {key: context[key] for key in (
            "run_id", "trace_id", "generated_at", "original_summary", "grounded_summary",
            "original_limitations", "revised_limitations", "human_review",
        )}
        review["methodology_sha256"] = hashlib.sha256(Path(methodology_path).read_bytes()).hexdigest()
        review.update(context["source_provenance"])
        review["pdf_sha256"] = hashlib.sha256(staged_pdf.read_bytes()).hexdigest()
        if "presentation_version" in context:
            review["presentation_version"] = context["presentation_version"]
            review["supplied_metadata"] = context["supplied_metadata"]
        if "review_approval" in context or "presentation_version" in context:
            if "review_approval" in context:
                review["review_approval"] = context["review_approval"]
            review["editorial_sections"] = [
                {"title": s["title"], "blocks": [{k: v for k, v in b.items() if k != "data"}
                                                for b in s["blocks"]]}
                for s in context["sections"]
            ]
        staged_pdf.with_suffix(".review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # A competing render or a newly created companion must never be overwritten.
        reserved = []
        try:
            for path in targets:
                with path.open("xb"):
                    reserved.append(path)
            for path in [*targets[1:], output]:
                (staged_pdf.parent / path.name).replace(path)
        except OSError:
            for path in reserved:
                path.unlink(missing_ok=True)
            raise
    return RenderReportOutput(pdf_path=str(output), run_id=run_id, trace_id=trace_id)
