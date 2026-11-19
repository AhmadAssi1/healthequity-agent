"""Task-neutral management presentation derived from validated audit evidence."""

import base64
import io
import math
import re
import textwrap
from collections import Counter


def validate_report_metadata(metadata, row_count, group_columns):
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("Report metadata must be a JSON object.")  # noqa: TRY004
    if type(metadata.get("row_count")) is not int or metadata["row_count"] != row_count:
        raise ValueError("Supplied model provenance row count does not match this report.")
    if metadata.get("task_type", "binary_classification") != "binary_classification":
        raise ValueError("Reports support binary_classification tasks only.")
    for key in ("task_name", "model_name", "positive_class", "observation_unit", "model_run_id"):
        if key in metadata and (not isinstance(metadata[key], str) or not metadata[key].strip()
                                or len(metadata[key]) > 200):
            raise ValueError(f"Metadata {key} must be nonblank text of at most 200 characters.")
    threshold = metadata.get("decision_threshold")
    if threshold is not None and (type(threshold) not in (int, float)
                                  or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Invalid supplied model decision threshold.")
    names = metadata.get("group_display_names", {})
    if (not isinstance(names, dict) or any(k not in group_columns for k in names)
            or any(not isinstance(v, str) or not v.strip() or len(v) > 100 for v in names.values())
            or len(set(names.values())) != len(names)):
        raise ValueError("Group display names must uniquely label mapped group columns.")
    resolved = [names.get(k, k.replace("_", " ").capitalize()) for k in group_columns]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Group display names must be distinct.")
    return metadata.copy()


def group_label(value):
    if value is None:
        return "Missing (null)"
    if value == "":
        return "Empty (blank)"
    return str(value)


def ordered_groups(groups):
    """Order explicit numeric intervals without guessing their real-world unit."""
    pattern = r"\[(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)\)"
    if groups and all(isinstance(g.group_value, str) and re.fullmatch(pattern, g.group_value) for g in groups):
        return sorted(groups, key=lambda g: float(re.fullmatch(pattern, g.group_value)[1]))
    return list(groups)


def _charts(groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    largest = max((getattr(g, k).value or 0 for g in groups for k in ("tpr", "fpr")), default=0)
    limit = 0.5 if largest <= 0.45 else 1.0
    for start in range(0, len(groups), 4):
        batch = groups[start:start + 4]
        width = 3.3
        panel_height = 0.36 * len(batch) + 0.14
        panel_gap = 0.64
        top_margin, bottom_margin = 0.29, 0.23
        height = 2 * panel_height + panel_gap + top_margin + bottom_margin
        figure, axes = plt.subplots(2, 1, figsize=(width, height))
        for ax, (key, title, color) in zip(axes, [
            ("tpr", "True positive rate (TPR)", "#267488"),
            ("fpr", "False positive rate (FPR)", "#965528"),
        ], strict=True):
            for i, group in enumerate(batch):
                value = getattr(group, key).value
                if value is None:
                    ax.text(0.01, i, "Unavailable", va="center", fontsize=10)
                    continue
                ax.barh(i, value, height=0.64, color=color, zorder=3)
                inside = value > limit * 0.64
                ax.text(value - 0.012 * limit if inside else value + 0.025 * limit, i, f"{value:.2%}",
                        va="center", ha="right" if inside else "left", fontsize=10, zorder=4,
                        color="white" if inside else "#253747")
            labels = ["\n".join(textwrap.wrap(group_label(g.group_value), 16, max_lines=2, placeholder="…"))
                      for g in batch]
            ax.set_yticks(range(len(batch)), labels)
            ax.set_ylim(len(batch) - 0.5, -0.5)
            ax.set_xlim(0, limit)
            ax.set_xticks([0, limit / 2, limit])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=5)
            ax.tick_params(axis="both", length=0, labelsize=9.5, pad=3, colors="#384e5b")
            ax.grid(axis="x", color="#e5ebef", linewidth=0.6)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        figure.subplots_adjust(left=0.35, right=0.94, bottom=bottom_margin / height,
                               top=1 - top_margin / height, hspace=panel_gap / panel_height)
        stream = io.BytesIO()
        figure.savefig(stream, format="png", dpi=200, facecolor="white")
        plt.close(figure)
        yield {"kind": "chart", "data": base64.b64encode(stream.getvalue()).decode("ascii"),
               "height_ratio": height / width,
               "text": (f"Groups {start + 1}-{start + len(batch)} of {len(groups)}. "
                        f"Both panels use a 0–{limit:.0%} scale. Bars show observed rates. "
                        "Group-level confidence intervals were not estimated. "
                        "Full group labels appear with the recorded rates.")}


def _neutral(text):
    return (text.replace("The demonstration task is hospital readmission within fewer than 30 days.", "")
            .replace("across demographic groups", "across mapped groups")
            .replace("such as a hospital stay", "with a supplied outcome and prediction")
            .replace("Repeated stays from the same patient", "Repeated observations from the same entity")
            .replace("repeated stays may require a patient-level bootstrap", "repeated observations may require a cluster bootstrap")
            .replace("repeated stays for one patient", "repeated observations for one entity")
            .replace("Repeated stays for one patient", "Repeated observations for one entity")
            .replace("patient-cluster-aware", "entity-cluster-aware")
            .replace("resamples stays, not patient clusters", "resamples observations, not entity clusters")
            .replace("duplicate source encounters or identify repeated patients", "duplicate source records or identify repeated entities")
            .replace("clinically appropriate", "appropriate for the intended use")
            .replace("clinical suitability", "suitability for operational use")
            .replace("Historical readmission outcomes may reflect differences in access to care as well as patient health.",
                     "Historical outcomes may reflect measurement practices and external conditions as well as the target phenomenon."))


def apply_management_presentation(context, findings, evidence, metadata):
    """Replace presentation, retaining validated values and a pending review status."""
    from app.reporting import METRIC_NAMES, SEVERITY, pvalue, rate

    names = metadata.get("group_display_names", {})
    def display(key):
        return names.get(key, key.replace("_", " ").capitalize())
    old_sections = context["sections"]
    appendix = old_sections[-2]
    for block in appendix["blocks"]:
        if "text" in block:
            block["text"] = _neutral(block["text"])
    appendix["title"] = "Technical Appendix - Methodology and Detailed Results"
    appendix["blocks"][0] = {"kind": "p", "text": "The appendix presents the implemented binary-classification method in task-neutral language. General defaults are distinguished from the actual settings below. The methodology source hash is retained in the provenance record."}
    policy = evidence.profile.suppression_policy
    settings = sorted({(b.total_resamples, b.seed, b.confidence_level) for b in evidence.bootstrap.results})
    appendix["blocks"].insert(1, {"kind": "table", "rows": [
        ["Actual setting", "Value used"],
        ["Minimum observations / positive outcomes", f"{policy.min_group_size} / {policy.min_positive_outcomes}"],
        ["Watch / substantial gap thresholds", f"> {100 * evidence.severity_policy.watch_gap:g} / > {100 * evidence.severity_policy.substantial_gap:g} percentage points; substantial also requires CI lower bound > 0"],
        ["Bootstrap resamples / seed / confidence", "; ".join(f"{n:,} / {seed} / {level:.0%}" for n, seed, level in settings)],
    ]})
    task = metadata.get("task_name")
    unit = metadata.get("observation_unit", "observations")
    positives = metadata.get("positive_class", "Not supplied; outcome 1 is the positive class")
    threshold = metadata.get("decision_threshold")
    counts = Counter(g.severity for a in findings.attributes for g in a.fairness_findings)
    substantial = [display(a.attribute) for a in findings.attributes if any(g.severity == "substantial" for g in a.fairness_findings)]
    attribute_count = len(findings.attributes)
    attribute_label = "attribute" if attribute_count == 1 else "attributes"
    summary = [
        {"kind": "p", "text": f"This audit assessed {findings.dataset.row_count:,} {unit} across {attribute_count} mapped {attribute_label}. " + (f"The supplied task is {task}." if task else "The task name was not supplied; results describe binary outcomes and predictions.")},
        {"kind": "p", "text": f"Configured audit rules classified {counts['substantial']} comparisons as substantial, {counts['watch']} as watch, {counts['no_evidence_found']} as below policy thresholds, and {counts['unavailable']} as unavailable. " + ("Attributes with substantial differences: " + ", ".join(substantial) + "." if substantial else "No comparison met the substantial rule; this does not establish fairness.")},
        {"kind": "p", "text": "Use available gaps, uncertainty intervals and data quality to prioritize investigation. Differences do not establish their causes or suitability for operational use. Unavailable comparisons and excluded categories remain unassessed."},
        {"kind": "heading", "text": "Audit at a glance"},
        {"kind": "table", "rows": [
            ["Item", "Details"], ["Observations assessed", f"{findings.dataset.row_count:,}"],
            ["Mapped attributes", ", ".join(display(a) for a in findings.dataset.group_columns)],
            ["Supplied observation unit", metadata.get("observation_unit", "Not supplied")],
            ["Supplied positive outcome", positives],
            ["Observed positive outcome rate", rate(evidence.profile.outcome_prevalence)],
            ["Probability scores supplied", "Yes" if findings.dataset.score_available else "No"],
            ["Supplied decision threshold", "Not supplied; cannot infer from binary predictions" if threshold is None else f"{threshold:.6g} ({threshold:.2%})"],
            ["Minimum group reporting requirements", f"{policy.min_group_size} observations and {policy.min_positive_outcomes} actual positive outcomes"],
        ]},
        {"kind": "p", "text": "Task descriptions, model identity and operating threshold are supplied metadata, not independently verified by this renderer. Binary predictions are audited as supplied; the threshold is not reapplied."},
    ]
    headline = old_sections[1]
    headline["title"] = "Fairness Results by Group Attribute"
    rows = headline["blocks"][1]["rows"]
    rows[0] = ["Attribute", "Measure", "Gap (percentage points)", "95% confidence interval", "Review priority"]
    for row, (attribute, gap) in zip(rows[1:], [(a, g) for a in findings.attributes for g in a.fairness_findings], strict=True):
        row[0] = display(attribute.attribute)
        assert row[1] == METRIC_NAMES[gap.metric] and row[4] == SEVERITY[gap.severity]
    appendix["blocks"].extend([
        {"kind": "heading", "text": "Detailed fairness interpretation"},
        headline["blocks"][2],
        {"kind": "p", "text": "The tables below report the TPR and FPR tests separately. Interpret these p-values and the confidence intervals for fairness gaps separately."},
    ])
    headline["blocks"] = [headline["blocks"][0], {"kind": "table", "role": "headline", "rows": rows},
        {"kind": "p", "text": "Demographic parity compares predicted-positive rates. Equalized odds is the larger of the true-positive-rate gap and false-positive-rate gap. Predictive parity (PPV) compares the share of positive predictions with actual positive outcomes."},
        {"kind": "p", "text": f"Substantial requires a gap above {100 * evidence.severity_policy.substantial_gap:g} percentage points and a confidence-interval lower bound above zero. Watch requires a gap above {100 * evidence.severity_policy.watch_gap:g} percentage points without meeting the substantial rule. Missing gaps or intervals are unavailable."},
        {"kind": "p", "text": "Priority labels are configurable reporting rules, separate from statistical significance. Below-threshold results do not prove fairness. Confidence intervals describe uncertainty, not an exact population difference."}]
    sections = [{"title": "Executive summary", "blocks": summary}, headline]
    for attribute in findings.attributes:
        groups = ordered_groups(attribute.groups)
        rows = [["Group", "Count", "Outcome rate", "Flagged", "TPR", "FPR", "PPV", "AUROC"]]
        for g in groups:
            rows.append([group_label(g.group_value), f"{g.count:,}", *[rate(getattr(g, k).value) for k in ("base_rate", "selection_rate", "tpr", "fpr", "ppv")], "Unavailable" if g.auroc.value is None else f"{g.auroc.value:.4f}"])
        excluded_count = len(attribute.suppressed_groups)
        eligible_label = "group" if len(groups) == 1 else "groups"
        excluded_label = "group" if excluded_count == 1 else "groups"
        group_note = f"{len(groups)} eligible {eligible_label} and {excluded_count} excluded {excluded_label}."
        if any(group.group_value in (None, "") for group in groups):
            group_note += " Empty (blank) and Missing (null) represent distinct source values."
        blocks = [{"kind": "p", "text": group_note}, {"kind": "table", "role": "group_rates", "rows": rows}]
        blocks.extend(_charts(groups))
        blocks.append({"kind": "p", "text": "Outcome rate: share with actual positive outcomes. Flagged: share predicted positive. TPR: share of actual positives detected. FPR: share of actual negatives predicted positive. PPV: share of positive predictions with actual positive outcomes. AUROC measures ranking, not probability calibration."})
        for group in groups:
            for key in ("tpr", "fpr", "ppv"):
                metric = getattr(group, key)
                if metric.value is None:
                    blocks.append({"kind": "p", "text": f"{group_label(group.group_value)} / {key.upper()}: unavailable. {metric.reason}"})
        for key, title in (("tpr", "Detection rates"), ("fpr", "False positive rates")):
            values = [getattr(g, key).value for g in groups if getattr(g, key).value is not None]
            if len(values) >= 2:
                description = f"{title} among eligible groups ranged from {min(values):.2%} to {max(values):.2%}."
            else:
                description = f"A range for {title.lower()} requires at least two available group rates."
            blocks.append({"kind": "p", "text": description})
        rate_names = {"selection_rate": "Predicted-positive rate", "tpr": "TPR", "fpr": "FPR", "ppv": "PPV"}
        significant = [rate_names[t.rate] for t in attribute.statistical_tests if t.p_value is not None and t.p_value < 0.05]
        unavailable = [rate_names[t.rate] for t in attribute.statistical_tests if t.p_value is None]
        blocks.append({"kind": "p", "text": "Unadjusted rate tests below the nominal 0.05 threshold: " + (", ".join(significant) if significant else "none") + ". Unavailable tests: " + (", ".join(unavailable) if unavailable else "none") + ". Tests above the threshold do not establish equality; omnibus tests do not identify differing pairs. Full results are in the appendix."})
        source = next(s for s in old_sections if s["title"] == f"Attribute detail: {attribute.attribute}")
        suppression = [b for b in source["blocks"] if b["kind"] == "bullet"]
        blocks.extend([{"kind": "heading", "text": "Categories excluded from comparisons"}, *suppression] if suppression else [{"kind": "p", "text": "No categories were excluded for this attribute."}])
        tests = [["Rate", "Test", "Statistic", "p-value", "Min. expected cell"]]
        for t in attribute.statistical_tests:
            tests.append([t.rate, t.method.replace("_", " "), "Unavailable" if t.statistic is None else f"{t.statistic:.4g}", pvalue(t.p_value), "Unavailable" if t.min_expected_count is None else f"{t.min_expected_count:.2f}"])
            if t.reason:
                blocks.append({"kind": "p", "text": f"{t.rate}: {t.reason}"})
        appendix["blocks"].extend([{"kind": "heading", "text": f"{display(attribute.attribute)}: component rate tests"}, {"kind": "table", "rows": tests}])
        sections.append({"title": f"Model Performance by {display(attribute.attribute)}", "blocks": blocks})
    limitations = [_neutral(s) for s in context["revised_limitations"]]
    sections.append({"title": "Data Quality and Limits of Interpretation", "blocks": [
        {"kind": "heading", "text": "Recorded data-quality warnings"},
        *[{"kind": "bullet", "text": s} for s in findings.warnings],
        *([] if findings.warnings else [{"kind": "p", "text": "No configured warnings were raised. This does not establish that the data are error-free."}]),
        {"kind": "heading", "text": "Interpretation limits"},
        *[{"kind": "bullet", "text": s} for s in limitations],
    ]})
    provenance = old_sections[-1]
    provenance["title"] = "Report Provenance"
    provenance["blocks"] = [block for block in provenance["blocks"]
                            if block.get("text") != context["disclaimer"]]
    provenance_table = provenance["blocks"][0]
    provenance_rows = [row for row in provenance_table["rows"] if row[0] != "Review status"]
    for row in provenance_rows:
        if row[0] == "Renderer":
            row[1] = "fpdf2 (96 × 176 mm pages), with a Jinja2 HTML companion"
    provenance_rows[3:3] = [
        ["Supplied model name", metadata.get("model_name", "Not supplied")],
        ["Supplied model run", metadata.get("model_run_id", "Not supplied")],
        ["Supplied task name", metadata.get("task_name", "Not supplied")],
    ]
    provenance_table["rows"] = provenance_rows
    context.update(title=f"{task} - Fairness Audit" if task else "Binary Classification Model - Fairness Audit",
                   subtitle="", review_banner="", disclaimer="",
                   sections=[*sections, appendix, provenance], revised_limitations=limitations,
                   appendix_headers=True, presentation_version="management-v2",
                   supplied_metadata=metadata, human_review="pending")
