"""Render audit narrative from trusted evidence after draft validation."""

import json

from app.agent_schemas import SummaryGroundingRecord
from app.audit_schemas import AuditFindings
from app.claims_validation import validate_claims
from app.tool_contracts import AuditEvidence
from fairness.policy import classify_severity

# A short transport value avoids embedding a long, changing report in an enum.
# It makes no quantitative claim. Full findings must validate before the local
# renderer replaces it with the evidence-bound report summary.
SUMMARY_PLACEHOLDER = "Summary is rendered locally from validated aggregate evidence."
LIMITATIONS_PLACEHOLDER = "Limitations are rendered locally from the audit methods and evidence."

_METRIC_NAMES = {
    "demographic_parity_difference": "Demographic parity difference",
    "equalized_odds_difference": "Equalized odds difference",
    "ppv_difference": "PPV difference",
}


def render_summary(evidence: AuditEvidence) -> str:
    """Render a quantitative summary directly from complete trusted evidence."""
    evidence = AuditEvidence.model_validate(evidence.model_dump())
    intervals = {(item.attribute, item.metric): item for item in evidence.bootstrap.results}
    attribute_count = len(evidence.metrics.results)
    attribute_label = "attribute" if attribute_count == 1 else "attributes"
    paragraphs = [
        f"This audit covers {evidence.profile.row_count} rows and {attribute_count} mapped {attribute_label}."
    ]
    for attribute in evidence.metrics.results:
        sentences = [
            (
                f"For attribute {json.dumps(attribute.attribute, ensure_ascii=False)}, "
                f"{len(attribute.groups)} reportable groups were analyzed; "
                f"{len(attribute.suppressed_groups)} suppressed groups have insufficient evidence."
            )
        ]
        for gap in attribute.gaps:
            interval = intervals[attribute.attribute, gap.metric]
            label = classify_severity(gap.value, interval.lower, interval.upper, evidence.severity_policy)
            value = "unavailable" if gap.value is None else repr(gap.value)
            sentence = f"{_METRIC_NAMES[gap.metric]} is {value}; policy severity is {label}."
            if interval.lower is None:
                sentence += " Its confidence interval is unavailable."
            else:
                sentence += f" Its 95% confidence interval is [{interval.lower!r}, {interval.upper!r}]."
            sentences.append(sentence)
        for test in evidence.significance.results:
            if test.attribute != attribute.attribute:
                continue
            if test.method == "unavailable":
                sentences.append(f"The {test.rate} comparison test is unavailable.")
            else:
                sentences.append(f"The {test.rate} comparison uses {test.method} with p={test.p_value!r}.")
        paragraphs.append(" ".join(sentences))
    paragraphs.append(
        "Severity and statistical significance answer different questions. "
        "These results describe associations and do not establish causation, clinical suitability, or legal compliance. "
        "No combined equalized-odds p-value is reported. Human review remains required."
    )
    return "\n\n".join(paragraphs)


def render_limitations(evidence: AuditEvidence) -> list[str]:
    """Describe the audit's methods and data limitations from recorded evidence."""
    limitations = [
        (
            "The results describe the supplied observations and mapped attributes. "
            "Applying them to other populations or time periods requires further evidence."
        ),
        (
            "The analysis assumes that observed outcomes, binary predictions, group labels, "
            "and any supplied scores are correctly aligned."
        ),
        (
            "Interpret policy severity, component rate tests, and bootstrap intervals separately. "
            "Assessing equivalence requires a dedicated analysis."
        ),
        (
            "The bootstrap assumes independent observations within each eligible group. "
            "Dependence between observations can affect the uncertainty estimates."
        ),
    ]
    if any(
        interval.observed == 0 and interval.lower is not None and interval.lower > 0
        for interval in evidence.bootstrap.results
    ):
        limitations.append(
            "Some reported percentile intervals have positive lower bounds even when the observed gap is zero. "
            "Use the component rate tests to assess differences in rates."
        )
    if any(test.method in {"chi_square", "fisher_monte_carlo"} for test in evidence.significance.results):
        limitations.append(
            "The omnibus tests compare rates jointly across the eligible groups. "
            "Identifying a statistically significant pair would require a post-hoc comparison."
        )
    if any(item.suppressed_groups for item in evidence.metrics.results):
        limitations.append("Groups suppressed under the reporting policy have insufficient evidence for comparison.")
    if not evidence.profile.score_available:
        limitations.append("Score-based metrics such as AUROC require a mapped score column.")
    if any(
        flag.startswith("Duplicate rows exist in the mapped audit columns")
        for flag in evidence.profile.data_quality_flags
    ):
        limitations.append(
            "The profiler found repeated combinations in the mapped audit columns. "
            "Identifying duplicate source observations requires additional source evidence."
        )
    return limitations


def ground_summary(
    findings: AuditFindings,
    evidence: AuditEvidence,
    *,
    include_limitations: bool = False,
) -> tuple[AuditFindings, SummaryGroundingRecord]:
    """Ground a fully validated draft without rescuing rejected model claims.

    The draft must pass full validation before the summary and any requested
    limitations can be replaced with text grounded in the recorded evidence.
    """
    findings = AuditFindings.model_validate(findings.model_dump())
    evidence = AuditEvidence.model_validate(evidence.model_dump())
    if not validate_claims(findings, evidence).passed:
        raise ValueError("The draft must pass claim validation before summary grounding.")
    proposed = findings.summary
    findings.summary = render_summary(evidence)
    if include_limitations:
        findings.limitations = render_limitations(evidence)
    findings = AuditFindings.model_validate(findings.model_dump())
    verdict = validate_claims(findings, evidence)
    return findings, SummaryGroundingRecord(
        proposed_summary=proposed,
        grounded_summary=findings.summary,
        grounded_limitations=findings.limitations if include_limitations else None,
        validation=verdict,
    )
