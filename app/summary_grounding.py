"""Render the quantitative summary from trusted facts, preserving the LLM draft."""

import json

from app.agent_schemas import SummaryGroundingRecord
from app.audit_schemas import AuditFindings
from app.claims_validation import validate_claims
from app.tool_contracts import AuditEvidence
from fairness.policy import classify_severity

_METRIC_NAMES = {
    "demographic_parity_difference": "Demographic parity difference",
    "equalized_odds_difference": "Equalized odds difference",
    "ppv_difference": "PPV difference",
}


def ground_summary(findings: AuditFindings, evidence: AuditEvidence) -> tuple[AuditFindings, SummaryGroundingRecord]:
    """Replace only the summary with an auditable rendering of aggregate evidence.

    The caller must first validate the full draft; this stage does not erase a
    failed draft to manufacture a passing verdict. No numerical fields change.
    P-values are reported without inventing a significance level or combined EOD
    test. Free-text limitations remain subject to the existing limited screening.
    """
    findings = AuditFindings.model_validate(findings.model_dump())
    evidence = AuditEvidence.model_validate(evidence.model_dump())
    if not validate_claims(findings, evidence).passed:
        raise ValueError("The draft must pass claim validation before summary grounding.")
    intervals = {(item.attribute, item.metric): item for item in evidence.bootstrap.results}
    paragraphs = [
        f"This audit covers {evidence.profile.row_count} rows and {len(evidence.metrics.results)} mapped attributes."
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
    proposed = findings.summary
    findings.summary = "\n\n".join(paragraphs)
    findings = AuditFindings.model_validate(findings.model_dump())
    verdict = validate_claims(findings, evidence)
    return findings, SummaryGroundingRecord(
        proposed_summary=proposed, grounded_summary=findings.summary, validation=verdict,
    )
