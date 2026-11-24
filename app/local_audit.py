"""Complete a statistical audit with the same trusted tools used by the agent."""

from app.audit_schemas import (
    AttributeFindings,
    AuditFindings,
    BootstrapConfidenceInterval,
    DatasetSummary,
    FairnessFinding,
    StatisticalTestResult,
    SuppressedGroup,
)
from app.claims_validation import validate_claims
from app.summary_grounding import render_limitations, render_summary
from app.tools import ToolExecutionError
from fairness.policy import classify_severity

LOCAL_AUDIT_LABEL = "Local statistical audit (no AI provider)"


def findings_from_evidence(evidence, *, summary, limitations):
    """Assemble findings directly from validated tool outputs."""
    intervals = {(item.attribute, item.metric): item for item in evidence.bootstrap.results}
    attributes = []
    for item in evidence.metrics.results:
        gaps = []
        for gap in item.gaps:
            boot = intervals[item.attribute, gap.metric]
            interval = None if boot.lower is None else BootstrapConfidenceInterval(
                lower=boot.lower, upper=boot.upper,
                valid_resamples=boot.valid_resamples, total_resamples=boot.total_resamples,
            )
            gaps.append(FairnessFinding(
                **gap.model_dump(), confidence_interval=interval,
                confidence_interval_reason=boot.reason,
                severity=classify_severity(gap.value, boot.lower, boot.upper, evidence.severity_policy),
            ))
        attributes.append(AttributeFindings(
            attribute=item.attribute, groups=item.groups, fairness_findings=gaps,
            suppressed_groups=[SuppressedGroup(
                group_value=group.group_value,
                suppression_reason="Insufficient evidence: " + ", ".join(group.reason_codes),
            ) for group in item.suppressed_groups],
            statistical_tests=[StatisticalTestResult(**test.model_dump(exclude={"attribute"}))
                               for test in evidence.significance.results if test.attribute == item.attribute],
        ))
    return AuditFindings(
        dataset=DatasetSummary(
            row_count=evidence.profile.row_count, column_count=evidence.profile.column_count,
            group_columns=[item.attribute for item in attributes],
            score_available=evidence.profile.score_available,
        ), attributes=attributes, summary=summary,
        warnings=evidence.profile.data_quality_flags, limitations=limitations,
    )


def run_local_audit(session):
    """Recompute the standard analysis and independently validate the findings."""
    session.invoke("compute_fairness_metrics", {})
    session.invoke("run_significance_test", {})
    session.invoke("bootstrap_ci", {})
    evidence = session.get_evidence()
    findings = findings_from_evidence(
        evidence, summary=render_summary(evidence), limitations=render_limitations(evidence),
    )
    if not validate_claims(findings, evidence).passed:
        raise ToolExecutionError("Local findings failed evidence validation.")
    return findings
