"""Apply severity using trusted numerical evidence, preserving correction records."""

from math import isclose

from app.agent_schemas import SeverityCorrection
from app.audit_schemas import AuditFindings
from app.tool_contracts import AuditEvidence
from fairness.policy import classify_severity


def _same_number(proposed: float | None, expected: float | None) -> bool:
    if proposed is None or expected is None:
        return proposed is expected
    return isclose(proposed, expected, rel_tol=1e-9, abs_tol=1e-12)


def apply_severity_policy(
    findings: AuditFindings,
    evidence: AuditEvidence,
) -> tuple[AuditFindings, list[SeverityCorrection]]:
    """Correct labels only after their gap and CI match trusted tool outputs.

    This is a policy check, not full numerical or narrative claim validation.
    Mismatched quantitative inputs fail rather than being silently repaired.
    """
    corrected = AuditFindings.model_validate(findings.model_dump())
    attributes = {item.attribute: item for item in evidence.metrics.results}
    intervals = {(item.attribute, item.metric): item for item in evidence.bootstrap.results}
    if {item.attribute for item in corrected.attributes} != set(attributes):
        raise ValueError("Finding attributes must match trusted severity evidence.")

    corrections = []
    for attribute in corrected.attributes:
        gaps = {item.metric: item for item in attributes[attribute.attribute].gaps}
        for finding in attribute.fairness_findings:
            key = (attribute.attribute, finding.metric)
            if finding.metric not in gaps or key not in intervals:
                raise ValueError("Trusted gap and interval evidence is required for severity.")
            gap = gaps[finding.metric]
            interval = intervals[key]
            if not _same_number(finding.value, gap.value) or not _same_number(gap.value, interval.observed):
                raise ValueError("The finding gap does not match trusted severity evidence.")

            ci = finding.confidence_interval
            if ci is None:
                if interval.lower is not None or interval.upper is not None:
                    raise ValueError("A trusted confidence interval is missing from the finding.")
            elif (
                not _same_number(ci.lower, interval.lower)
                or not _same_number(ci.upper, interval.upper)
                or ci.valid_resamples != interval.valid_resamples
                or ci.total_resamples != interval.total_resamples
            ):
                raise ValueError("The finding interval does not match trusted severity evidence.")

            expected = classify_severity(
                gap.value, interval.lower, interval.upper, evidence.severity_policy,
            )
            if finding.severity != expected:
                corrections.append(SeverityCorrection(
                    attribute=attribute.attribute, metric=finding.metric,
                    proposed_severity=finding.severity, corrected_severity=expected,
                    reason="Recomputed from trusted gap, bootstrap interval, and severity policy.",
                ))
                finding.severity = expected

    return AuditFindings.model_validate(corrected.model_dump()), corrections
