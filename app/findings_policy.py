"""Apply severity using trusted numerical evidence, preserving correction records."""

from copy import deepcopy
from math import isclose

from app.agent_schemas import SeverityCorrection
from app.audit_schemas import AuditFindings
from app.tool_contracts import AuditEvidence
from fairness.policy import classify_severity

LOCAL_P_VALUE_INSTRUCTION = "Return null for every p_value field. Exact p-values are inserted locally."


def restore_local_p_values(payload, evidence):
    """Fill null p-value placeholders from their exact attribute and rate evidence."""
    if not isinstance(payload, dict) or not isinstance(payload.get("attributes"), list):
        return payload
    payload = deepcopy(payload)
    tests = {(item.attribute, item.rate, item.method): item.p_value for item in evidence.significance.results}
    for attribute in payload["attributes"]:
        if not isinstance(attribute, dict) or not isinstance(attribute.get("statistical_tests"), list):
            continue
        for test in attribute["statistical_tests"]:
            if not isinstance(test, dict) or test.get("p_value", False) is not None:
                continue
            key = (attribute.get("attribute"), test.get("rate"), test.get("method"))
            if all(isinstance(value, str) for value in key) and key in tests:
                test["p_value"] = tests[key]
    return payload


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
