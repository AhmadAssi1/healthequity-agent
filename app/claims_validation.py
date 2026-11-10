"""Deterministic evidence checks and conservative English narrative screening.

A passing verdict means these checks passed, not unrestricted semantic proof.
No model calls, raw patient rows, or inferred suppressed counts are used.
"""

import re
from decimal import Decimal
from math import isclose

from app.audit_schemas import AuditFindings
from app.tool_contracts import AuditEvidence, ClaimIssue, ClaimsValidationOutput
from fairness.policy import classify_severity


def numbers_match(claimed: float | None, trusted: float | None) -> bool:
    if claimed is None or trusted is None:
        return claimed is None and trusted is None
    return isclose(claimed, trusted, rel_tol=1e-9, abs_tol=1e-12)


def _identity(value):
    # Python otherwise treats True, 1, and 1.0 as equal group labels.
    return type(value).__name__, value


def validate_claims(
    findings: AuditFindings,
    evidence: AuditEvidence,
) -> ClaimsValidationOutput:
    """Check complete structured claims and screen common unsupported wording.

    Narrative screening is deliberately limited to recognizable English patterns
    and numeric support; it cannot prove arbitrary prose or attribution correct.
    """
    findings = AuditFindings.model_validate(findings.model_dump())
    evidence = AuditEvidence.model_validate(evidence.model_dump())
    issues = []

    def issue(code, path, message):
        issues.append(ClaimIssue(code=code, message=f"{path}: {message}"))

    def compare(claimed, trusted, path):
        if isinstance(trusted, float) or isinstance(claimed, float):
            equal = numbers_match(claimed, trusted)
        else:
            equal = type(claimed) is type(trusted) and claimed == trusted
        if not equal:
            issue("evidence_mismatch", path, "Does not match trusted evidence.")

    profile = evidence.profile
    for field in ("row_count", "column_count", "score_available"):
        compare(getattr(findings.dataset, field), getattr(profile, field), f"dataset.{field}")
    expected_attributes = {item.attribute: item for item in evidence.metrics.results}
    if set(findings.dataset.group_columns) != set(expected_attributes):
        issue("unsupported_group", "dataset.group_columns", "Attributes differ from trusted evidence.")

    intervals = {(item.attribute, item.metric): item for item in evidence.bootstrap.results}
    tests = {(item.attribute, item.rate): item for item in evidence.significance.results}
    for attribute in findings.attributes:
        path = f"attributes.{attribute.attribute}"
        trusted = expected_attributes.get(attribute.attribute)
        if trusted is None:
            continue
        for field in ("groups", "suppressed_groups"):
            proposed_groups = {_identity(g.group_value): g for g in getattr(attribute, field)}
            trusted_groups = {_identity(g.group_value): g for g in getattr(trusted, field)}
            if proposed_groups.keys() != trusted_groups.keys():
                issue("unsupported_group", f"{path}.{field}", "Missing, extra, or incorrectly typed group labels.")
            for key in proposed_groups.keys() & trusted_groups.keys():
                proposed, original = proposed_groups[key], trusted_groups[key]
                group_path = f"{path}.{field}[{key!r}]"
                if field == "suppressed_groups":
                    if not re.search(r"insufficient|not enough|below.*minim", proposed.suppression_reason, re.IGNORECASE):
                        issue("missing_insufficient_evidence", group_path, "State insufficient evidence explicitly.")
                    continue
                compare(proposed.count, original.count, f"{group_path}.count")
                for rate in ("base_rate", "selection_rate", "tpr", "fpr", "ppv", "auroc"):
                    compare(getattr(proposed, rate).value, getattr(original, rate).value, f"{group_path}.{rate}")
                    compare(getattr(proposed, rate).reason, getattr(original, rate).reason, f"{group_path}.{rate}.reason")

        gaps = {item.metric: item for item in trusted.gaps}
        for gap in attribute.fairness_findings:
            gap_path = f"{path}.{gap.metric}"
            original = gaps[gap.metric]
            ci = intervals.get((attribute.attribute, gap.metric))
            compare(gap.value, original.value, f"{gap_path}.value")
            compare(gap.reason, original.reason, f"{gap_path}.reason")
            if ci is None:
                issue("evidence_mismatch", gap_path, "No trusted interval evidence.")
                continue
            compare(gap.value, ci.observed, f"{gap_path}.observed")
            compare(gap.confidence_interval_reason, ci.reason, f"{gap_path}.confidence_interval_reason")
            if gap.confidence_interval is None:
                if ci.lower is not None:
                    issue("evidence_mismatch", gap_path, "A trusted interval was omitted.")
            else:
                for field in ("lower", "upper", "valid_resamples", "total_resamples"):
                    compare(getattr(gap.confidence_interval, field), getattr(ci, field), f"{gap_path}.{field}")
            compare(gap.severity, classify_severity(original.value, ci.lower, ci.upper, evidence.severity_policy),
                    f"{gap_path}.severity")

        proposed_tests = {item.rate: item for item in attribute.statistical_tests}
        trusted_tests = {rate: test for (name, rate), test in tests.items() if name == attribute.attribute}
        if proposed_tests.keys() != trusted_tests.keys():
            issue("evidence_mismatch", f"{path}.statistical_tests", "Every trusted rate test must appear once.")
        for rate in proposed_tests.keys() & trusted_tests.keys():
            for field in ("method", "statistic", "p_value", "min_expected_count", "reason"):
                compare(getattr(proposed_tests[rate], field), getattr(trusted_tests[rate], field),
                        f"{path}.tests.{rate}.{field}")

    # Preserve trusted quality warnings rather than letting the narrative erase them.
    for warning in profile.data_quality_flags:
        if warning not in findings.warnings:
            issue("evidence_mismatch", "warnings", "A trusted data-quality warning was omitted.")

    texts = [("summary", findings.summary)]
    texts += [(f"warnings[{i}]", text) for i, text in enumerate(findings.warnings)]
    texts += [(f"limitations[{i}]", text) for i, text in enumerate(findings.limitations)]
    # Only visible aggregate numeric evidence is eligible for narrative support.
    supported_numbers = []

    def collect(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key not in ("seed", "group_value"):
                    collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)
        elif type(node) in (int, float):
            supported_numbers.append(node)

    collect(evidence.model_dump())
    supported_numbers.append(len(expected_attributes))
    number_pattern = r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?\s*(?:%|percentage points?|percent)?"
    for path, text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
            # Screen affirmative claims; explicit negation/limitations are allowed.
            affirmative = not re.search(r"\b(?:not|no|cannot|never|without|neither)\b", sentence, re.IGNORECASE)
            if affirmative and re.search(r"\b(?:causes?|caused by|proves? causation|causal effect)\b", sentence, re.IGNORECASE):
                issue("causal_language", path, "Causal conclusions are unsupported by this audit.")
            if affirmative and re.search(r"\b(?:proves? fairness|guarantees? fairness|is fair|legally compliant|clinically safe|validation (?:has )?passed)\b", sentence, re.IGNORECASE):
                issue("evidence_mismatch", path, "An approval or fairness guarantee is unsupported.")
            mentioned = [item for name, item in expected_attributes.items()
                         if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", sentence, re.IGNORECASE)]
            suppressed = any(item.suppressed_groups for item in (mentioned or expected_attributes.values()))
            if suppressed and re.search(r"\bno (?:\w+\s+){0,3}groups? (?:were |was |are )?suppressed\b|\ball groups (?:were |are )?(?:eligible|included)\b", sentence, re.IGNORECASE):
                issue("missing_insufficient_evidence", path, "The narrative contradicts group suppression.")
        for match in re.finditer(number_pattern, text, re.IGNORECASE):
            token = match.group().strip()
            numeric = re.match(r"[-+\d,.eE]+", token).group()
            value = float(numeric.replace(",", ""))
            exponent = Decimal(numeric.replace(",", "")).as_tuple().exponent
            rounded = "." in numeric or "e" in numeric.lower()
            tolerance = 0.5 * 10.0 ** exponent if rounded and -300 <= exponent <= 300 else 0.0
            tolerance += 1e-12
            percent = bool(re.search(r"%|percent", token, re.IGNORECASE))
            candidates = [n * 100 for n in supported_numbers if 0 <= n <= 1] if percent else supported_numbers
            if not any(abs(value - n) <= tolerance for n in candidates):
                issue("unsupported_number", path, "A narrative number has no matching aggregate evidence.")

    return ClaimsValidationOutput(passed=not issues, issues=issues)
