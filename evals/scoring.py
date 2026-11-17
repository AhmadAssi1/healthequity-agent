"""Score all scheduled cases, retaining failures in every applicable denominator."""

import math

from app.audit_schemas import AuditFindings
from evals.scenarios import RATES, Scenario

STAGES = {
    "profile_dataset", "check_group_sizes", "select_analyses",
    "compute_fairness_metrics", "run_significance_test", "bootstrap_ci",
}


def score_findings(case: Scenario, payload: dict, trace: dict) -> dict:
    """Check independent fixture values and actual executed analysis coverage.

    Statistical quantities beyond rates/gaps are checked by the existing claims
    validator against deterministic tool evidence, not claimed as independently
    verified by the fixture oracle. Missing evidence never receives credit.
    """
    issues, numeric = [], []

    def number(path, actual, expected):
        passed = (
            actual is None if expected is None else
            isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)
        )
        numeric.append(bool(passed))
        if not passed:
            issues.append(path)

    findings = AuditFindings.model_validate(payload)
    expected = case.expected
    number("dataset.row_count", findings.dataset.row_count, expected["row_count"])
    number("dataset.column_count", findings.dataset.column_count, len(case.data.columns))
    attributes = {item.attribute: item for item in findings.attributes}
    shape_ok = (
        set(attributes) == set(expected["attributes"])
        and set(findings.dataset.group_columns) == set(case.mapping.groups)
        and findings.dataset.score_available == expected["score_available"]
    )
    suppression_ok = True
    methods_ok = True
    for name, answers in expected["attributes"].items():
        item = attributes.get(name)
        if item is None:
            shape_ok = suppression_ok = methods_ok = False
            number(f"{name}.missing", None, 1)
            continue
        groups = {group.group_value: group for group in item.groups}
        suppressed = {group.group_value: group for group in item.suppressed_groups}
        suppression_ok &= (
            set(groups) == set(answers["eligible_groups"])
            and set(suppressed) == set(answers["suppressed_groups"])
            and all("insufficient evidence" in g.suppression_reason.lower()
                    for g in suppressed.values())
        )
        for label in answers["eligible_groups"]:
            group = groups.get(label)
            oracle = answers["groups"][label]
            if group is None:
                number(f"{name}.missing_group", None, oracle["count"])
                continue
            number(f"{name}.count", group.count, oracle["count"])
            number(f"{name}.base_rate", group.base_rate.value,
                   oracle["positive_outcomes"] / oracle["count"])
            for rate in RATES:
                number(f"{name}.{rate}", getattr(group, rate).value, oracle["rates"][rate])
            if not expected["score_available"] and group.auroc.value is not None:
                shape_ok = False
        gaps = {gap.metric: gap.value for gap in item.fairness_findings}
        for metric, value in answers["gaps"].items():
            number(f"{name}.{metric}", gaps.get(metric), value)
        methods_ok &= {test.rate: test.method for test in item.statistical_tests} == answers["rate_tests"]
    if expected["duplicate_rows"] and not any("duplicate" in w.lower() for w in findings.warnings):
        shape_ok = False

    executions = [record for turn in trace.get("turns", [])
                  for record in turn.get("tool_executions", [])]
    coverage = {stage: set() for stage in STAGES}
    successful_stages = set()
    calls_ok = True
    proposal_decisions = []
    for record in executions:
        call, result = record["call"], record["result"]
        name, arguments = call["name"], call["arguments"]
        if name not in STAGES or result["is_error"]:
            calls_ok = False
            continue
        successful_stages.add(name)
        requested = arguments.get("attributes") or case.mapping.groups
        calls_ok &= set(requested) <= set(case.mapping.groups)
        if name == "select_analyses":
            # Omitted proposal delegates selection; it is not a correct model
            # proposal. Separately assess its corrected executable plan below.
            proposal = arguments.get("proposed_plan")
            if proposal is not None:
                proposal_decisions.append(_plan_matches(proposal, case))
            calls_ok &= _plan_matches(result["output"].get("plan", {}), case)
        if name == "compute_fairness_metrics":
            coverage[name].update(r["attribute"] for r in result["output"].get("results", []))
        elif name == "run_significance_test":
            for row in result["output"].get("results", []):
                coverage[name].add((row["attribute"], row["rate"]))
                answers = expected["attributes"].get(row["attribute"], {})
                calls_ok &= row["method"] == answers.get("rate_tests", {}).get(row["rate"])
            if arguments.get("test_name"):
                attr, rate = requested[0], arguments["rates"][0]
                calls_ok &= arguments["test_name"] == expected["attributes"][attr]["rate_tests"][rate]
        elif name == "bootstrap_ci":
            calls_ok &= arguments.get("n_resamples", 1000) == 1000
            calls_ok &= arguments.get("seed", 42) == 42
            coverage[name].update((r["attribute"], r["metric"])
                                  for r in result["output"].get("results", []))
    attrs = set(expected["attributes"])
    calls_ok &= successful_stages == STAGES
    calls_ok &= coverage["compute_fairness_metrics"] == attrs
    calls_ok &= coverage["run_significance_test"] == {(a, r) for a in attrs for r in RATES}
    calls_ok &= coverage["bootstrap_ci"] == {
        (a, m) for a, answer in expected["attributes"].items() for m in answer["gaps"]
    }
    final_verdict = trace.get("claims_validation")
    grounding = trace.get("summary_grounding")
    automated_claims_pass = bool(
        final_verdict and final_verdict.get("passed")
        and not final_verdict.get("issues")
        and grounding and grounding.get("validation", {}).get("passed")
    )
    return {
        "schema_valid": True,
        "numeric_pass": bool(numeric) and all(numeric) and bool(shape_ok),
        "numeric_checks": len(numeric), "numeric_checks_passed": sum(numeric),
        "numeric_mismatches": issues,
        "suppression_correct": bool(suppression_ok),
        "tool_selection_correct": bool(calls_ok and methods_ok and shape_ok),
        "tool_request_errors": [
            {"tool": record["call"]["name"],
             "code": record["result"]["output"].get("code", "tool_error")}
            for record in executions if record["result"]["is_error"]
        ],
        "model_proposals_correct": sum(proposal_decisions),
        "model_proposals_total": len(proposal_decisions),
        "automated_claims_pass": automated_claims_pass,
        "human_review": "pending",
    }


def _plan_matches(plan: dict, case: Scenario) -> bool:
    """Compare decisions only; prose and ordering are not planning accuracy."""
    if not isinstance(plan, dict):
        return False
    expected = case.expected
    attributes = plan.get("attributes", [])
    if not isinstance(attributes, list) or any(not isinstance(a, dict) for a in attributes):
        return False
    by_name = {a.get("attribute"): a for a in attributes}
    if len(by_name) != len(attributes) or set(by_name) != set(expected["attributes"]):
        return False
    if plan.get("score_available") != expected["score_available"]:
        return False
    for name, answers in expected["attributes"].items():
        item = by_name[name]
        if (item.get("eligible_group_count") != len(answers["eligible_groups"])
                or item.get("compute_auroc") != expected["score_available"]):
            return False
        tests = item.get("statistical_tests", [])
        metrics = item.get("fairness_metrics", [])
        if not isinstance(tests, list) or not isinstance(metrics, list):
            return False
        if any(not isinstance(row, dict) for row in [*tests, *metrics]):
            return False
        if len(tests) != len(RATES) or {t.get("rate"): t.get("method") for t in tests} != answers["rate_tests"]:
            return False
        if len(metrics) != 3 or {m.get("metric") for m in metrics} != set(answers["gaps"]):
            return False
        if any(m.get("available") is not True or m.get("bootstrap") is not True for m in metrics):
            return False
    return True


def summarize(rows: list[dict], mode: str) -> dict:
    """Audit-level rates include failed valid cases; refusals use their own base."""
    valid = [r for r in rows if r["expected_status"] == "audit"]
    invalid = [r for r in rows if r["expected_status"] == "refuse"]

    def rate(items, field):
        return sum(r.get(field) is True for r in items) / len(items) if items else None

    proposed = sum(r.get("model_proposals_total", 0) for r in valid)
    proposal_correct = sum(r.get("model_proposals_correct", 0) for r in valid)
    return {
        "mode": mode, "scheduled": len(rows),
        "valid_cases": len(valid), "invalid_cases": len(invalid),
        "completed_valid_audits": sum(r["status"] == "completed" for r in valid),
        "tool_selection_accuracy": rate(valid, "tool_selection_correct"),
        "numeric_accuracy": rate(valid, "numeric_pass"),
        "numeric_accuracy_scope": "All fixture counts, base rates, four group rates, three gaps, and dataset shape per audit; 1e-6 relative/1e-9 absolute tolerance. CIs, AUROC, p-values and severity use tool-evidence validation, not an independent oracle.",
        "schema_validity_rate": rate(valid, "schema_valid"),
        "suppression_correctness": rate(valid, "suppression_correct"),
        "refusal_correctness": rate(invalid, "refusal_correct"),
        "automated_final_claims_pass_rate": rate(valid, "automated_claims_pass"),
        "unsupported_claim_rate": None,
        "unsupported_claim_rate_reason": "Unrestricted narrative requires human review; automated verdicts alone do not certify zero unsupported claims.",
        "model_plan_proposal_accuracy": proposal_correct / proposed if proposed else None,
        "model_plan_proposal_count": proposed,
        "live_benchmark_passed": False,
        "benchmark_status": "pending_human_claim_review" if mode == "live" else "offline_smoke_only",
    }
