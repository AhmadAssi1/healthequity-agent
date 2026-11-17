"""Scripted protocol smoke provider, explicitly not a model evaluation."""

from app.audit_schemas import (
    AttributeFindings,
    AuditFindings,
    BootstrapConfidenceInterval,
    DatasetSummary,
    FairnessFinding,
    StatisticalTestResult,
    SuppressedGroup,
)
from app.llm_schemas import LLMResponse, TokenUsage, ToolCall
from fairness.policy import classify_severity

STAGES = [
    "profile_dataset", "check_group_sizes", "select_analyses",
    "compute_fairness_metrics", "run_significance_test", "bootstrap_ci",
]


def draft_from_tools(session):
    """Copy trusted outputs to exercise the pipeline, never the fixture oracle."""
    evidence = session.get_evidence()
    intervals = {(b.attribute, b.metric): b for b in evidence.bootstrap.results}
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
                group_value=s.group_value,
                suppression_reason="Insufficient evidence: " + ", ".join(s.reason_codes),
            ) for s in item.suppressed_groups],
            statistical_tests=[StatisticalTestResult(**t.model_dump(exclude={"attribute"}))
                               for t in evidence.significance.results if t.attribute == item.attribute],
        ))
    return AuditFindings(
        dataset=DatasetSummary(
            row_count=evidence.profile.row_count, column_count=evidence.profile.column_count,
            group_columns=[a.attribute for a in attributes],
            score_available=evidence.profile.score_available,
        ), attributes=attributes,
        summary="Observed associations require review.",
        warnings=evidence.profile.data_quality_flags,
        limitations=["Rows are treated as independent observations."],
    ).model_dump(mode="json")


class OfflineProvider:
    model_name = "offline-scripted"

    def __init__(self, session):
        self.session = session
        self.step = 0

    def generate(self, **kwargs):
        structured = None
        calls = []
        text = None
        if kwargs["response_schema"] is not None:
            structured = draft_from_tools(self.session)
            allowed_summary = kwargs["response_schema"]["properties"]["summary"].get("enum")
            if allowed_summary:
                structured["summary"] = allowed_summary[0]
        elif self.step < len(STAGES):
            calls = [ToolCall(call_id=f"offline_{self.step}", name=STAGES[self.step], arguments={})]
            self.step += 1
        else:
            text = "READY_FOR_FINDINGS"
        return LLMResponse(
            tool_calls=calls, text=text, structured_output=structured,
            usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            finish_reason="completed", latency_ms=0.0,
        )
