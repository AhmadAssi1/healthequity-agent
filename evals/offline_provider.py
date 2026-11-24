"""Scripted protocol smoke provider, explicitly not a model evaluation."""

from app.llm_schemas import LLMResponse, TokenUsage, ToolCall
from app.local_audit import findings_from_evidence

STAGES = [
    "profile_dataset", "check_group_sizes", "select_analyses",
    "compute_fairness_metrics", "run_significance_test", "bootstrap_ci",
]


def draft_from_tools(session):
    """Copy trusted outputs to exercise the pipeline, never the fixture oracle."""
    return findings_from_evidence(
        session.get_evidence(),
        summary="Observed associations require review.",
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
            properties = kwargs["response_schema"]["properties"]
            allowed_summary = properties["summary"].get("enum")
            if allowed_summary:
                structured["summary"] = allowed_summary[0]
            allowed_limitations = properties["limitations"]["items"].get("enum")
            if allowed_limitations:
                structured["limitations"] = [allowed_limitations[0]]
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
