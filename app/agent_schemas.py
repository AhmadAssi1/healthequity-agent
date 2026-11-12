"""Structured execution records for the audit agent."""

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from app.agent_config import AgentConfig
from app.audit_schemas import AuditBaseModel, AuditFindings
from app.llm_config import LLMConfig
from app.llm_schemas import LLMMessage, LLMResponse, ToolCall, ToolResult
from app.tool_contracts import ClaimsValidationOutput


class ToolExecutionRecord(AuditBaseModel):
    call: ToolCall
    result: ToolResult
    latency_ms: float = Field(ge=0)
    fatal_error: bool = False

    @model_validator(mode="after")
    def validate_result_matches_call(self):
        if self.call.call_id != self.result.call_id:
            raise ValueError(
                "The tool result must match the original call ID."
            )

        if self.call.name != self.result.name:
            raise ValueError(
                "The tool result must match the requested tool name."
            )

        return self

class AgentTurnRecord(AuditBaseModel):
    turn_number: int = Field(ge=1)
    response: LLMResponse | None
    tool_executions: list[ToolExecutionRecord] = Field(
        default_factory=list,
    )
    error: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"\S",
    )
    latency_ms: float = Field(ge=0)
    input_messages: list[LLMMessage] = Field(default_factory=list)
    structured_response_requested: bool = False

    @model_validator(mode="after")
    def validate_turn(self):
        if self.response is None:
            if self.error is None:
                raise ValueError(
                    "A turn without a response requires an error."
                )

            if self.tool_executions:
                raise ValueError(
                    "Tools cannot execute without a model response."
                )

            return self

        requested = self.response.tool_calls
        executed = self.tool_executions

        if len(executed) > len(requested):
            raise ValueError(
                "Executed tools must not exceed requested tools."
            )

        for record, call in zip(
            executed,
            requested[:len(executed)],
            strict=True,
        ):
            if record.call != call:
                raise ValueError(
                    "Executed tools must match the requested calls in order."
                )

        return self

SeverityLabel = Literal[
    "no_evidence_found",
    "watch",
    "substantial",
    "unavailable",
]


class SeverityCorrection(AuditBaseModel):
    attribute: str = Field(min_length=1, pattern=r"\S")
    metric: Literal[
        "demographic_parity_difference",
        "equalized_odds_difference",
        "ppv_difference",
    ]
    proposed_severity: SeverityLabel
    corrected_severity: SeverityLabel
    reason: str = Field(min_length=1, pattern=r"\S")

    @model_validator(mode="after")
    def validate_actual_correction(self):
        if self.proposed_severity == self.corrected_severity:
            raise ValueError(
                "A correction must change the proposed severity."
            )
        return self


class SummaryGroundingRecord(AuditBaseModel):
    proposed_summary: str = Field(min_length=1, pattern=r"\S")
    grounded_summary: str = Field(min_length=1, pattern=r"\S")
    source: Literal["trusted_tool_evidence"] = "trusted_tool_evidence"
    validation: ClaimsValidationOutput


class AgentRunTrace(AuditBaseModel):
    run_id: str = Field(min_length=1, pattern=r"\S")
    agent_config: AgentConfig
    llm_config: LLMConfig

    started_at: AwareDatetime
    finished_at: AwareDatetime | None
    status: Literal["running", "completed", "failed"]

    system_instruction: str = Field(
        min_length=1,
        pattern=r"\S",
    )
    initial_messages: list[LLMMessage] = Field(min_length=1)
    turns: list[AgentTurnRecord] = Field(default_factory=list)
    severity_corrections: list[SeverityCorrection] = Field(default_factory=list)
    claims_validation: ClaimsValidationOutput | None = None
    claims_validation_attempts: list[ClaimsValidationOutput] = Field(default_factory=list, max_length=2)
    summary_grounding: SummaryGroundingRecord | None = None
    error: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"\S",
    )

    @model_validator(mode="after")
    def validate_run(self):
        if self.status == "completed" and self.summary_grounding is not None and not self.summary_grounding.validation.passed:
            raise ValueError("A completed audit requires its grounded summary to pass validation.")
        if self.claims_validation_attempts:
            if self.claims_validation != self.claims_validation_attempts[-1]:
                raise ValueError("The final claims verdict must match the last recorded attempt.")
            if any(verdict.passed for verdict in self.claims_validation_attempts[:-1]):
                raise ValueError("Claims validation must stop after a passing verdict.")
        if self.status == "completed" and self.claims_validation is not None and not self.claims_validation.passed:
            raise ValueError("A completed audit cannot have a failed claims verdict.")
        if self.status == "running":
            if self.finished_at is not None or self.error is not None:
                raise ValueError(
                    "A running audit must not have a finish time or run error."
                )
        else:
            if self.finished_at is None:
                raise ValueError(
                    "A finished audit requires a finish time."
                )

            if self.status == "completed" and self.error is not None:
                raise ValueError(
                    "A completed audit must not have a run error."
                )

            if self.status == "failed" and self.error is None:
                raise ValueError(
                    "A failed audit requires an error."
                )

        if (
            self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError(
                "Finish time must not precede start time."
            )

        expected_numbers = list(range(1, len(self.turns) + 1))

        if [turn.turn_number for turn in self.turns] != expected_numbers:
            raise ValueError(
                "Turn numbers must be consecutive and start at one."
            )

        if len(self.turns) > self.agent_config.max_model_turns:
            raise ValueError(
                "The model-turn limit was exceeded."
            )

        tool_count = sum(
            len(turn.tool_executions) for turn in self.turns
        )

        if tool_count > self.agent_config.max_tool_calls:
            raise ValueError(
                "The tool-call limit was exceeded."
            )

        return self

class AgentRunResult(AuditBaseModel):
    findings: AuditFindings
    trace: AgentRunTrace
    claims_validation: ClaimsValidationOutput | None = None

    @model_validator(mode="after")
    def validate_completed_run(self):
        if self.trace.summary_grounding is not None and self.trace.summary_grounding.validation != self.claims_validation:
            raise ValueError("The final verdict must match grounded-summary validation.")
        if self.claims_validation is not None and not self.claims_validation.passed:
            raise ValueError("A successful run cannot have a failed claims verdict.")
        if self.trace.claims_validation is not None and self.claims_validation != self.trace.claims_validation:
            raise ValueError("Result and trace must contain the same claims verdict.")
        if self.trace.status != "completed":
            raise ValueError(
                "A successful run result requires a completed trace."
            )

        return self
