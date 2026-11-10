"""Provider-independent models for LLM interactions."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from app.audit_schemas import AuditBaseModel


class ToolCall(AuditBaseModel):
    call_id: str = Field(min_length=1, pattern=r"\S")
    name: str = Field(min_length=1, pattern=r"\S")
    arguments: dict[str, JsonValue]

class TokenUsage(AuditBaseModel):
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    total_tokens: int | None = Field(ge=0)

class LLMResponse(AuditBaseModel):
    tool_calls: list[ToolCall]
    text: str | None
    structured_output: dict[str, JsonValue] | None
    usage: TokenUsage
    finish_reason: str | None = Field(min_length=1)
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_unique_call_ids(self):
        call_ids = [call.call_id for call in self.tool_calls]

        if len(call_ids) != len(set(call_ids)):
            raise ValueError(
                "Tool call IDs must be unique within a response."
            )

        return self

class ToolResult(AuditBaseModel):
    call_id: str = Field(min_length=1, pattern=r"\S")
    name: str = Field(min_length=1, pattern=r"\S")
    output: dict[str, JsonValue]
    is_error: bool

class LLMMessage(AuditBaseModel):
    role: Literal["user", "assistant", "tool"]
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_message(self):
        if self.tool_calls and self.role != "assistant":
            raise ValueError(
                "Only assistant messages may contain tool calls."
            )

        if self.tool_results and self.role != "tool":
            raise ValueError(
                "Only tool messages may contain tool results."
            )

        if self.role == "tool":
            if self.text is not None or not self.tool_results:
                raise ValueError(
                    "Tool messages require results and must not contain text."
                )
        elif not (
            (self.text is not None and self.text.strip())
            or self.tool_calls
        ):
            raise ValueError(
                "User and assistant messages must not be empty."
            )

        for items in (self.tool_calls, self.tool_results):
            call_ids = [item.call_id for item in items]

            if len(call_ids) != len(set(call_ids)):
                raise ValueError(
                    "Tool call IDs must be unique within a message."
                )

        return self