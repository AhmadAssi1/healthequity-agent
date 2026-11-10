"""Common interface for language-model providers."""

from typing import Protocol

from pydantic import JsonValue

from app.llm_schemas import LLMMessage, LLMResponse
from app.tool_contracts import ToolDefinition


class StructuredOutputError(ValueError):
    def __init__(
        self,
        message: str,
        response: LLMResponse,
    ):
        super().__init__(message)
        self.response = response.model_copy(deep=True)
        
class LLMProvider(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def generate(
        self,
        *,
        system_instruction: str,
        messages: list[LLMMessage],
        tools: list[ToolDefinition],
        response_schema: dict[str, JsonValue] | None = None,
    ) -> LLMResponse:
        ...