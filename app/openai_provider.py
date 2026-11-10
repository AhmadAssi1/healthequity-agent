"""OpenAI implementation of the language-model provider."""

import json
from copy import deepcopy
from time import perf_counter

from openai import OpenAI
from openai.types.responses import Response
from pydantic import JsonValue, ValidationError

from app.llm_config import LLMConfig
from app.llm_provider import StructuredOutputError
from app.llm_schemas import (
    LLMMessage,
    LLMResponse,
    TokenUsage,
    ToolCall,
)
from app.tool_contracts import ToolDefinition


class OpenAIProvider:
    def __init__(
        self,
        config: LLMConfig,
        client: OpenAI | None = None,
    ):
        self._config = config.model_copy(deep=True)

        self._client = (
            client
            if client is not None
            else OpenAI(
                timeout=self._config.timeout_seconds,
                max_retries=0,
            )
        )

    @property
    def model_name(self) -> str:
        return self._config.model_name
    
    @staticmethod
    def _tool_declarations(
        tools: list[ToolDefinition],
    ) -> list[dict]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(tool.input_schema),
                "strict": False,
            }
            for tool in tools
            if tool.available
        ]
    @staticmethod
    def _messages_to_input(
        messages: list[LLMMessage],
    ) -> list[dict]:
        items = []

        for message in messages:
            if message.role == "tool":
                for result in message.tool_results:
                    items.append({
                        "type": "function_call_output",
                        "call_id": result.call_id,
                        "output": json.dumps(
                            {
                                "is_error": result.is_error,
                                "output": result.output,
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                    })

                continue

            if message.text is not None:
                items.append({
                    "role": message.role,
                    "content": message.text,
                })

            for call in message.tool_calls:
                items.append({
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                })

        return items

    @staticmethod
    def _parse_response(
        response: Response,
        latency_ms: float,
        expect_structured: bool = False,
    ) -> LLMResponse:
        if response.error is not None:
            raise ValueError(
                "OpenAI returned an error response."
            )

        if response.status != "completed":
            raise ValueError(
                f"OpenAI response was not completed: {response.status}"
            )

        tool_calls = []

        for item in response.output:
            if item.type == "function_call":
                if item.status not in (None, "completed"):
                    raise ValueError(
                        "OpenAI returned an incomplete tool call."
                    )

                arguments = json.loads(item.arguments)

                if not isinstance(arguments, dict):
                    raise ValueError(
                        "Tool arguments must be a JSON object."
                    )

                tool_calls.append(
                    ToolCall(
                        call_id=item.call_id,
                        name=item.name,
                        arguments=arguments,
                    )
                )

            elif item.type == "message":
                if any(
                    part.type == "refusal"
                    for part in item.content
                ):
                    raise ValueError(
                        "OpenAI refused the request."
                    )

            else:
                raise ValueError(
                    f"Unsupported response item: {item.type}"
                )

        text = response.output_text or None
        usage = response.usage

        parsed = LLMResponse(
            tool_calls=tool_calls,
            text=text,
            structured_output=None,
            usage=TokenUsage(
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
            ),
            finish_reason=response.status,
            latency_ms=latency_ms,
        )

        if expect_structured:
            if tool_calls:
                raise StructuredOutputError(
                    "A final structured response must not contain tool calls.",
                    parsed,
                )

            if text is None or not text.strip():
                raise StructuredOutputError(
                    "The structured response is empty.",
                    parsed,
                )

            try:
                structured_output = json.loads(text)
            except json.JSONDecodeError as error:
                raise StructuredOutputError(
                    "The structured response is not valid JSON.",
                    parsed,
                ) from error

            if not isinstance(structured_output, dict):
                raise StructuredOutputError(
                    "The structured response must be a JSON object.",
                    parsed,
                )

            try:
                return LLMResponse.model_validate({
                    **parsed.model_dump(),
                    "structured_output": structured_output,
                })
            except ValidationError as error:
                raise StructuredOutputError(
                    "The structured response contains invalid JSON values.",
                    parsed,
                ) from error

        if not tool_calls and (
            text is None or not text.strip()
        ):
            raise ValueError(
                "OpenAI returned no text or tool calls."
            )

        return parsed

    
    def generate(
        self,
        *,
        system_instruction: str,
        messages: list[LLMMessage],
        tools: list[ToolDefinition],
        response_schema: dict[str, JsonValue] | None = None,
    ) -> LLMResponse:
        if not system_instruction.strip():
            raise ValueError(
                "The system instruction must not be blank."
            )

        if not messages:
            raise ValueError(
                "At least one message is required."
            )

        request = {
            "model": self.model_name,
            "instructions": system_instruction,
            "input": self._messages_to_input(messages),
            "store": False,
        }

        if response_schema is not None:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "audit_findings",
                    "schema": deepcopy(response_schema),
                    "strict": True,
                }
            }
        else:
            declarations = self._tool_declarations(tools)

            if declarations:
                request["tools"] = declarations
                request["parallel_tool_calls"] = False

        started = perf_counter()

        response = self._client.responses.create(**request)

        latency_ms = (perf_counter() - started) * 1000

        return self._parse_response(
            response=response,
            latency_ms=latency_ms,
            expect_structured=response_schema is not None,
        )
        
