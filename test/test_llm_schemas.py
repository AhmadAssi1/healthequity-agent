"""Provider-neutral conversation and usage validation."""

import pytest
from pydantic import ValidationError

from app.llm_schemas import LLMMessage, LLMResponse, TokenUsage, ToolCall, ToolResult


def call():
    return ToolCall(call_id="call_1", name="profile_dataset", arguments={})


def result():
    return ToolResult(call_id="call_1", name="profile_dataset", output={"rows": 10}, is_error=False)


@pytest.mark.parametrize("payload", [
    {"role": "user", "text": "Audit the profile."},
    {"role": "assistant", "text": "Ready."},
    {"role": "assistant", "tool_calls": [call()]},
    {"role": "assistant", "text": "Checking.", "tool_calls": [call()]},
    {"role": "tool", "tool_results": [result()]},
])
def test_valid_messages_round_trip(payload):
    message = LLMMessage.model_validate(payload)
    assert LLMMessage.model_validate_json(message.model_dump_json()) == message


@pytest.mark.parametrize("payload", [
    {"role": "user"}, {"role": "user", "text": " \t"},
    {"role": "assistant"}, {"role": "tool"},
    {"role": "user", "text": "Hi.", "tool_calls": [call()]},
    {"role": "assistant", "text": "Ready.", "tool_results": [result()]},
    {"role": "tool", "text": "Not allowed.", "tool_results": [result()]},
    {"role": "assistant", "tool_calls": [call(), call()]},
    {"role": "tool", "tool_results": [result(), result()]},
    {"role": "system", "text": "Separate instruction."},
    {"role": "user", "text": "Hi.", "extra": 1},
])
def test_invalid_messages_are_rejected(payload):
    with pytest.raises(ValidationError):
        LLMMessage.model_validate(payload)


def test_message_lists_are_not_shared():
    first = LLMMessage(role="assistant", text="First.")
    second = LLMMessage(role="assistant", text="Second.")
    first.tool_calls.append(call())
    assert second.tool_calls == []


def test_unknown_usage_is_explicit_and_totals_are_not_invented():
    unknown = TokenUsage(input_tokens=None, output_tokens=None, total_tokens=None)
    assert unknown.total_tokens is None
    usage = TokenUsage(input_tokens=10, output_tokens=2, total_tokens=15)
    assert usage.total_tokens == 15


@pytest.mark.parametrize("value", [-1, True, "10"])
def test_usage_counts_are_strict_nonnegative_integers(value):
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=value, output_tokens=0, total_tokens=0)


@pytest.mark.parametrize("changes", [
    {"tool_calls": [call(), call()]}, {"latency_ms": -1.0},
    {"latency_ms": float("nan")}, {"structured_output": {"value": float("inf")}},
])
def test_response_rejects_duplicate_ids_and_nonfinite_values(changes):
    with pytest.raises(ValidationError):
        LLMResponse.model_validate({
            "tool_calls": [], "text": "Ready.", "structured_output": None,
            "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
            "finish_reason": "completed", "latency_ms": 1.0, **changes,
        })
