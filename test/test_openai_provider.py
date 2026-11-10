"""Offline provider tests using real SDK response models and an injected client."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai.types.responses import Response
from pydantic import ValidationError

from app.audit_schemas import AuditFindings
from app.llm_config import LLMConfig
from app.llm_provider import StructuredOutputError
from app.llm_schemas import LLMMessage, ToolCall, ToolResult
from app.openai_provider import OpenAIProvider
from app.tool_contracts import ToolDefinition


def test_audit_schema_requires_nullable_reasons_in_strict_final_request():
    provider, create = provider_with_response(sdk_response('{"example":true}'))
    schema = AuditFindings.model_json_schema()
    provider.generate(
        system_instruction="Return audit findings.",
        messages=[LLMMessage(role="user", text="Use supplied evidence.")],
        tools=[], response_schema=schema,
    )
    wire_format = create.call_args.kwargs["text"]["format"]
    assert wire_format["strict"] is True

    def check_objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for child in node.values():
                check_objects(child)
        elif isinstance(node, list):
            for child in node:
                check_objects(child)

    check_objects(wire_format["schema"])
    for name, field in (
        ("MetricResult", "reason"),
        ("FairnessFinding", "confidence_interval_reason"),
        ("StatisticalTestResult", "reason"),
    ):
        definition = wire_format["schema"]["$defs"][name]
        assert field in definition["required"]
        assert {"type": "null"} in definition["properties"][field]["anyOf"]
        assert "null" in definition["properties"][field]["description"]


def sdk_response(text=None, *, calls=None, status="completed", usage=True, refusal=False, reasoning=False):
    output = []
    if text is not None or refusal:
        output.append({
            "type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
            "content": ([{"type": "refusal", "refusal": "Test refusal."}] if refusal else [
                {"type": "output_text", "text": text, "annotations": []},
            ]),
        })
    output.extend(calls or [])
    if reasoning:
        output.append({"type": "reasoning", "id": "reasoning_test", "summary": []})
    return Response.model_validate({
        "id": "resp_test", "created_at": 0.0, "model": "gpt-4.1-mini",
        "object": "response", "output": output, "parallel_tool_calls": False,
        "tool_choice": "auto", "tools": [], "status": status,
        "usage": ({
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        } if usage else None),
    })


def sdk_call(arguments='{"attributes":["race"]}', call_id="call_1", status="completed"):
    return {
        "type": "function_call", "id": f"fc_{call_id}", "call_id": call_id,
        "name": "compute_fairness_metrics", "arguments": arguments, "status": status,
    }


@pytest.fixture
def tool():
    return ToolDefinition(
        name="compute_fairness_metrics", description="Compute group metrics.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object"}, available=True,
    )


def provider_with_response(response):
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))
    provider = OpenAIProvider(LLMConfig(model_name="gpt-4.1-mini"), client=client)
    return provider, client.responses.create


def test_client_constructor_receives_timeout_and_disabled_sdk_retries(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("app.openai_provider.OpenAI", factory)
    config = LLMConfig(model_name="gpt-4.1-mini", timeout_seconds=17.0)
    provider = OpenAIProvider(config)
    factory.assert_called_once_with(timeout=17.0, max_retries=0)
    config.model_name = "changed"
    assert provider.model_name == "gpt-4.1-mini"


def test_tool_declarations_exclude_unavailable_tools_and_copy_schemas(tool):
    disabled = tool.model_copy(update={"name": "draft_findings", "available": False})
    original = deepcopy(tool.input_schema)
    declarations = OpenAIProvider._tool_declarations([tool, disabled])
    assert len(declarations) == 1
    assert declarations[0] == {
        "type": "function", "name": tool.name, "description": tool.description,
        "parameters": original, "strict": False,
    }
    declarations[0]["parameters"]["properties"]["extra"] = {"type": "string"}
    assert tool.input_schema == original


@pytest.mark.parametrize("is_error", [False, True])
def test_message_history_preserves_call_ids_arguments_and_result_envelope(is_error):
    call = ToolCall(call_id="call_1", name="compute_fairness_metrics", arguments={"attributes": ["race"]})
    result = ToolResult(call_id=call.call_id, name=call.name, output={"value": None}, is_error=is_error)
    messages = [
        LLMMessage(role="user", text="Audit the profile."),
        LLMMessage(role="assistant", text="Checking groups.", tool_calls=[call]),
        LLMMessage(role="tool", tool_results=[result]),
    ]
    original = [message.model_dump() for message in messages]
    converted = OpenAIProvider._messages_to_input(messages)
    assert converted[:2] == [
        {"role": "user", "content": "Audit the profile."},
        {"role": "assistant", "content": "Checking groups."},
    ]
    assert converted[2]["type"] == "function_call"
    assert converted[2]["call_id"] == converted[3]["call_id"] == "call_1"
    assert json.loads(converted[2]["arguments"]) == call.arguments
    assert json.loads(converted[3]["output"]) == {"is_error": is_error, "output": result.output}
    assert [message.model_dump() for message in messages] == original


def test_parse_text_and_record_actual_usage_and_latency():
    result = OpenAIProvider._parse_response(sdk_response("Ready."), 12.5)
    assert result.text == "Ready."
    assert result.tool_calls == []
    assert result.structured_output is None
    assert result.usage.model_dump() == {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    assert result.finish_reason == "completed"
    assert result.latency_ms == 12.5


def test_parse_function_call_without_text():
    result = OpenAIProvider._parse_response(sdk_response(calls=[sdk_call()]), 1.0)
    assert result.text is None
    assert result.tool_calls[0].call_id == "call_1"
    assert result.tool_calls[0].arguments == {"attributes": ["race"]}


def test_missing_usage_is_unknown_not_zero():
    result = OpenAIProvider._parse_response(sdk_response("Ready.", usage=False), 1.0)
    assert result.usage.model_dump() == {"input_tokens": None, "output_tokens": None, "total_tokens": None}


def test_parse_structured_object_without_claiming_audit_validation():
    result = OpenAIProvider._parse_response(sdk_response('{"example":true}'), 1.0, True)
    assert result.structured_output == {"example": True}
    # The adapter parses JSON; AuditFindings validation belongs to orchestration.


@pytest.mark.parametrize("status", ["failed", "incomplete", "cancelled", "queued", "in_progress", None])
def test_incomplete_or_unsuccessful_responses_are_rejected(status):
    with pytest.raises(ValueError, match="not completed"):
        OpenAIProvider._parse_response(sdk_response("Partial.", status=status), 1.0)


def test_provider_error_is_not_treated_as_success():
    response = sdk_response("Ready.")
    response.error = {"code": "test_error"}
    with pytest.raises(ValueError, match="error response"):
        OpenAIProvider._parse_response(response, 1.0)


@pytest.mark.parametrize("status", ["incomplete", "in_progress"])
def test_incomplete_tool_call_is_rejected(status):
    with pytest.raises(ValueError, match="incomplete tool call"):
        OpenAIProvider._parse_response(sdk_response(calls=[sdk_call(status=status)]), 1.0)


@pytest.mark.parametrize("arguments", ["not JSON", "[]", "null", "42", '{"x":NaN}', '{"x":Infinity}'])
def test_tool_arguments_require_a_finite_json_object(arguments):
    with pytest.raises(ValueError):
        OpenAIProvider._parse_response(sdk_response(calls=[sdk_call(arguments)]), 1.0)


def test_duplicate_call_ids_are_rejected():
    with pytest.raises(ValidationError, match="unique"):
        OpenAIProvider._parse_response(sdk_response(calls=[sdk_call(), sdk_call()]), 1.0)


def test_refusals_are_explicit_failures():
    with pytest.raises(ValueError, match="refused"):
        OpenAIProvider._parse_response(sdk_response(refusal=True), 1.0)


def test_unsupported_output_is_not_silently_dropped():
    with pytest.raises(ValueError, match="Unsupported response item"):
        OpenAIProvider._parse_response(sdk_response("Ready.", reasoning=True), 1.0)


@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_response_is_not_accepted(text):
    with pytest.raises(ValueError, match="no text or tool calls"):
        OpenAIProvider._parse_response(sdk_response(text), 1.0)


@pytest.mark.parametrize("text", [None, "", "   ", "[]", "42", "null", "bad JSON", '{"x":NaN}'])
def test_structured_mode_requires_a_finite_json_object(text):
    with pytest.raises(ValueError):
        OpenAIProvider._parse_response(sdk_response(text), 1.0, True)


def test_structured_mode_rejects_tool_calls():
    with pytest.raises(ValueError, match="must not contain tool calls"):
        OpenAIProvider._parse_response(sdk_response(calls=[sdk_call()]), 1.0, True)


@pytest.mark.parametrize("text", [None, "", "bad JSON", "[]", '{"x":NaN}'])
def test_structured_errors_preserve_response_text_usage_and_latency(text):
    with pytest.raises(StructuredOutputError) as caught:
        OpenAIProvider._parse_response(sdk_response(text), 17.5, True)
    saved = caught.value.response
    assert saved.text == (text or None)
    assert saved.usage.total_tokens == 12
    assert saved.latency_ms == 17.5
    assert saved.structured_output is None


def test_refusal_is_not_classified_as_a_retryable_schema_error():
    with pytest.raises(ValueError) as caught:
        OpenAIProvider._parse_response(sdk_response(refusal=True), 1.0, True)
    assert not isinstance(caught.value, StructuredOutputError)


def test_generate_sends_stateless_function_call_request(tool):
    provider, create = provider_with_response(sdk_response(calls=[sdk_call()]))
    messages = [LLMMessage(role="user", text="Analyze the aggregate profile.")]
    result = provider.generate(system_instruction="Use audit tools.", messages=messages, tools=[tool])
    request = create.call_args.kwargs
    assert create.call_count == 1
    assert request["model"] == "gpt-4.1-mini"
    assert request["instructions"] == "Use audit tools."
    assert request["store"] is False
    assert request["parallel_tool_calls"] is False
    assert len(request["tools"]) == 1
    assert "text" not in request
    assert result.tool_calls[0].name == tool.name
    assert result.latency_ms >= 0


def test_generate_final_mode_sends_schema_without_tools_and_copies_input(tool):
    provider, create = provider_with_response(sdk_response('{"example":true}'))
    schema = {"type": "object", "properties": {"example": {"type": "boolean"}}}
    original = deepcopy(schema)
    result = provider.generate(
        system_instruction="Return structured findings.",
        messages=[LLMMessage(role="user", text="Summarize the evidence.")],
        tools=[tool], response_schema=schema,
    )
    request = create.call_args.kwargs
    assert "tools" not in request
    assert "parallel_tool_calls" not in request
    assert request["text"]["format"] == {
        "type": "json_schema", "name": "audit_findings", "schema": original, "strict": True,
    }
    request["text"]["format"]["schema"]["properties"].clear()
    assert schema == original
    assert result.structured_output == {"example": True}


def test_generate_omits_empty_tool_list():
    provider, create = provider_with_response(sdk_response("Ready."))
    provider.generate(system_instruction="Explain.", messages=[LLMMessage(role="user", text="Hello.")], tools=[])
    assert "tools" not in create.call_args.kwargs


@pytest.mark.parametrize("instruction, messages", [("   ", [LLMMessage(role="user", text="Hi.")]), ("Audit.", [])])
def test_invalid_request_is_rejected_before_calling_client(instruction, messages):
    provider, create = provider_with_response(sdk_response("Ready."))
    with pytest.raises(ValueError):
        provider.generate(system_instruction=instruction, messages=messages, tools=[])
    create.assert_not_called()


def test_transport_errors_propagate_without_hidden_retries():
    provider, create = provider_with_response(sdk_response("Ready."))
    create.side_effect = TimeoutError("Test timeout.")
    with pytest.raises(TimeoutError):
        provider.generate(system_instruction="Audit.", messages=[LLMMessage(role="user", text="Hi.")], tools=[])
    assert create.call_count == 1
