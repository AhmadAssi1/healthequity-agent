"""Provider schema compatibility probe stays bounded and diagnostic text redacted."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import BadRequestError

from app.llm_config import LLMConfig
from app.summary_grounding import SUMMARY_PLACEHOLDER
from evals.probe import probe_schema, safe_bad_request_message


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_probe_uses_exact_runtime_schema_in_one_capped_request(status):
    create = Mock(return_value=SimpleNamespace(status=status, error=None))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = probe_schema(LLMConfig(model_name="test-model"), client)
    assert result["schema_accepted"]
    assert create.call_count == 1
    request = create.call_args.kwargs
    assert request["max_output_tokens"] == 16 and request["store"] is False
    schema = request["text"]["format"]["schema"]
    assert schema["properties"]["summary"] == {"type": "string", "enum": [SUMMARY_PLACEHOLDER]}
    assert schema["$defs"]["AttributeFindings"]["properties"]["attribute"]["enum"] == ["group"]
    assert "tools" not in request
    assert "row_count" not in request["input"]


def test_probe_rejection_makes_no_retry():
    response = SimpleNamespace(status_code=400, headers={}, request=SimpleNamespace(method="POST", url="https://example.invalid"))
    error = BadRequestError("Rejected", response=response, body={"message": "Invalid schema"})
    create = Mock(side_effect=error)
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    with pytest.raises(BadRequestError):
        probe_schema(LLMConfig(model_name="test-model"), client)
    assert create.call_count == 1


def test_bad_request_diagnostics_do_not_print_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-key-value")
    response = SimpleNamespace(status_code=400, headers={}, request=SimpleNamespace(method="POST", url="https://example.invalid"))
    error = BadRequestError("Rejected", response=response, body={
        "message": "Invalid schema: private-key-value sk-proj-secret-token pattern not allowed",
    })
    message = safe_bad_request_message(error)
    assert "private-key-value" not in message and "sk-proj-secret-token" not in message
    assert "pattern not allowed" in message
