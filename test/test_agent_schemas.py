"""Run limits and audit-trace consistency at the serialization boundary."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.agent_config import AgentConfig
from app.agent_schemas import (
    AgentRunTrace,
    AgentTurnRecord,
    SeverityCorrection,
    ToolExecutionRecord,
)
from app.llm_config import LLMConfig
from app.llm_schemas import LLMMessage, LLMResponse, TokenUsage, ToolCall, ToolResult


def call(call_id="call_1"):
    return ToolCall(call_id=call_id, name="profile_dataset", arguments={})


def execution(call_id="call_1"):
    return ToolExecutionRecord(
        call=call(call_id),
        result=ToolResult(call_id=call_id, name="profile_dataset", output={}, is_error=False),
        latency_ms=1.0,
    )


def response(calls=None):
    return LLMResponse(
        tool_calls=calls or [], text="Ready.", structured_output=None,
        usage=TokenUsage(input_tokens=None, output_tokens=None, total_tokens=None),
        finish_reason="completed", latency_ms=1.0,
    )


@pytest.fixture
def trace_payload():
    return {
        "run_id": "run_test", "agent_config": AgentConfig().model_dump(),
        "llm_config": LLMConfig(model_name="test-model").model_dump(),
        "started_at": datetime.now(UTC), "finished_at": None, "status": "running",
        "system_instruction": "Audit using trusted evidence.",
        "initial_messages": [LLMMessage(role="user", text="Audit.").model_dump()],
        "turns": [], "error": None,
    }


@pytest.mark.parametrize("changes", [
    {"max_tool_calls": 0}, {"max_tool_calls": 13}, {"max_tool_calls": True},
    {"max_model_turns": 0}, {"max_model_turns": 25},
    {"max_schema_retries": -1}, {"max_schema_retries": 2},
    {"max_duration_seconds": 0.0}, {"max_duration_seconds": 181.0},
    {"max_duration_seconds": float("nan")}, {"max_duration_seconds": float("inf")},
])
def test_agent_config_enforces_execution_limits(changes):
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(changes)


@pytest.mark.parametrize("changes", [
    {"call_id": "wrong"}, {"name": "other_tool"},
])
def test_tool_execution_requires_matching_result(changes):
    payload = execution().model_dump()
    payload["result"].update(changes)
    with pytest.raises(ValidationError, match="must match"):
        ToolExecutionRecord.model_validate(payload)


@pytest.mark.parametrize("payload", [
    {"response": None},
    {"response": None, "error": "Failed.", "tool_executions": [execution()]},
    {"response": response(), "tool_executions": [execution()]},
    {"response": response([call(), call("call_2")]), "tool_executions": [execution("call_2")]},
])
def test_turn_rejects_missing_errors_unrequested_tools_and_wrong_order(payload):
    with pytest.raises(ValidationError):
        AgentTurnRecord.model_validate({"turn_number": 1, "latency_ms": 1.0, **payload})


def test_partial_execution_prefix_is_valid_for_a_stopped_turn():
    turn = AgentTurnRecord(
        turn_number=1, response=response([call(), call("call_2")]),
        tool_executions=[execution()], error="Stopped after the first tool.", latency_ms=2.0,
    )
    assert AgentTurnRecord.model_validate_json(turn.model_dump_json()) == turn


@pytest.mark.parametrize("changes", [
    {"status": "running", "error": "Unexpected."},
    {"status": "running", "finished_at": datetime.now(UTC) + timedelta(days=1)},
    {"status": "completed"}, {"status": "failed"},
    {"status": "failed", "finished_at": datetime.now(UTC) + timedelta(days=1)},
    {"status": "completed", "finished_at": datetime.now(UTC) + timedelta(days=1), "error": "Failed."},
    {"started_at": datetime(2026, 1, 1)}, {"initial_messages": []},
])
def test_trace_rejects_inconsistent_lifecycle_and_naive_time(trace_payload, changes):
    with pytest.raises(ValidationError):
        AgentRunTrace.model_validate({**trace_payload, **changes})


def test_finish_time_cannot_precede_start(trace_payload):
    trace_payload.update(status="failed", error="Stopped.")
    trace_payload["finished_at"] = trace_payload["started_at"] - timedelta(seconds=1)
    with pytest.raises(ValidationError, match="precede"):
        AgentRunTrace.model_validate(trace_payload)


def test_turn_numbers_must_be_consecutive(trace_payload):
    trace_payload["turns"] = [AgentTurnRecord(turn_number=2, response=response(), latency_ms=1.0)]
    with pytest.raises(ValidationError, match="consecutive"):
        AgentRunTrace.model_validate(trace_payload)


def test_trace_enforces_model_turn_budget(trace_payload):
    trace_payload["agent_config"]["max_model_turns"] = 1
    trace_payload["turns"] = [
        AgentTurnRecord(turn_number=i, response=response(), latency_ms=1.0) for i in [1, 2]
    ]
    with pytest.raises(ValidationError, match="model-turn limit"):
        AgentRunTrace.model_validate(trace_payload)


def test_trace_enforces_executed_tool_budget(trace_payload):
    trace_payload["agent_config"]["max_tool_calls"] = 1
    trace_payload["turns"] = [AgentTurnRecord(
        turn_number=1, response=response([call(), call("call_2")]),
        tool_executions=[execution(), execution("call_2")], latency_ms=2.0,
    )]
    with pytest.raises(ValidationError, match="tool-call limit"):
        AgentRunTrace.model_validate(trace_payload)


def severity_correction_payload():
    return {
        "attribute": "group", "metric": "ppv_difference",
        "proposed_severity": "watch", "corrected_severity": "no_evidence_found",
        "reason": "Recomputed from trusted evidence using the severity policy.",
    }


def test_severity_correction_round_trips_with_trace(trace_payload):
    trace = AgentRunTrace.model_validate({
        **trace_payload, "severity_corrections": [severity_correction_payload()],
    })
    restored = AgentRunTrace.model_validate_json(trace.model_dump_json())
    assert restored == trace
    assert restored.severity_corrections[0].proposed_severity == "watch"
    assert restored.severity_corrections[0].corrected_severity == "no_evidence_found"


@pytest.mark.parametrize("changes", [
    {"corrected_severity": "watch"}, {"proposed_severity": "high"},
    {"metric": "accuracy"}, {"attribute": "   "}, {"reason": ""},
    {"unknown_field": 1},
])
def test_severity_correction_rejects_invalid_or_unchanged_records(changes):
    with pytest.raises(ValidationError):
        SeverityCorrection.model_validate({**severity_correction_payload(), **changes})


def test_older_traces_default_to_independent_empty_correction_lists(trace_payload):
    first = AgentRunTrace.model_validate(trace_payload)
    second = AgentRunTrace.model_validate(trace_payload)
    first.severity_corrections.append(SeverityCorrection(**severity_correction_payload()))
    assert second.severity_corrections == []
