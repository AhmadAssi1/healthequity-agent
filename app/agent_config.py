"""Execution limits for the audit agent."""

from pydantic import Field

from app.audit_schemas import AuditBaseModel


class AgentConfig(AuditBaseModel):
    max_claim_rewrites: int = Field(default=1, ge=0, le=1)
    max_tool_calls: int = Field(
        default=12,
        ge=1,
        le=12,
    )
    max_model_turns: int = Field(
        default=16,
        ge=1,
        le=24,
    )
    max_schema_retries: int = Field(
        default=1,
        ge=0,
        le=1,
    )
    max_duration_seconds: float = Field(
        default=180.0,
        gt=0,
        le=180,
    )
