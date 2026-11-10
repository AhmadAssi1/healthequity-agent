"""Validated configuration for language-model providers."""
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field

from app.audit_schemas import AuditBaseModel


class LLMConfig(AuditBaseModel):
    provider: Literal["openai"] = "openai"
    model_name: str = Field(min_length=1, pattern=r"^\S+$")
    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=180,
    )

def load_llm_config() -> LLMConfig:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path, override=False)

    model_name = os.getenv("OPENAI_MODEL")

    if model_name is None:
        raise ValueError(
            "OPENAI_MODEL must be set."
        )

    try:
        timeout_seconds = float(
            os.getenv("OPENAI_TIMEOUT_SECONDS", "30")
        )
    except ValueError:
        raise ValueError(
            "OPENAI_TIMEOUT_SECONDS must be a number."
        ) from None

    return LLMConfig(
        model_name=model_name,
        timeout_seconds=timeout_seconds,
    )