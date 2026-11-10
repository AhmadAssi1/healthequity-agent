"""Configuration loading without reading the real project secrets."""

import pytest
from pydantic import ValidationError

from app.llm_config import LLMConfig, load_llm_config


@pytest.fixture
def isolated_env(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_TIMEOUT_SECONDS", raising=False)
    loader = []
    monkeypatch.setattr("app.llm_config.load_dotenv", lambda path, override: loader.append((path, override)))
    return monkeypatch, loader


def test_defaults_and_provider_are_openai_only():
    config = LLMConfig(model_name="gpt-4.1-mini")
    assert config.provider == "openai"
    assert config.timeout_seconds == 30.0


@pytest.mark.parametrize("changes", [
    {"provider": "gemini"}, {"model_name": ""}, {"model_name": "bad model"},
    {"timeout_seconds": 0}, {"timeout_seconds": 181.0},
    {"timeout_seconds": float("inf")}, {"timeout_seconds": float("nan")},
    {"timeout_seconds": "30"}, {"extra": True},
])
def test_config_rejects_invalid_values(changes):
    with pytest.raises(ValidationError):
        LLMConfig.model_validate({"model_name": "gpt-4.1-mini", **changes})


def test_environment_values_are_loaded_and_override_disabled(isolated_env):
    monkeypatch, calls = isolated_env
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "17.5")
    config = load_llm_config()
    assert config.model_name == "test-model"
    assert config.timeout_seconds == 17.5
    assert len(calls) == 1 and calls[0][1] is False
    assert calls[0][0].name == ".env"


def test_missing_timeout_uses_default(isolated_env):
    monkeypatch, _ = isolated_env
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    assert load_llm_config().timeout_seconds == 30.0


def test_missing_model_fails_clearly(isolated_env):
    with pytest.raises(ValueError, match="OPENAI_MODEL must be set"):
        load_llm_config()


@pytest.mark.parametrize("value", ["", "bad", "NaN", "Infinity", "0", "181"])
def test_invalid_timeout_from_environment_fails(isolated_env, value):
    monkeypatch, _ = isolated_env
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError):
        load_llm_config()
