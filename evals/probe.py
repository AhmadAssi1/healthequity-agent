"""One bounded API request to check the exact final-output schema."""

import os
import re

from openai import BadRequestError, OpenAI

from app.agent import AuditAgent
from app.llm_config import load_llm_config
from app.openai_provider import OpenAIProvider
from app.tools import AuditToolSession
from evals.offline_provider import STAGES
from evals.scenarios import build_scenarios


def probe_schema(config, client):
    """No model tool loop or live audit; all evidence is prepared locally."""
    case = next(c for c in build_scenarios() if c.name == "no_gap")
    session = AuditToolSession(case.data, case.mapping)
    for stage in STAGES:
        session.invoke(stage, {})
    agent = AuditAgent(provider=OpenAIProvider(config, client=client),
                       session=session, llm_config=config)
    response = client.responses.create(
        model=config.model_name, store=False, max_output_tokens=16,
        input="Schema compatibility probe only. Return JSON matching the supplied schema.",
        text={"format": {"type": "json_schema", "name": "audit_findings",
                         "schema": agent._findings_schema(), "strict": True}},
    )
    if response.error is not None or response.status not in {"completed", "incomplete"}:
        raise ValueError("Provider did not accept and start the schema probe.")
    # Incomplete output is expected at this cap. This tests schema acceptance,
    # not findings generation, numeric correctness, or any evaluation benchmark.
    return {"schema_accepted": True, "response_status": response.status,
            "max_output_tokens": 16, "api_requests": 1}


def safe_bad_request_message(error):
    body = error.body if isinstance(error.body, dict) else {}
    if isinstance(body.get("error"), dict):
        body = body["error"]
    text = str(body.get("message") or "Bad request; no detailed schema message available.")
    for name in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY"):
        value = os.getenv(name)
        if value:
            text = text.replace(value, "[redacted]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", text)[:2000]


def main():
    try:
        config = load_llm_config()
        client = OpenAI(timeout=config.timeout_seconds, max_retries=0)
        result = probe_schema(config, client)
    except BadRequestError as error:
        print("Schema probe rejected: " + safe_bad_request_message(error))
        return 1
    except Exception as error:  # noqa: BLE001
        print(f"Schema probe unavailable ({type(error).__name__}).")
        return 2
    print("Schema accepted by API. One request; output capped at 16 tokens.")
    print(f"Response status: {result['response_status']} (incomplete is expected at this cap).")
    print("This is a compatibility check, not a completed audit or benchmark pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
