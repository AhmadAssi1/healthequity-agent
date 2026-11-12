"""Run a guarded audit and persist aggregate artifacts, including failures."""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError

from app.agent import AuditAgent
from app.agent_config import AgentConfig
from app.agent_errors import AgentRunError
from app.data import load_data
from app.llm_config import load_llm_config
from app.openai_provider import OpenAIProvider
from app.schemas import ColumnMapping
from app.tools import AuditToolSession
from app.validation import create_dataset_fingerprint, validate_dataset

DEFAULT_GROUPS = ("race", "gender", "age", "payer_code")


def _write_json(path: Path, payload: dict) -> None:
    """Replace complete JSON atomically; never leave a partially written artifact."""
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m healthequity")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit a CSV containing binary outcomes and predictions.")
    audit.add_argument("file", type=Path)
    audit.add_argument("--outcome", default="outcome")
    audit.add_argument("--prediction", default="prediction")
    audit.add_argument("--groups", nargs="+", help="Group columns; defaults to present race/gender/age/payer_code columns.")
    score = audit.add_mutually_exclusive_group()
    score.add_argument("--score", help="Risk-score column; detects 'score' when omitted.")
    score.add_argument("--no-score", action="store_true", help="Ignore any risk-score column.")
    audit.add_argument("--output-dir", type=Path, default=Path("outputs/agent_runs"))
    audit.add_argument("--max-seconds", type=float, default=180.0, help="Agent budget, up to 180 seconds.")
    return parser


def _audit(args) -> int:
    started = perf_counter()
    directory = args.output_dir / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "phase": "input_validation",
        "started_at": datetime.now(UTC).isoformat(), "finished_at": None,
        "elapsed_seconds": None, "trace_id": None, "error": None,
    }
    # Confirm artifact storage works before creating a provider or spending tokens.
    _write_json(directory / "run.json", record)
    print(f"Artifacts: {directory}", flush=True)

    def finish(status, error=None):
        record.update(status=status, error=error, finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=round(perf_counter() - started, 3))
        _write_json(directory / "run.json", record)

    try:
        if args.file.suffix.lower() != ".csv":
            raise ValueError("This command supports CSV files only.")
        frame = load_data(args.file)
        groups = args.groups if args.groups is not None else [g for g in DEFAULT_GROUPS if g in frame.columns]
        if not groups:
            raise ValueError("No default group columns found; supply --groups COLUMN [COLUMN ...].")
        score = None if args.no_score else args.score or ("score" if "score" in frame.columns else None)
        mapping = ColumnMapping(outcome=args.outcome, prediction=args.prediction, groups=groups, score=score)
        validate_dataset(frame, mapping)
        config = AgentConfig(max_duration_seconds=args.max_seconds)
        session = AuditToolSession(frame, mapping)
        record.update(mapping=mapping.model_dump(), fingerprint=create_dataset_fingerprint(frame).model_dump())
    except ValidationError:
        finish("failed", "Invalid column mapping or agent budget; check --help and column names.")
        print(f"Input error: {record['error']}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as error:
        # Validation messages describe schema/columns, not rows. Do not expose
        # parser exceptions, which can include fragments of a patient record.
        message = str(error) if type(error) is ValueError else f"Unable to read CSV ({type(error).__name__})."
        finish("failed", message)
        print(f"Input error: {message}", file=sys.stderr)
        return 2

    record["phase"] = "provider_setup"
    _write_json(directory / "run.json", record)
    try:
        llm_config = load_llm_config()
        provider = OpenAIProvider(llm_config)
        agent = AuditAgent(provider=provider, session=session, llm_config=llm_config, config=config)
        record.update(phase="audit", llm_config=llm_config.model_dump())
        _write_json(directory / "run.json", record)
        print(f"Dataset rows: {len(frame)}\nRunning live audit...", flush=True)
        result = agent.run()
    except AgentRunError as error:
        record["trace_id"] = error.trace.run_id
        _write_json(directory / "trace.json", error.trace.model_dump(mode="json"))
        if error.trace.claims_validation is not None:
            _write_json(directory / "claims_validation.json", error.trace.claims_validation.model_dump(mode="json"))
        finish("failed", str(error))
        print(f"Status: FAILED\nError: {error}\nTrace: {directory / 'trace.json'}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        finish("interrupted", "Interrupted by the user; a complete agent trace is unavailable.")
        print("Audit interrupted; run.json was saved.", file=sys.stderr)
        return 130
    except OSError:
        raise  # Storage errors must be reported as storage failures.
    except Exception as error:  # noqa: BLE001
        # Provider setup/transport exceptions may contain API credentials.
        finish("failed", f"Provider setup failed ({type(error).__name__}); check OpenAI configuration.")
        print(f"Error: {record['error']}", file=sys.stderr)
        return 2

    record.update(phase="saving", trace_id=result.trace.run_id)
    _write_json(directory / "trace.json", result.trace.model_dump(mode="json"))
    _write_json(directory / "findings.json", result.findings.model_dump(mode="json"))
    if result.claims_validation is not None:
        _write_json(directory / "claims_validation.json", result.claims_validation.model_dump(mode="json"))
    finish("completed")
    print(f"Status: completed\nElapsed seconds: {record['elapsed_seconds']}")
    print(f"Model turns: {len(result.trace.turns)}")
    print(f"Tool calls: {sum(len(t.tool_executions) for t in result.trace.turns)}")
    print(f"Severity corrections: {len(result.trace.severity_corrections)}")
    print(f"Claims validation: {result.claims_validation.passed if result.claims_validation else None}")
    print("Human review of narrative findings remains required.")
    print(f"Trace: {directory / 'trace.json'}\nFindings: {directory / 'findings.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _audit(args)
    except OSError as error:
        print(f"Cannot save audit artifacts ({type(error).__name__}); check output directory and disk space.", file=sys.stderr)
        return 3
