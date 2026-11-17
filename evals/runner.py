"""Run/checkpoint evaluation cases and save aggregate artifacts only."""

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError

from app.agent import AuditAgent
from app.agent_config import AgentConfig
from app.agent_errors import AgentRunError
from app.llm_config import LLMConfig, load_llm_config
from app.openai_provider import OpenAIProvider
from app.tools import AuditToolSession
from app.validation import validate_dataset
from evals.offline_provider import OfflineProvider
from evals.scenarios import build_scenarios
from evals.scoring import score_findings, summarize
from fairness.policy import SeverityPolicy, SuppressionPolicy
from healthequity.cli import _write_json


class CountingProvider:
    def __init__(self, provider):
        self.provider = provider
        self.calls = 0

    @property
    def model_name(self):
        return self.provider.model_name

    def generate(self, **kwargs):
        self.calls += 1
        return self.provider.generate(**kwargs)


def usage_totals(trace):
    """Unknown usage (including transport errors) remains unknown, never zero."""
    responses = [turn.get("response") for turn in trace.get("turns", [])]
    usage = {}
    for name in ("input_tokens", "output_tokens"):
        values = [r.get("usage", {}).get(name) if r else None for r in responses]
        usage[name] = sum(values) if values and all(v is not None for v in values) else None
    return usage


def schema_failure_count(trace):
    """Claim rejections and provider errors are distinct from schema failures."""
    schema_errors = {
        "Structured output parsing failed.",
        "AuditFindings schema validation failed.",
    }
    return sum(t.get("structured_response_requested", False)
               and t.get("error") in schema_errors for t in trace.get("turns", []))


def run_suite(cases, *, output_dir, mode, llm_config, provider_factory,
              input_price=None, output_price=None, progress=print):
    """Factory is called only after validation, once per valid case.

    All fixture cases are checkpointed, including unexpected refusals, provider
    failures and scorer failures. API exception text is never persisted.
    """
    if mode not in {"live", "offline"}:
        raise ValueError("Mode must be live or offline.")
    if (input_price is None) != (output_price is None):
        raise ValueError("Supply both input and output prices, or neither.")
    for value in (input_price, output_price):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Token prices must be finite and nonnegative.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [{"scenario": case.name, "expected_status": case.expected["status"],
             "status": "not_run", "provider_calls": 0, "refusal_correct": False,
             "human_review": "pending", "input_tokens": None, "output_tokens": None,
             "estimated_cost_usd": None, "elapsed_seconds": None} for case in cases]

    def checkpoint():
        scorecard = {**summarize(rows, mode), "model": llm_config.model_name,
                     "started_at": started_at, "updated_at": datetime.now(UTC).isoformat(),
                     "prices_usd_per_million": {"input": input_price, "output": output_price},
                     "cases": rows}
        _write_json(output_dir / "scorecard.json", scorecard)
        (output_dir / "scorecard.md").write_text(render_scorecard(scorecard), encoding="utf-8")
        return scorecard

    started_at = datetime.now(UTC).isoformat()
    checkpoint()
    for index, case in enumerate(cases):
        progress(f"[{mode}] {case.name}: starting", flush=True)
        started = perf_counter()
        row = {"scenario": case.name, "expected_status": case.expected["status"],
               "status": "failed", "provider_calls": 0, "refusal_correct": False,
               "human_review": "pending", "input_tokens": None, "output_tokens": None,
               "estimated_cost_usd": None}
        trace = None
        provider = None
        try:
            validate_dataset(case.data, case.mapping)
        except ValueError as error:
            correct = (case.expected["status"] == "refuse"
                       and case.expected["error_contains"] in str(error))
            row.update(status="refused", refusal_correct=correct,
                       error_type=type(error).__name__)
        else:
            if case.expected["status"] == "refuse":
                row.update(error_type="ExpectedRefusalNotRaised")
            else:
                try:
                    session = AuditToolSession(
                        case.data, case.mapping,
                        suppression_policy=SuppressionPolicy(**case.expected["policy"]),
                        severity_policy=SeverityPolicy(),
                    )
                    provider = CountingProvider(provider_factory(session))
                    result = AuditAgent(
                        provider=provider, session=session, llm_config=llm_config,
                        config=AgentConfig(),
                    ).run()
                    trace = result.trace.model_dump(mode="json")
                    findings = result.findings.model_dump(mode="json")
                    _write_json(output_dir / f"{case.name}_findings.json", findings)
                    row.update(score_findings(case, findings, trace))
                    row["status"] = "completed"
                except AgentRunError as error:
                    trace = error.trace.model_dump(mode="json")
                    row["error_type"] = "AgentRunError"
                except (ValidationError, ValueError) as error:
                    row["error_type"] = type(error).__name__
                except OSError:
                    raise  # Never disguise storage failures as an agent failure.
                except Exception as error:  # noqa: BLE001
                    row["error_type"] = type(error).__name__
        if provider is not None:
            row["provider_calls"] = provider.calls
        if trace is not None:
            _write_json(output_dir / f"{case.name}_trace.json", trace)
            row["trace_id"] = trace["run_id"]
            row.update(usage_totals(trace))
            row["schema_retry_or_failure_count"] = schema_failure_count(trace)
            row["claim_rewrite_count"] = max(0, len(trace["claims_validation_attempts"]) - 1)
            row["severity_correction_count"] = len(trace["severity_corrections"])
            if input_price is not None and all(row[k] is not None for k in ("input_tokens", "output_tokens")):
                row["estimated_cost_usd"] = (
                    row["input_tokens"] * input_price + row["output_tokens"] * output_price
                ) / 1_000_000
        if mode == "offline":
            row["estimated_cost_usd"] = 0.0
        row["elapsed_seconds"] = round(perf_counter() - started, 3)
        rows[index] = row
        checkpoint()
        progress(f"[{mode}] {case.name}: {row['status']} ({row['elapsed_seconds']}s)", flush=True)
    return checkpoint()


def render_scorecard(scorecard):
    def pct(value):
        return "unknown" if value is None else f"{100 * value:.1f}%"

    lines = [
        "# Agent evaluation scorecard", "",
        f"Mode: **{scorecard['mode']}**. Model: `{scorecard['model']}`.", "",
        "Offline uses a scripted provider and is not measured LLM performance.", "",
        "| Measure | Result |", "|---|---:|",
    ]
    for field in ("tool_selection_accuracy", "numeric_accuracy", "schema_validity_rate",
                  "suppression_correctness", "refusal_correctness", "automated_final_claims_pass_rate"):
        lines.append(f"| {field} | {pct(scorecard[field])} |")
    lines += ["", scorecard["numeric_accuracy_scope"], "",
              ("Tool selection covers successful execution, attribute/rate/metric coverage, selected test methods, "
              "and seeded 1,000-resample intervals. Model plan proposals are scored separately; "
              "delegating to the deterministic planner is not a correct model proposal."), "",
              ("Unsupported narrative claim rate: **not established**; human review pending. "
              "Automated claims validation cannot certify unrestricted prose. Benchmark remains open."), "",
              "| Scenario | Status | Tool selection | Numeric | Seconds | Est. USD |",
              "|---|---|---:|---:|---:|---:|"]
    for row in scorecard["cases"]:
        cost = row["estimated_cost_usd"]
        lines.append(
            f"| {row['scenario']} | {row['status']} | {row.get('tool_selection_correct', 'n/a')} "
            f"| {row.get('numeric_pass', 'n/a')} | {row['elapsed_seconds']} "
            f"| {cost if cost is not None else 'unknown'} |"
        )
    lines += ["", ("Failed valid audits remain in every audit-level rate denominator. "
              "Numeric accuracy is the fraction of valid scenarios passing every scoped numeric check, "
              "not a token-weighted fraction. Prices are explicitly supplied USD per million tokens; "
              "without prices or complete usage, live cost is unknown. Failed transport requests may "
              "incur unreported usage. Historical failed runs are not overwritten."), ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m evals")
    parser.add_argument("--live", action="store_true", help="Use the configured API provider; consumes tokens.")
    parser.add_argument("--scenario", action="append", help="Select case IDs; repeat for several cases.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluations"))
    parser.add_argument("--input-price", type=float, help="USD per million input tokens, supplied explicitly.")
    parser.add_argument("--output-price", type=float, help="USD per million output tokens, supplied explicitly.")
    parser.add_argument("--mlflow", action="store_true", help="Log the saved suite and per-case outputs to agent-evals.")
    parser.add_argument("--tracking-uri", default=None, help="MLflow URI; defaults to MLFLOW_TRACKING_URI or sqlite:///mlflow.db.")
    args = parser.parse_args(argv)
    try:
        cases = build_scenarios()
        if args.scenario:
            names = set(args.scenario)
            if names - {case.name for case in cases}:
                parser.error("Unknown scenario ID.")
            cases = [case for case in cases if case.name in names]
        config = load_llm_config() if args.live else LLMConfig(model_name="offline-scripted")
        factory = (lambda session: OpenAIProvider(config)) if args.live else OfflineProvider
        directory = args.output_dir / uuid4().hex
        scorecard = run_suite(
            cases, output_dir=directory, mode="live" if args.live else "offline",
            llm_config=config, provider_factory=factory,
            input_price=args.input_price, output_price=args.output_price,
        )
        print(f"Scorecard: {directory / 'scorecard.md'}", flush=True)
        if args.mlflow:
            from evals.tracking import log_evaluation
            try:
                receipt = log_evaluation(directory, tracking_uri=args.tracking_uri)
            except Exception as error:  # noqa: BLE001
                print(f"MLflow export failed ({type(error).__name__}); evaluation outputs are retained.")
                return 2
            print(f"MLflow agent-evals run: {receipt['run_id']}", flush=True)
        audits_ok = scorecard["completed_valid_audits"] == scorecard["valid_cases"]
        checks_ok = all(r.get("numeric_pass") and r.get("tool_selection_correct")
                        and r.get("suppression_correct") and r.get("automated_claims_pass")
                        for r in scorecard["cases"] if r["expected_status"] == "audit")
        refusals_ok = all(r["refusal_correct"] for r in scorecard["cases"] if r["expected_status"] == "refuse")
        return 0 if audits_ok and checks_ok and refusals_ok else 1
    except (OSError, ValueError) as error:
        print(f"Evaluation unavailable ({type(error).__name__}); check configuration, source, and output directory.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
