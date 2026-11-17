"""Export immutable evaluation outputs to MLflow without repeating API calls."""

import argparse
import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from healthequity.cli import _write_json

EXPERIMENT = "agent-evals"
SUMMARY_METRICS = (
    "scheduled", "valid_cases", "invalid_cases", "completed_valid_audits",
    "tool_selection_accuracy", "numeric_accuracy", "schema_validity_rate",
    "suppression_correctness", "refusal_correctness", "automated_final_claims_pass_rate",
    "unsupported_claim_rate", "model_plan_proposal_accuracy", "model_plan_proposal_count",
    "live_benchmark_passed",
)
CASE_METRICS = (
    "provider_calls", "elapsed_seconds", "input_tokens", "output_tokens",
    "estimated_cost_usd", "numeric_checks", "numeric_checks_passed",
    "schema_valid", "numeric_pass", "suppression_correct", "tool_selection_correct",
    "automated_claims_pass", "schema_retry_or_failure_count", "claim_rewrite_count",
    "severity_correction_count", "model_proposals_correct", "model_proposals_total",
)


def case_passed(row):
    if row["expected_status"] == "refuse":
        return row["status"] == "refused" and row.get("refusal_correct") is True and row["provider_calls"] == 0
    return row["status"] == "completed" and all(
        row.get(key) is True for key in (
            "schema_valid", "numeric_pass", "suppression_correct",
            "tool_selection_correct", "automated_claims_pass",
        )
    )


def log_evaluation(directory, *, tracking_uri=None, client=None):
    """Log parent/child runs and an allowlist of aggregate artifacts.

    The receipt prevents repeated exports of unchanged outputs to the same store.
    An export failure is surfaced, terminates the parent as FAILED, and never
    changes a scorecard's evaluation status or human-review state.
    """
    directory = Path(directory).resolve()
    source = directory / "scorecard.json"
    source_bytes = source.read_bytes()
    scorecard = json.loads(source_bytes)
    rows = scorecard["cases"]
    if not rows or any(not re.fullmatch(r"[a-z][a-z0-9_]*", r["scenario"]) for r in rows):
        raise ValueError("Invalid or empty evaluation scenario list.")
    if len({r["scenario"] for r in rows}) != len(rows):
        raise ValueError("Evaluation scenario IDs must be unique.")
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    uri_hash = hashlib.sha256(tracking_uri.encode()).hexdigest()
    # Trace/findings bytes also define the export identity. A later edit must
    # never inherit an earlier receipt merely because the scorecard is unchanged.
    artifacts = {"suite": [source]}
    if (directory / "scorecard.md").exists():
        artifacts["suite"].append(directory / "scorecard.md")
    for row in rows:
        name = row["scenario"]
        artifacts[name] = [directory / f"{name}_{kind}.json" for kind in ("trace", "findings")
                           if (directory / f"{name}_{kind}.json").exists()]
    digest = hashlib.sha256()
    for path in sorted((p for paths in artifacts.values() for p in paths), key=lambda p: p.name):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    artifact_hash = digest.hexdigest()
    identity = hashlib.sha256(f"{artifact_hash}:{uri_hash}:{EXPERIMENT}".encode()).hexdigest()
    receipt_path = directory / "mlflow-export.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"exports": {}}
    if identity in receipts["exports"]:
        return {**receipts["exports"][identity], "reused": True}
    if client is None:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment is not None else client.create_experiment(
        EXPERIMENT, artifact_location=(Path("mlartifacts") / EXPERIMENT).resolve().as_uri(),
    )
    tags = {
        "evaluation.mode": scorecard["mode"], "evaluation.source_batch": directory.name,
        "evaluation.source_sha256": source_hash,
        "evaluation.human_review": "pending",
        "evaluation.benchmark_status": scorecard["benchmark_status"],
        "evaluation.numeric_scope": scorecard["numeric_accuracy_scope"],
    }
    if os.getenv("GITHUB_SHA"):
        tags["mlflow.source.git.commit"] = os.environ["GITHUB_SHA"]
    parent = client.create_run(experiment_id, tags=tags,
                               run_name=f"{scorecard['mode']}-{directory.name}")
    parent_id = parent.info.run_id
    children = {}

    def metrics(run_id, values, names):
        for key in names:
            value = values.get(key)
            if isinstance(value, (int, float, bool)) and math.isfinite(value):
                client.log_metric(run_id, key, float(value))

    try:
        client.log_param(parent_id, "model", scorecard["model"])
        client.log_param(parent_id, "mode", scorecard["mode"])
        client.log_param(parent_id, "source_batch", directory.name)
        for key, price in scorecard.get("prices_usd_per_million", {}).items():
            client.log_param(parent_id, f"{key}_usd_per_million", "unknown" if price is None else price)
        metrics(parent_id, scorecard, SUMMARY_METRICS)
        latencies = [r["elapsed_seconds"] for r in rows
                     if r["expected_status"] == "audit" and r.get("elapsed_seconds") is not None]
        if latencies:
            client.log_metric(parent_id, "mean_audit_seconds", sum(latencies) / len(latencies))
        for path in artifacts["suite"]:
            client.log_artifact(parent_id, str(path), artifact_path="evaluation")
        for row in rows:
            name = row["scenario"]
            child = client.create_run(experiment_id, tags={
                "mlflow.parentRunId": parent_id, "evaluation.mode": scorecard["mode"],
                "evaluation.status": row["status"], "evaluation.human_review": "pending",
                "evaluation.scenario": name,
            }, run_name=name)
            child_id = child.info.run_id
            children[name] = child_id
            try:
                client.log_param(child_id, "expected_status", row["expected_status"])
                if row.get("error_type"):
                    client.set_tag(child_id, "evaluation.error_type", row["error_type"])
                metrics(child_id, row, CASE_METRICS)
                if row["expected_status"] == "refuse":
                    metrics(child_id, row, ["refusal_correct"])
                for path in artifacts[name]:
                    client.log_artifact(child_id, str(path), artifact_path="evaluation")
                client.set_terminated(child_id, status="FINISHED" if case_passed(row) else "FAILED")
            except Exception:
                client.set_terminated(child_id, status="FAILED")
                raise
        client.set_terminated(parent_id, status="FINISHED" if all(case_passed(r) for r in rows) else "FAILED")
    except Exception:
        client.set_terminated(parent_id, status="FAILED")
        raise
    receipt = {"run_id": parent_id, "case_run_ids": children, "experiment": EXPERIMENT,
               "source_sha256": source_hash, "artifact_sha256": artifact_hash,
               "tracking_uri_sha256": uri_hash, "logged_at": datetime.now(UTC).isoformat()}
    receipts["exports"][identity] = receipt
    _write_json(receipt_path, receipts)
    return {**receipt, "reused": False}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m evals.tracking")
    parser.add_argument("directory", type=Path, help="Existing directory containing scorecard.json.")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args(argv)
    try:
        receipt = log_evaluation(args.directory, tracking_uri=args.tracking_uri)
    except Exception as error:  # noqa: BLE001
        print(f"MLflow export failed ({type(error).__name__}); evaluation outputs are retained.")
        return 2
    print(f"MLflow experiment: {EXPERIMENT}; run: {receipt['run_id']}; reused: {receipt['reused']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
