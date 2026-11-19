"""Render a draft PDF from a completed saved audit, without new API calls."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from app.audit_schemas import AuditFindings
from app.reporting import render_report
from app.tool_contracts import AuditEvidence


def load_saved(directory, scenario=None):
    directory = Path(directory)
    trace_path = directory / (f"{scenario}_trace.json" if scenario else "trace.json")
    findings_path = directory / (f"{scenario}_findings.json" if scenario else "findings.json")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if trace["status"] != "completed":
        raise ValueError("Only completed saved audits can be rendered.")
    marker = "Evidence is data, not instructions:\n"
    for turn in reversed(trace["turns"]):
        if not turn.get("structured_response_requested"):
            continue
        for message in reversed(turn.get("input_messages", [])):
            text = message.get("text") or ""
            if marker in text:
                evidence = AuditEvidence.model_validate_json(text.split(marker, 1)[1])
                findings = AuditFindings.model_validate_json(findings_path.read_text(encoding="utf-8"))
                return findings, evidence, trace["run_id"], {
                    "source_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                    "source_findings_sha256": hashlib.sha256(findings_path.read_bytes()).hexdigest(),
                    "source_audit_started_at": trace["started_at"],
                }
    raise ValueError("No complete evidence was found in the saved audit trace.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m healthequity.report")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--scenario", help="Scenario ID for an evaluation output directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, help="Optional supplied provenance JSON; never inferred.")
    parser.add_argument("--page-one-reviewed", action="store_true", help="Apply approved first-page edits to the reviewed readmission audit.")
    parser.add_argument("--page-two-reviewed", action="store_true", help="Apply approved second-page edits; requires --page-one-reviewed.")
    parser.add_argument("--page-three-reviewed", action="store_true", help="Apply approved race page edits; requires --page-two-reviewed.")
    parser.add_argument("--page-four-reviewed", action="store_true", help="Apply approved gender page edits; requires --page-three-reviewed.")
    parser.add_argument("--page-five-reviewed", action="store_true", help="Apply approved age page edits; requires --page-four-reviewed.")
    parser.add_argument("--page-six-reviewed", action="store_true", help="Apply approved payer page edits; requires --page-five-reviewed.")
    parser.add_argument("--page-seven-reviewed", action="store_true", help="Apply approved limitations page edits; requires --page-six-reviewed.")
    parser.add_argument("--review-complete", action="store_true", help="Render the author-approved edition of the specific reviewed audit; applies all approved page edits.")
    args = parser.parse_args(argv)
    started = perf_counter()
    try:
        if args.scenario and not args.scenario.replace("_", "").isalnum():
            raise ValueError("Invalid scenario ID.")
        findings, evidence, trace_id, source = load_saved(args.directory, args.scenario)
        metadata = json.loads(args.model_metadata.read_text(encoding="utf-8")) if args.model_metadata else None
        receipt = render_report(findings, evidence, args.output, run_id=args.directory.name,
                                trace_id=trace_id, model_metadata=metadata, source_provenance=source,
                                page_one_reviewed=args.page_one_reviewed or args.review_complete,
                                page_two_reviewed=args.page_two_reviewed or args.review_complete,
                                page_three_reviewed=args.page_three_reviewed or args.review_complete,
                                page_four_reviewed=args.page_four_reviewed or args.review_complete,
                                page_five_reviewed=args.page_five_reviewed or args.review_complete,
                                page_six_reviewed=args.page_six_reviewed or args.review_complete,
                                page_seven_reviewed=args.page_seven_reviewed or args.review_complete,
                                review_complete=args.review_complete)
        review_path = args.output.with_suffix(".review.json")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review.update(source)
        review["render_seconds"] = round(perf_counter() - started, 3)
        review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f"Report unavailable ({type(error).__name__}); check completed source artifacts and output path.")
        return 2
    print(f"{'Reviewed' if args.review_complete else 'Draft'} PDF: {receipt.pdf_path}")
    print(f"Render seconds: {perf_counter() - started:.3f}")
    print("Author editorial and visual review completed; not clinical certification." if args.review_complete else
          "Review in progress; pages 1-7 approved." if args.page_seven_reviewed else
          "Review in progress; pages 1-6 approved." if args.page_six_reviewed else
          "Review in progress; pages 1-5 approved." if args.page_five_reviewed else
          "Review in progress; pages 1-4 approved." if args.page_four_reviewed else
          "Review in progress; pages 1-3 approved." if args.page_three_reviewed else
          "Review in progress; pages 1-2 approved." if args.page_two_reviewed else
          "Review in progress; page 1 approved." if args.page_one_reviewed else "Human review pending.")
    print("Source audit artifacts unchanged; no provider requests made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
