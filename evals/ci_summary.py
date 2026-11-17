"""Publish saved Markdown scorecards to CI logs and the GitHub job summary."""

import os
from pathlib import Path


def main():
    paths = sorted(Path("outputs/evaluations").glob("*/scorecard.md"))
    report = "\n\n".join(p.read_text(encoding="utf-8") for p in paths)
    if not report:
        report = "No evaluation scorecard was produced; inspect the failed configuration/probe step.\n"
    print(report)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write(report + "\n")


if __name__ == "__main__":
    main()
