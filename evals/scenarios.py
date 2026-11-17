"""Reproducible inputs and independent count-based evaluation expectations.

This module does not call the agent, fairness engine, or planning guardrail to
derive expected answers. CIs, severity, narrative scoring, and the evaluation
runner belong to the subsequent evaluation step.
"""

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.schemas import ColumnMapping
from fairness.scenarios import make_scenario

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/samples/diabetes_audit_sample.csv"
MANIFEST = Path(__file__).with_name("diabetes_slice_answers.json")
RATES = ("selection_rate", "tpr", "fpr", "ppv")


def source_rows_hash(header: list[str], rows: list[list[str]]) -> str:
    """Hash original CSV tokens, without platform-dependent float formatting."""
    canonical = json.dumps([header, rows], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class Scenario:
    """A fresh dataframe, mapping, and trusted expectations for one case."""

    name: str
    description: str
    data: pd.DataFrame
    mapping: ColumnMapping
    expected: dict


def count_answers(counts: dict[str, dict[str, int]]) -> dict:
    """Elementary arithmetic oracle from predeclared confusion counts.

    Default suppression is fixed at n >= 50 and positives >= 10. Keeping
    evaluation policy explicit prevents config changes silently moving targets.
    Test expectations include each rate's method, not a combined EOD p-value.
    """
    groups, eligible, suppressed = {}, [], []
    for label, cells in counts.items():
        tn, fp, fn, tp = (cells[key] for key in ("tn", "fp", "fn", "tp"))
        n = tn + fp + fn + tp
        ratios = {
            "selection_rate": (tp + fp, n),
            "tpr": (tp, tp + fn),
            "fpr": (fp, fp + tn),
            "ppv": (tp, tp + fp),
        }
        reportable = n >= 50 and tp + fn >= 10
        (eligible if reportable else suppressed).append(label)
        groups[label] = {
            "count": n,
            "positive_outcomes": tp + fn,
            "confusion": cells.copy(),
            "rates": {key: a / b if b else None for key, (a, b) in ratios.items()},
        }
    gaps, tests = {}, {}
    for rate in RATES:
        values = [groups[label]["rates"][rate] for label in eligible]
        available = len(values) >= 2 and None not in values
        gaps[rate] = max(values) - min(values) if available else None
        if not available:
            tests[rate] = "unavailable"
            continue
        table = []
        for label in eligible:
            c = counts[label]
            a, b = {
                "selection_rate": (c["tp"] + c["fp"], groups[label]["count"]),
                "tpr": (c["tp"], c["tp"] + c["fn"]),
                "fpr": (c["fp"], c["fp"] + c["tn"]),
                "ppv": (c["tp"], c["tp"] + c["fp"]),
            }[rate]
            table.append((a, b - a))
        successes = sum(a for a, _ in table)
        failures = sum(b for _, b in table)
        if not successes or not failures:
            tests[rate] = "unavailable"
            continue
        total = successes + failures
        minimum = min(
            (a + b) * col / total
            for a, b in table for col in (successes, failures)
        )
        if minimum < 5:
            tests[rate] = "fisher_exact" if len(table) == 2 else "fisher_monte_carlo"
        else:
            tests[rate] = "two_proportion_z" if len(table) == 2 else "chi_square"
    eod = (
        max(gaps["tpr"], gaps["fpr"])
        if gaps["tpr"] is not None and gaps["fpr"] is not None else None
    )
    return {
        "groups": groups,
        "eligible_groups": eligible,
        "suppressed_groups": suppressed,
        "gaps": {
            "demographic_parity_difference": gaps["selection_rate"],
            "equalized_odds_difference": eod,
            "ppv_difference": gaps["ppv"],
        },
        "rate_tests": tests,
    }


def _synthetic(name, description, counts, *, score=True):
    data = make_scenario(counts, seed=42)
    if score:
        # Distinct valid scores avoid accidental duplicate warnings in most
        # fixtures. These are fixture probabilities, not calibrated model scores.
        data["score"] = (
            0.1 + 0.5 * data["prediction"]
            + np.arange(len(data)) / (10 * len(data))
        )
    return Scenario(
        name, description, data,
        ColumnMapping(
            outcome="outcome", prediction="prediction", groups=["group"],
            score="score" if score else None,
        ),
        {
            "status": "audit", "row_count": len(data),
            "score_available": score, "duplicate_rows": 0 if score else None,
            "attributes": {"group": count_answers(counts)},
            "policy": {"min_group_size": 50, "min_positive_outcomes": 10},
        },
    )


def build_scenarios(source_path: str | Path = SOURCE) -> list[Scenario]:
    """Build exactly twelve cases; fail explicitly if the pinned demo changes.

    Mislabelled predictions use text labels, a detectable contract violation.
    A semantically reversed but binary prediction cannot be inferred from values.
    """
    base = {"tn": 50, "fp": 50, "fn": 50, "tp": 50}
    same = {"A": base, "B": base}
    cases = [
        _synthetic("no_gap", "Identical group confusion counts.", same),
        _synthetic("selection_gap", "Selection gap 0.20; other gaps may coexist.", {
            "A": base, "B": {"tn": 30, "fp": 70, "fn": 30, "tp": 70},
        }),
        _synthetic("tpr_gap", "TPR gap 0.20; other gaps may coexist.", {
            "A": base, "B": {"tn": 50, "fp": 50, "fn": 30, "tp": 70},
        }),
        _synthetic("ppv_gap", "PPV gap 0.20; other gaps may coexist.", {
            "A": base, "B": {"tn": 70, "fp": 30, "fn": 30, "tp": 70},
        }),
        _synthetic("tiny_group", "Suppress Tiny and LowPositive, retain A/B.", {
            **same,
            "Tiny": {"tn": 4, "fp": 4, "fn": 6, "tp": 6},
            "LowPositive": {"tn": 30, "fp": 25, "fn": 2, "tp": 3},
        }),
        _synthetic("multiple_groups", "Three eligible groups; chi-square tests.", {
            **same, "C": {"tn": 70, "fp": 30, "fn": 30, "tp": 70},
        }),
        _synthetic("missing_score", "Audit labels; no score-based analyses.", same, score=False),
        _synthetic("missing_outcome", "Refuse absent mapped outcome.", same),
        _synthetic("mislabelled_prediction", "Refuse text prediction labels.", same),
        _synthetic("duplicated_rows", "Warn and retain all duplicated rows.", same),
    ]
    missing, mislabelled, duplicated = cases[7:10]
    missing.data = missing.data.drop(columns="outcome")
    missing.expected = {
        "status": "refuse", "error_contains": "Missing mapped columns",
        "provider_calls": 0,
    }
    mislabelled.data["prediction"] = mislabelled.data["prediction"].map({0: "NO", 1: "YES"})
    mislabelled.expected = {
        "status": "refuse", "error_contains": "numeric binary labels",
        "provider_calls": 0,
    }
    duplicated.data = pd.concat([duplicated.data, duplicated.data], ignore_index=True)
    duplicated.expected.update(row_count=800, duplicate_rows=400)
    duplicated.expected["attributes"]["group"] = count_answers({
        label: {key: 2 * value for key, value in base.items()} for label in same
    })
    # Without scores, repeated label/group combinations are legitimate but still
    # trigger the current profiler's duplicate warning. Do not hide this overlap.
    cases[6].expected["duplicate_rows"] = 392

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_path = Path(source_path)
    source_bytes = source_path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source_bytes).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Diabetes evaluation source hash mismatch; review fixture provenance.")
    raw_rows = list(csv.reader(io.StringIO(source_bytes.decode("utf-8"))))
    header, records = raw_rows[0], raw_rows[1:]
    source = pd.read_csv(io.BytesIO(source_bytes), keep_default_na=False, float_precision="round_trip")
    for spec in manifest["slices"]:
        indices = spec["source_row_indices"]
        if (len(indices) != spec["row_count"] or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < len(records) for i in indices)):
            raise ValueError("Invalid frozen Diabetes source row indices.")
        if source_rows_hash(header, [records[i] for i in indices]) != spec["source_rows_sha256"]:
            raise ValueError("Diabetes slice hash mismatch; review selected source rows.")
        data = source.iloc[indices].reset_index(drop=True)
        if not data["gender"].eq(spec["gender"]).all():
            raise ValueError("Frozen Diabetes slice does not match its declared gender.")
        cases.append(Scenario(
            spec["name"], f"Fixed 1,000-stay {spec['gender']} Diabetes slice; audit race.",
            data,
            ColumnMapping(outcome="outcome", prediction="prediction", groups=["race"], score="score"),
            {
                "status": "audit", "row_count": spec["row_count"],
                "score_available": True, "duplicate_rows": spec["duplicate_rows"],
                "attributes": {"race": count_answers(spec["confusion_counts"])},
                "source_sha256": manifest["source_sha256"],
                "policy": {"min_group_size": 50, "min_positive_outcomes": 10},
            },
        ))
    return cases
