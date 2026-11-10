"""Deterministic rules for selecting fairness analyses."""

import numpy as np

from app.tool_schemas import (
    AnalysisPlan,
    AttributeAnalysisCounts,
    AttributeAnalysisPlan,
    FairnessMetricPlan,
    GroupAnalysisCounts,
    RateTestPlan,
)


def _rate_counts(
    group: GroupAnalysisCounts,
    rate: str,
) -> tuple[int, int]:
    confusion = group.confusion

    counts = {
        "selection_rate": (
            confusion.tp + confusion.fp,
            group.count,
        ),
        "tpr": (
            confusion.tp,
            confusion.tp + confusion.fn,
        ),
        "fpr": (
            confusion.fp,
            confusion.fp + confusion.tn,
        ),
        "ppv": (
            confusion.tp,
            confusion.tp + confusion.fp,
        ),
    }

    if rate not in counts:
        raise ValueError(f"Unsupported rate: {rate}")

    return counts[rate]

def plan_rate_test(
    groups: list[GroupAnalysisCounts],
    rate: str,
) -> RateTestPlan:
    if len(groups) < 2:
        return RateTestPlan(
            rate=rate,
            method="unavailable",
            min_expected_count=None,
            reason="At least two eligible groups are required.",
        )

    counts = [_rate_counts(group, rate) for group in groups]

    if any(denominator == 0 for _, denominator in counts):
        return RateTestPlan(
            rate=rate,
            method="unavailable",
            min_expected_count=None,
            reason=(
                "The rate is undefined for at least one eligible group "
                "because its denominator is zero."
            ),
        )

    table = np.array(
        [
            [numerator, denominator - numerator]
            for numerator, denominator in counts
        ],
        dtype=np.int64,
    )

    expected = (
        np.outer(table.sum(axis=1), table.sum(axis=0))
        / table.sum()
    )
    minimum = float(expected.min())

    if (table.sum(axis=0) == 0).any():
        return RateTestPlan(
            rate=rate,
            method="unavailable",
            min_expected_count=minimum,
            reason=(
                "The comparison table requires both successes "
                "and failures."
            ),
        )

    if minimum < 5:
        method = (
            "fisher_exact"
            if len(groups) == 2
            else "fisher_monte_carlo"
        )
        reason = (
            "At least one expected cell count is below five; "
            "use the Fisher method appropriate for the table size."
        )
    else:
        method = (
            "two_proportion_z"
            if len(groups) == 2
            else "chi_square"
        )
        reason = (
            "All expected cell counts are at least five; "
            "use the asymptotic test appropriate for the group count."
        )

    return RateTestPlan(
        rate=rate,
        method=method,
        min_expected_count=minimum,
        reason=reason,
    )

def plan_fairness_metrics(
    groups: list[GroupAnalysisCounts],
) -> list[FairnessMetricPlan]:
    requirements = {
        "demographic_parity_difference": ("selection_rate",),
        "equalized_odds_difference": ("tpr", "fpr"),
        "ppv_difference": ("ppv",),
    }

    plans = []

    for metric, required_rates in requirements.items():
        if len(groups) < 2:
            available = False
            reason = "At least two eligible groups are required."
        else:
            undefined_rates = [
                rate
                for rate in required_rates
                if any(
                    _rate_counts(group, rate)[1] == 0
                    for group in groups
                )
            ]

            available = not undefined_rates

            if available:
                reason = (
                    "The required rates are defined "
                    "for every eligible group."
                )
            else:
                reason = (
                    "Undefined rates in at least one eligible group: "
                    + ", ".join(undefined_rates)
                    + "."
                )

        plans.append(
            FairnessMetricPlan(
                metric=metric,
                available=available,
                bootstrap=available,
                reason=reason,
            )
        )

    return plans

def plan_attribute_analyses(
    counts: AttributeAnalysisCounts,
    score_available: bool,
) -> AttributeAnalysisPlan:
    groups = counts.groups

    if not score_available:
        compute_auroc = False
        auroc_reason = "No score column was provided."
    elif not groups:
        compute_auroc = False
        auroc_reason = "No eligible groups are available."
    elif not any(
        0 < group.positive_outcomes < group.count
        for group in groups
    ):
        compute_auroc = False
        auroc_reason = (
            "AUROC is undefined because no eligible group "
            "contains both outcome classes."
        )
    else:
        compute_auroc = True
        auroc_reason = (
            "Scores are available and at least one eligible group "
            "contains both outcome classes."
        )

    return AttributeAnalysisPlan(
        attribute=counts.attribute,
        eligible_group_count=len(groups),
        fairness_metrics=plan_fairness_metrics(groups),
        statistical_tests=[
            plan_rate_test(groups, rate)
            for rate in ("selection_rate", "tpr", "fpr", "ppv")
        ],
        compute_auroc=compute_auroc,
        auroc_reason=auroc_reason,
    )

def select_analyses(
    counts: list[AttributeAnalysisCounts],
    score_available: bool,
) -> AnalysisPlan:
    return AnalysisPlan(
        score_available=score_available,
        attributes=[
            plan_attribute_analyses(
                counts=attribute,
                score_available=score_available,
            )
            for attribute in counts
        ],
    )