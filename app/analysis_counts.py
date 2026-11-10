"""Internal confusion counts for deterministic analysis selection."""

import numpy as np
import pandas as pd

from app.group_checks import check_group_sizes
from app.profiling import profile_dataset
from app.schemas import ColumnMapping
from app.tool_schemas import (
    AttributeAnalysisCounts,
    ConfusionCounts,
    GroupAnalysisCounts,
)
from fairness.policy import SuppressionPolicy


def compute_analysis_counts(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    policy: SuppressionPolicy | None = None,
) -> list[AttributeAnalysisCounts]:
    if mapping.prediction is None:
        raise ValueError(
            "A prediction column must be mapped before computing analysis counts."
        )

    profile = profile_dataset(df, mapping)
    eligibility = check_group_sizes(profile, policy)
    results = []

    for source, decisions in zip(
        profile.attributes,
        eligibility.attributes,
        strict=True,
    ):
        eligible_groups = [
            group
            for group, decision in zip(
                source.groups,
                decisions.groups,
                strict=True,
            )
            if decision.reportable
        ]

        counts = {
            (type(group.group_value), group.group_value): {
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
            }
            for group in eligible_groups
        }

        for label, outcome, prediction in zip(
            df[source.attribute],
            df[mapping.outcome],
            df[mapping.prediction],
            strict=True,
        ):
            if pd.isna(label):
                label = None
            elif isinstance(label, np.generic):
                label = label.item()

            identity = (type(label), label)

            if identity not in counts:
                continue

            cell = ("tn", "fp", "fn", "tp")[
                int(outcome) * 2 + int(prediction)
            ]
            counts[identity][cell] += 1

        results.append(
            AttributeAnalysisCounts(
                attribute=source.attribute,
                groups=[
                    GroupAnalysisCounts(
                        **group.model_dump(),
                        confusion=ConfusionCounts(
                            **counts[
                                (type(group.group_value), group.group_value)
                            ]
                        ),
                    )
                    for group in eligible_groups
                ],
            )
        )

    return results
