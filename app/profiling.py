"""Deterministic dataset profiling."""

import numpy as np
import pandas as pd

from app.schemas import ColumnMapping
from app.tool_schemas import AttributeProfile, DatasetProfile, GroupCounts
from app.validation import validate_dataset


def _profile_attribute(
    df: pd.DataFrame,
    attribute: str,
    outcome_column: str,
) -> AttributeProfile:
    """Summarize one attribute after dataset validation."""
    counts = {}

    for label, outcome in zip(
        df[attribute],
        df[outcome_column],
        strict=True,
    ):
        if not pd.api.types.is_scalar(label):
            raise ValueError(
                "Group values must be scalar categories."
            )

        if pd.isna(label):
            label = None
        elif isinstance(label, np.generic):
            label = label.item()

        if label is not None and not isinstance(
            label, (str, int, float, bool)
        ):
            raise ValueError(
                "Group values must be text, numeric, Boolean, or missing."
            )

        identity = (type(label), label)

        if identity not in counts:
            counts[identity] = {
                "group_value": label,
                "count": 0,
                "positive_outcomes": 0,
            }

        counts[identity]["count"] += 1
        counts[identity]["positive_outcomes"] += int(outcome)

    return AttributeProfile(
        attribute=attribute,
        groups=[
            GroupCounts(**values)
            for values in counts.values()
        ],
    )

def profile_dataset(
    df: pd.DataFrame,
    mapping: ColumnMapping,
) -> DatasetProfile:
    """Build an internal profile without modifying the dataset."""
    validate_dataset(df, mapping)

    row_count = len(df)
    positive_outcomes = int(df[mapping.outcome].sum())
    flags = []

    audit_columns = [mapping.outcome, *mapping.groups]

    if mapping.prediction is not None:
        audit_columns.append(mapping.prediction)
    else:
        flags.append(
            "Predictions are not mapped; fairness analysis requires predictions."
        )

    if mapping.score is not None:
        audit_columns.append(mapping.score)

    audit_columns = list(dict.fromkeys(audit_columns))

    if df[audit_columns].duplicated().any():
        flags.append(
            "Duplicate rows exist in the mapped audit columns; "
            "no rows were removed."
        )

    if positive_outcomes in (0, row_count):
        flags.append(
            "The outcome column contains only one class."
        )

    attributes = []

    for attribute in mapping.groups:
        if df[attribute].isna().any():
            flags.append(
                f"Missing values in '{attribute}' form a separate group."
            )

        attributes.append(
            _profile_attribute(
                df=df,
                attribute=attribute,
                outcome_column=mapping.outcome,
            )
        )

    return DatasetProfile(
        row_count=row_count,
        column_count=len(df.columns),
        positive_outcomes=positive_outcomes,
        outcome_prevalence=positive_outcomes / row_count,
        score_available=mapping.score is not None,
        attributes=attributes,
        data_quality_flags=flags,
    )