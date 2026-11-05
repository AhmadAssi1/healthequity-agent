from math import isfinite
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas import ColumnMapping
from app.validation import validate_dataset


class SuppressionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    min_group_size: int = Field(default=50, ge=1)
    min_positive_outcomes: int = Field(default=10, ge=0)

class SeverityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    watch_gap: float = Field(default=0.05, ge=0, le=1)
    substantial_gap: float = Field(default=0.10, ge=0, le=1)

    @model_validator(mode="after")
    def validate_threshold_order(self):
        if self.watch_gap >= self.substantial_gap:
            raise ValueError(
                "watch_gap must be smaller than substantial_gap."
            )

        return self



DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

def load_severity_policy(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> SeverityPolicy:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Configuration must be a mapping.")

    settings = config.get("severity")

    if not isinstance(settings, dict):
        raise TypeError(
            "Configuration must contain a 'severity' mapping."
        )

    return SeverityPolicy.model_validate(settings)

def load_suppression_policy(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> SuppressionPolicy:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Configuration must be a mapping.")

    settings = config.get("fairness")

    if not isinstance(settings, dict):
        raise TypeError("Configuration must contain a 'fairness' mapping.")

    return SuppressionPolicy.model_validate(settings)


def check_group_eligibility(
    count: int,
    positive_outcomes: int,
    policy: SuppressionPolicy,
) -> dict:

    for value in (count, positive_outcomes):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
        ):
            raise TypeError("Group counts must be integers.")
    if count < 0 or not 0 <= positive_outcomes <= count:
        raise ValueError(
            "Counts must satisfy 0 <= positive_outcomes <= count."
        )

    reasons = []

    if count < policy.min_group_size:
        reasons.append(
            f"Group size {count} is below minimum "
            f"{policy.min_group_size}."
        )

    if positive_outcomes < policy.min_positive_outcomes:
        reasons.append(
            f"Positive outcomes {positive_outcomes} are below minimum "
            f"{policy.min_positive_outcomes}."
        )

    return {
        "reportable": len(reasons) == 0,
        "suppression_reason": " ".join(reasons) if reasons else None,
    }


def check_group_sizes(
    df: pd.DataFrame,
    group_column: str,
    policy: SuppressionPolicy | None = None,
) -> pd.DataFrame:
    if policy is None:
        policy = load_suppression_policy()

    mapping = ColumnMapping(
        outcome="outcome",
        groups=[group_column],
    )
    validate_dataset(df, mapping)

    codes, labels = pd.factorize(df[group_column], sort=False)

    working = pd.DataFrame({
        "group_code": codes,
        "outcome": df["outcome"].to_numpy(),
    })

    summary = working.groupby("group_code")["outcome"].agg(
        count="size",
        positive_outcomes="sum",
    ).astype("int64")

    decisions = [
        check_group_eligibility(
            count=int(row.count),
            positive_outcomes=int(row.positive_outcomes),
            policy=policy,
        )
        for row in summary.itertuples()
    ]

    summary["reportable"] = [
        decision["reportable"] for decision in decisions
    ]
    summary["suppression_reason"] = [
        decision["suppression_reason"] for decision in decisions
    ]

    summary.index = pd.Index(
        [None if code == -1 else labels[code] for code in summary.index],
        name=group_column,
        tupleize_cols=False,
        dtype=object,
    )

    return summary


def classify_severity(
    gap: float | None,
    ci_lower: float | None,
    ci_upper: float | None,
    policy: SeverityPolicy | None = None,
) -> str:
    if policy is None:
        policy = load_severity_policy()

    values = (gap, ci_lower, ci_upper)

    if any(value is None or pd.isna(value) for value in values):
        return "unavailable"

    if not all(isfinite(value) for value in values):
        raise ValueError("Gap and confidence bounds must be finite.")

    if not 0 <= gap <= 1:
        raise ValueError("Gap must be between 0 and 1.")

    if not 0 <= ci_lower <= ci_upper <= 1:
        raise ValueError(
            "Confidence bounds must satisfy 0 <= lower <= upper <= 1."
        )

    if gap > policy.substantial_gap and ci_lower > 0:
        return "substantial"

    if gap > policy.watch_gap:
        return "watch"

    return "no_evidence_found"


def filter_reportable_groups(
    df: pd.DataFrame,
    group_column: str,
    policy: SuppressionPolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligibility = check_group_sizes(
        df,
        group_column,
        policy,
    )

    eligible_labels = eligibility.index[
        eligibility["reportable"]
    ]

    mask = df[group_column].isin(
        eligible_labels.dropna()
    )

    if eligible_labels.isna().any():
        mask = mask | df[group_column].isna()

    filtered = df.loc[mask].copy()

    return filtered, eligibility