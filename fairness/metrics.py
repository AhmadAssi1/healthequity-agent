"""Deterministic per-group metrics and fairness comparisons.

These functions produce raw calculations, not reportable audit findings.
Apply suppression before comparing eligible groups. Statistical uncertainty,
severity policies and narrative reporting are implemented separately.
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import precision_score, roc_auc_score

from app.schemas import ColumnMapping
from app.validation import validate_dataset

_AUTO_REFERENCE = object()
_RATE_COLUMNS = ("base_rate", "selection_rate", "tpr", "fpr", "ppv")
_UNDEFINED_REASONS = {
    "tpr": "No positive outcomes (TP + FN = 0).",
    "fpr": "No negative outcomes (FP + TN = 0).",
    "ppv": "No positive predictions (TP + FP = 0).",
    "auroc": "Ground truth contains only one class.",
}


def base_rate(y_true, y_pred):
    """Fraction of observations with a positive ground-truth outcome."""
    return np.mean(y_true)


def safe_tpr(y_true, y_pred):
    """Return NaN when there are no positive outcomes (TP + FN = 0)."""
    if not np.any(np.asarray(y_true) == 1):
        return np.nan
    return true_positive_rate(y_true, y_pred)


def safe_fpr(y_true, y_pred):
    """Return NaN when there are no negative outcomes (FP + TN = 0)."""
    if not np.any(np.asarray(y_true) == 0):
        return np.nan
    return false_positive_rate(y_true, y_pred)


def safe_ppv(y_true, y_pred):
    """Return NaN when there are no positive predictions (TP + FP = 0)."""
    if not np.any(np.asarray(y_pred) == 1):
        return np.nan
    return precision_score(y_true, y_pred)


def safe_auroc(y_true, y_score):
    """Return NaN when ground truth contains only one class."""
    if np.unique(y_true).size < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def compute_group_metrics(
    df: pd.DataFrame,
    group_column: str,
    score_column: str | None = None,
) -> pd.DataFrame:
    """Compute metrics for observed groups in a canonical audit dataset.

    Outcome and prediction columns must be named ``outcome`` and ``prediction``.
    Invalid labels/scores raise validation errors instead of dropping rows.
    A missing group is included with index label None, distinct from a literal
    group named "Missing". Only observed categories are included.

    Undefined metrics are NaN; their denominator conditions are documented in
    the safe metric functions. These raw results are not a report: statistical
    uncertainty and suppression are separate work. ``compute_fairness_gaps``
    adds gap calculations, reference comparisons and unavailability reasons.
    The input dataframe is not modified.
    """
    mapping = ColumnMapping(
        outcome="outcome",
        prediction="prediction",
        groups=[group_column],
        score=score_column,
    )
    validate_dataset(df, mapping)

    # Integer codes keep missing values and mixed category types compatible
    # with Fairlearn while preserving the original group labels in the result.
    group_codes, group_labels = pd.factorize(df[group_column], sort=False)
    frame = MetricFrame(
        metrics={
            "count": count,
            "base_rate": base_rate,
            "selection_rate": selection_rate,
            "tpr": safe_tpr,
            "fpr": safe_fpr,
            "ppv": safe_ppv,
        },
        y_true=df["outcome"],
        y_pred=df["prediction"],
        sensitive_features=group_codes,
    )
    result = frame.by_group.copy()

    if score_column is not None:
        auc_frame = MetricFrame(
            metrics={"auroc": safe_auroc},
            y_true=df["outcome"],
            y_pred=df[score_column],
            sensitive_features=group_codes,
        )
        result["auroc"] = auc_frame.by_group["auroc"]

    result.index = pd.Index(
        [None if code == -1 else group_labels[code] for code in result.index],
        name=group_column,
        tupleize_cols=False,
        dtype=object if -1 in result.index else None,
    )
    return result


def _validate_group_metrics(group_metrics: pd.DataFrame) -> None:
    """Check the aggregate-table contract before deriving comparisons."""
    if not group_metrics.columns.is_unique or not group_metrics.index.is_unique:
        raise ValueError("Group labels and metric column names must be unique.")
    if group_metrics.index.isna().sum() > 1:
        raise ValueError("Only one missing-value group is allowed.")
    required = ["count", *_RATE_COLUMNS]
    missing = [column for column in required if column not in group_metrics]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")
    for column in required + (["auroc"] if "auroc" in group_metrics else []):
        values = group_metrics[column]
        if (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
        ):
            raise ValueError(f"Metric '{column}' must be numeric.")
        if column == "count":
            if not (
                np.isfinite(values) & (values > 0) & (values % 1 == 0)
            ).all():
                raise ValueError("Group counts must be positive finite integers.")
        elif not (values.isna() | values.between(0, 1)).all():
            raise ValueError(f"Metric '{column}' must be in [0, 1] or NaN.")


def _pairwise_gap(group_metrics: pd.DataFrame, column: str) -> dict:
    """Require all supplied groups to have a defined rate; never skip NaNs."""
    unavailable = {
        "value": np.nan,
        "min_group": None,
        "max_group": None,
        "reason": None,
    }
    if len(group_metrics) < 2:
        return {
            **unavailable,
            "reason": "At least two eligible groups are required.",
        }
    values = group_metrics[column]
    if values.isna().any():
        return {
            **unavailable,
            "reason": f"'{column}' is undefined for at least one eligible group.",
        }
    # Positional lookup preserves labels such as None and numeric category codes.
    low = int(np.argmin(values.to_numpy()))
    high = int(np.argmax(values.to_numpy()))
    return {
        "value": float(values.iloc[high] - values.iloc[low]),
        "min_group": group_metrics.index[low],
        "max_group": group_metrics.index[high],
        "reason": None,
    }


def compute_fairness_gaps(
    group_metrics: pd.DataFrame,
    reference_group=_AUTO_REFERENCE,
) -> dict:
    """Compare a table returned by ``compute_group_metrics``.

    Pass only policy-eligible groups when producing a report. This function
    does not decide eligibility and never silently removes groups with NaN.

    Returns a dict containing:
    - ``gaps``: value, minimum/maximum group labels and unavailable reason for
      DPD, TPR gap, FPR gap, EOD and PPV difference. Gaps are maximum pairwise
      differences (max - min), independent of the reference choice.
    - ``reference_group`` and ``reference_group_method``: an explicit reference,
      or the largest supplied group by default (first table row breaks ties).
      This convention does not imply clinical advantage. Explicit None selects
      the missing-value group. An empty table has no reference.
    - ``reference_differences``: signed rate(group) - rate(reference), including
      the reference row. Undefined operands produce NaN, even for that row.
    - ``undefined_metric_reasons``: a parallel table of reasons for NaN metrics.
    - ``disparate_impact_ratio``: min(selection rate) / max(selection rate),
      a supporting indicator; unavailable when the maximum is zero.

    EOD is max(TPR gap, FPR gap); it is unavailable if either component is
    unavailable. ``equalized_odds_component`` identifies the governing rate
    (TPR breaks ties). Extremal labels use the first row on ties. NaN is an
    internal unavailable value; report serializers must convert it to null.
    """
    _validate_group_metrics(group_metrics)

    if reference_group is _AUTO_REFERENCE:
        reference = (
            group_metrics.index[int(np.argmax(group_metrics["count"].to_numpy()))]
            if len(group_metrics)
            else None
        )
        reference_method = "largest_group" if len(group_metrics) else "unavailable"
    else:
        if reference_group not in group_metrics.index:
            raise ValueError("Reference group must be present among eligible groups.")
        reference = reference_group
        reference_method = "user_selected"

    rate_columns = list(_RATE_COLUMNS)
    if "auroc" in group_metrics:
        rate_columns.append("auroc")
    reasons = pd.DataFrame(
        None, index=group_metrics.index, columns=rate_columns, dtype=object,
    )
    for column, reason in _UNDEFINED_REASONS.items():
        if column in reasons:
            reasons.loc[group_metrics[column].isna(), column] = reason
    # Base and selection rates are defined for nonempty valid input groups.
    for column in ("base_rate", "selection_rate"):
        reasons.loc[group_metrics[column].isna(), column] = (
            f"'{column}' is unavailable in the supplied metric table."
        )

    comparisons = group_metrics[rate_columns].copy()
    if len(group_metrics):
        comparisons = comparisons.subtract(group_metrics.loc[reference, rate_columns])

    gaps = {
        "demographic_parity_difference": _pairwise_gap(group_metrics, "selection_rate"),
        "tpr_gap": _pairwise_gap(group_metrics, "tpr"),
        "fpr_gap": _pairwise_gap(group_metrics, "fpr"),
        "ppv_difference": _pairwise_gap(group_metrics, "ppv"),
    }
    tpr_gap, fpr_gap = gaps["tpr_gap"], gaps["fpr_gap"]
    component = None
    if tpr_gap["reason"] or fpr_gap["reason"]:
        gaps["equalized_odds_difference"] = {
            "value": np.nan,
            "min_group": None,
            "max_group": None,
            "reason": "Both TPR and FPR gaps must be available. "
            + " ".join(dict.fromkeys(
                gap["reason"] for gap in (tpr_gap, fpr_gap) if gap["reason"]
            )),
        }
    else:
        component = "tpr" if tpr_gap["value"] >= fpr_gap["value"] else "fpr"
        gaps["equalized_odds_difference"] = dict(gaps[f"{component}_gap"])

    dpd = gaps["demographic_parity_difference"]
    ratio = {"value": np.nan, "reason": dpd["reason"]}
    if dpd["reason"] is None:
        rates = group_metrics["selection_rate"]
        if rates.max() == 0:
            ratio["reason"] = "All selection rates are zero (ratio is 0/0)."
        else:
            ratio["value"] = float(rates.min() / rates.max())

    return {
        "reference_group": reference,
        "reference_group_method": reference_method,
        "gaps": pd.DataFrame.from_dict(gaps, orient="index").rename_axis("metric"),
        "equalized_odds_component": component,
        "reference_differences": comparisons,
        "undefined_metric_reasons": reasons,
        "disparate_impact_ratio": ratio,
    }
