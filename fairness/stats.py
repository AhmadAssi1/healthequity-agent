"""Significance tests and bootstrap intervals for binary fairness audits."""

from dataclasses import dataclass
from math import sqrt
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import MonteCarloMethod, chi2_contingency, fisher_exact, norm

from app.schemas import ColumnMapping
from app.validation import validate_dataset


@dataclass(frozen=True)
class ProportionTestResult:
    method: str
    statistic: float | None
    p_value: float | None
    min_expected_count: float
    reason: str | None = None


@dataclass(frozen=True)
class BootstrapInterval:
    observed: float | None
    lower: float | None
    upper: float | None
    valid_resamples: int
    total_resamples: int


def _count_table(positive_counts, totals):
    positive_counts, totals = list(positive_counts), list(totals)
    if len(positive_counts) != len(totals) or len(totals) < 2:
        raise ValueError("Provide matching counts and totals for at least two groups.")
    for positives, total in zip(positive_counts, totals):
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in (positives, total)
        ):
            raise ValueError("Group counts and totals must be integers.")
        if total <= 0 or not 0 <= positives <= total:
            raise ValueError("Each group needs a positive total and a valid positive count.")
    return np.column_stack((positive_counts, np.subtract(totals, positive_counts)))


def _z_test(table):
    totals = table.sum(axis=1)
    pooled = table[:, 0].sum() / totals.sum()
    if not 0 < pooled < 1:
        raise ValueError("The z-test requires both outcome categories.")
    standard_error = sqrt(pooled * (1 - pooled) * np.sum(1 / totals))
    rates = table[:, 0] / totals
    z = float((rates[0] - rates[1]) / standard_error)
    return z, float(2 * norm.sf(abs(z)))


def two_proportion_z_test(x_a, n_a, x_b, n_b) -> tuple[float, float]:
    """Return the signed pooled z statistic and the two-sided p-value."""
    return _z_test(_count_table([x_a, x_b], [n_a, n_b]))


def compare_proportions(
    positive_counts, totals, *, n_resamples: int = 10_000, seed: int = 42,
) -> ProportionTestResult:
    """Test equal rates across independent groups using aggregate counts.

    Pass only eligible groups. For TPR, the denominator is actual positives;
    for FPR it is actual negatives, and for PPV it is predicted positives.
    If an expected cell count is below 5, use Fisher's fixed-margin null.
    Its p-value is exact for 2-by-2 tables and estimated by Monte Carlo for
    larger tables. Otherwise use a pooled z-test or Pearson chi-square.
    """
    table = _count_table(positive_counts, totals)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    minimum = float(expected.min())
    if (table.sum(axis=0) == 0).any():
        return ProportionTestResult(
            "unavailable", None, None, minimum,
            "The table needs both positive and negative observations.",
        )
    if minimum < 5:
        if len(table) == 2:
            p_value = fisher_exact(table, alternative="two-sided").pvalue
            return ProportionTestResult("fisher_exact", None, float(p_value), minimum)
        method = MonteCarloMethod(n_resamples=n_resamples, rng=np.random.default_rng(seed))
        p_value = fisher_exact(table, method=method).pvalue
        return ProportionTestResult(
            "fisher_monte_carlo", None, float(np.asarray(p_value).item()), minimum,
        )
    if len(table) == 2:
        statistic, p_value = _z_test(table)
        return ProportionTestResult("two_proportion_z", statistic, p_value, minimum)
    result = chi2_contingency(table, correction=False)
    return ProportionTestResult(
        "chi_square", float(result.statistic), float(result.pvalue), minimum,
    )


def _rate(numerator, denominator):
    return np.divide(
        numerator, denominator,
        out=np.full(numerator.shape, np.nan),
        where=denominator > 0,
    )


def _fairness_gaps(confusion_counts):
    # The last axis stores counts for (outcome, prediction): 00, 01, 10, 11.
    tn, fp, fn, tp = np.moveaxis(confusion_counts, -1, 0)
    selection = _rate(tp + fp, tp + fp + tn + fn)
    tpr = _rate(tp, tp + fn)
    fpr = _rate(fp, fp + tn)
    ppv = _rate(tp, tp + fp)
    return {
        "demographic_parity_difference": np.ptp(selection, axis=-1),
        "equalized_odds_difference": np.maximum(
            np.ptp(tpr, axis=-1), np.ptp(fpr, axis=-1),
        ),
        "ppv_difference": np.ptp(ppv, axis=-1),
    }


def bootstrap_fairness_intervals(
    df: pd.DataFrame,
    group_column: str,
    *,
    n_resamples: int = 1_000,
    seed: int = 42,
) -> dict[str, BootstrapInterval]:
    """Return percentile 95% intervals for the three headline gaps.

    Pass only eligible groups. Resampling preserves group size and paired
    outcome/prediction labels. Sampling confusion counts from a multinomial
    is equivalent to resampling individual rows within each group for these
    metrics. Undefined draws are counted and excluded per metric. When the
    observed gap is undefined, no interval is reported. Assumes independent
    rows; repeated stays may require a patient-level bootstrap.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, Integral) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    mapping = ColumnMapping(
        outcome="outcome", prediction="prediction", groups=[group_column],
    )
    validate_dataset(df, mapping)
    codes, labels = pd.factorize(df[group_column], sort=False, use_na_sentinel=False)
    if len(labels) < 2:
        raise ValueError("At least two eligible groups are required.")

    outcome = df["outcome"].to_numpy(dtype=int)
    prediction = df["prediction"].to_numpy(dtype=int)
    counts = np.zeros((len(labels), 4), dtype=int)
    np.add.at(counts, (codes, 2 * outcome + prediction), 1)

    rng = np.random.default_rng(seed)
    resamples = np.stack([
        rng.multinomial(group.sum(), group / group.sum(), size=n_resamples)
        for group in counts
    ], axis=1)
    observed = _fairness_gaps(counts)
    intervals = {}
    for metric, values in _fairness_gaps(resamples).items():
        valid = values[np.isfinite(values)]
        point = float(observed[metric]) if np.isfinite(observed[metric]) else None
        lower = upper = None
        if point is not None and len(valid):
            lower, upper = map(float, np.percentile(valid, [2.5, 97.5]))
        intervals[metric] = BootstrapInterval(point, lower, upper, len(valid), n_resamples)
    return intervals
