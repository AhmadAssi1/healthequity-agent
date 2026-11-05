import numpy as np
import pandas as pd


def make_group(
    name: str,
    *,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> pd.DataFrame:
    counts = (tn, fp, fn, tp)

    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        for value in counts
    ):
        raise TypeError("Confusion counts must be integers.")

    if any(value < 0 for value in counts):
        raise ValueError("Confusion counts must be nonnegative.")

    if sum(counts) == 0:
        raise ValueError("A group must contain at least one row.")

    pairs = np.repeat(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        repeats=counts,
        axis=0,
    )

    return pd.DataFrame({
        "outcome": pairs[:, 0],
        "prediction": pairs[:, 1],
        "group": name,
    })


def make_scenario(
    groups: dict[str, dict[str, int]],
    *,
    seed: int = 42,
) -> pd.DataFrame:
    if not groups:
        raise ValueError("Provide at least one group.")

    frames = [
        make_group(name, **counts)
        for name, counts in groups.items()
    ]

    df = pd.concat(frames, ignore_index=True)

    return df.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

def make_gap_scenario(
    metric: str,
    gap: float,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    if metric not in {"selection_rate", "tpr", "ppv"}:
        raise ValueError(
            "metric must be 'selection_rate', 'tpr', or 'ppv'."
        )

    if not np.isfinite(gap) or not 0 <= gap <= 0.5:
        raise ValueError("gap must be between 0 and 0.5.")

    shift = round(gap * 100)

    if not np.isclose(gap * 100, shift, rtol=0, atol=1e-9):
        raise ValueError("gap must be a multiple of 0.01.")

    group_a = {"tn": 50, "fp": 50, "fn": 50, "tp": 50}
    group_b = group_a.copy()

    if metric == "selection_rate":
        group_b["tn"] -= shift
        group_b["fp"] += shift
        group_b["fn"] -= shift
        group_b["tp"] += shift

    elif metric == "tpr":
        group_b["fn"] -= shift
        group_b["tp"] += shift

    else:  # PPV
        group_b["fp"] -= shift
        group_b["tp"] += shift
        group_b["fn"] -= shift
        group_b["tn"] += shift

    return make_scenario(
        {"A": group_a, "B": group_b},
        seed=seed,
    )