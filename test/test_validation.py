import pandas as pd
import pytest

from app.schemas import ColumnMapping
from app.validation import create_dataset_fingerprint, validate_dataset


def test_empty_dataset_rejected():
    df = pd.DataFrame()
    mapping = ColumnMapping(outcome="actual", groups=["gender"])

    with pytest.raises(ValueError, match="at least one data row"):
        validate_dataset(df, mapping)


def test_missing_mapped_column_rejected():
    df = pd.DataFrame({"actual": [1]})
    mapping = ColumnMapping(outcome="actual", groups=["gender"])

    with pytest.raises(ValueError, match="Missing mapped columns"):
        validate_dataset(df, mapping)

def test_duplicate_column_names_rejected():
    df = pd.DataFrame(
        [[1, 0, "Female"]],
        columns=["actual", "actual", "gender"],
    )
    mapping = ColumnMapping(outcome="actual", groups=["gender"])

    with pytest.raises(ValueError, match="Column names must be unique"):
        validate_dataset(df, mapping)


@pytest.mark.parametrize("column", ["actual", "predicted"])
def test_missing_labels_rejected(column):
    df = pd.DataFrame({
        "actual": [1, 0],
        "predicted": [1, 0],
        "gender": ["Female", "Male"],
    })
    df.loc[0, column] = float("nan")

    mapping = ColumnMapping(
        outcome="actual",
        prediction="predicted",
        groups=["gender"],
    )

    with pytest.raises(
        ValueError,
        match=f"Column '{column}' must not contain missing values",
    ):
        validate_dataset(df, mapping)

@pytest.mark.parametrize("column", ["actual", "predicted"])
@pytest.mark.parametrize(
    "invalid_values",
    [
        [0, 2],
        [0, 0.5],
        ["0", "1"],
        [False, True],
    ],
)
def test_invalid_binary_labels_rejected(column, invalid_values):
    df = pd.DataFrame({
        "actual": [1, 0],
        "predicted": [1, 0],
        "gender": ["Female", "Male"],
    })
    df[column] = invalid_values

    mapping = ColumnMapping(
        outcome="actual",
        prediction="predicted",
        groups=["gender"],
    )

    with pytest.raises(ValueError, match=f"Column '{column}'"):
        validate_dataset(df, mapping)

def test_valid_dataset_accepted():
    df = pd.DataFrame({
        "actual": [1.0, 0.0],
        "predicted": [1, 0],
        "gender": ["Female", "Male"],
    })
    mapping = ColumnMapping(
        outcome="actual",
        prediction="predicted",
        groups=["gender"],
    )

    validate_dataset(df, mapping)


@pytest.mark.parametrize(
    "scores",
    [
        [0.5, None],
        [-0.1, 0.5],
        [0.5, 1.1],
        [float("inf"), 0.5],
        [float("-inf"), 0.5],
        ["0.2", "0.8"],
        [False, True],
    ],
)
def test_invalid_scores_rejected(scores):
    df = pd.DataFrame({
        "actual": [1, 0],
        "predicted": [1, 0],
        "gender": ["Female", "Male"],
        "probability": scores,
    })
    mapping = ColumnMapping(
        outcome="actual",
        prediction="predicted",
        groups=["gender"],
        score="probability",
    )

    with pytest.raises(ValueError, match="Score"):
        validate_dataset(df, mapping)


@pytest.mark.parametrize("scores", [[0.0, 1.0], [0.25, 0.85]])
def test_valid_scores_accepted(scores):
    df = pd.DataFrame({
        "actual": [1, 0],
        "predicted": [1, 0],
        "gender": ["Female", "Male"],
        "probability": scores,
    })
    mapping = ColumnMapping(
        outcome="actual",
        prediction="predicted",
        groups=["gender"],
        score="probability",
    )

    validate_dataset(df, mapping)

def test_dataset_fingerprint():
    df = pd.DataFrame({
        "actual": [1, 0],
        "gender": ["Female", "Male"],
    })

    fingerprint = create_dataset_fingerprint(df)
    repeated = create_dataset_fingerprint(df.copy())
    reordered = create_dataset_fingerprint(df[["gender", "actual"]])

    assert fingerprint.row_count == 2
    assert len(fingerprint.columns_hash) == 64
    assert fingerprint == repeated
    assert fingerprint.columns_hash != reordered.columns_hash