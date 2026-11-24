"""Session-only LightGBM predictions for auditing a held-out subset."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.schemas import ColumnMapping
from app.validation import validate_dataset

DEMO_MODEL_NAME = "Session-trained LightGBM demo model"
DEMO_THRESHOLD = 0.5


def describe_demo_model(train_rows, test_rows):
    return (f"Demo model trained on this upload. LightGBM was fitted on {train_rows:,} rows. "
            f"The audit covers {test_rows:,} held-out rows.")


def train_demo_predictions(frame, mapping, features):
    """Fit on training rows and return only held-out outcomes, groups and predictions."""
    from lightgbm import LGBMClassifier
    from lightgbm.basic import LightGBMError

    if mapping.prediction is not None:
        raise ValueError("Demo training requires an unmapped prediction column.")
    validate_dataset(frame, mapping.model_copy(update={"score": None}))
    audits = [mapping.outcome, *mapping.groups]
    reserved = {*audits, mapping.score}
    if len(set(audits)) != len(audits):
        raise ValueError("Use a different CSV column for the outcome and each audit group.")
    if not features or len(set(features)) != len(features) or any(
        feature in reserved or feature not in frame.columns for feature in features
    ):
        raise ValueError("Choose unique feature columns separate from the outcome, score and audit groups.")
    outcome = frame[mapping.outcome].astype(int)
    counts = outcome.value_counts()
    if len(counts) != 2 or counts.min() < 4:
        raise ValueError("Demo training requires at least four rows for each outcome (0 and 1).")

    train_rows, test_rows = train_test_split(
        np.arange(len(frame)), test_size=0.25, random_state=42, stratify=outcome,
    )
    test_rows.sort()
    train = pd.DataFrame(index=train_rows)
    test = pd.DataFrame(index=test_rows)
    for position, feature in enumerate(features):
        values = frame[feature]
        name = f"feature_{position}"
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            values = values.astype(float).replace([np.inf, -np.inf], np.nan)
            train[name] = values.iloc[train_rows].to_numpy()
            test[name] = values.iloc[test_rows].to_numpy()
        else:
            train[name] = pd.Categorical(values.iloc[train_rows])
            test[name] = pd.Categorical(values.iloc[test_rows], categories=train[name].cat.categories)
    if not (train.nunique() > 1).any():
        raise ValueError("Choose at least one feature with multiple usable values in the training rows.")

    model = LGBMClassifier(
        objective="binary", num_leaves=15, learning_rate=0.05, n_estimators=54,
        min_child_samples=min(20, max(2, len(train_rows) // 10)),
        random_state=42, deterministic=True, force_col_wise=True, n_jobs=1, verbosity=-1,
    )
    try:
        model.fit(train, outcome.iloc[train_rows].to_numpy())
        scores = model.predict_proba(test)[:, 1]
    except LightGBMError as error:
        raise ValueError("LightGBM could not train with the selected feature columns.") from error

    held_out = frame.iloc[test_rows].loc[:, audits].copy()
    prediction, score = "_demo_prediction", "_demo_score"
    while prediction in held_out.columns:
        prediction += "_"
    while score in held_out.columns:
        score += "_"
    held_out[score] = scores
    held_out[prediction] = (scores >= DEMO_THRESHOLD).astype(int)
    return held_out, ColumnMapping(outcome=mapping.outcome, groups=mapping.groups, prediction=prediction, score=score)
