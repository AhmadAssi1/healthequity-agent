import hashlib
import json

import pandas as pd

from app.schemas import ColumnMapping, DatasetFingerprint


def validate_dataset(df: pd.DataFrame, mapping: ColumnMapping):
    if df.empty:
        raise ValueError("Dataset must contain at least one data row.")

    
        
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")
    required_columns = [mapping.outcome, *mapping.groups]

    if mapping.prediction is not None:
        required_columns.append(mapping.prediction)

    if mapping.score is not None:
        required_columns.append(mapping.score)

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing mapped columns: {missing_columns}")

    
    label_columns = [mapping.outcome]

    if mapping.prediction is not None:
        label_columns.append(mapping.prediction)

    for column in label_columns:
        if df[column].isna().any():
            raise ValueError(
                f"Column '{column}' must not contain missing values."
            )
        
        
        values = df[column]

        if (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
        ):
            raise ValueError(
                f"Column '{column}' must contain numeric binary labels."
            )

        if not values.isin([0, 1]).all():
            raise ValueError(
                f"Column '{column}' must contain only 0 or 1."
            )   


        
        if mapping.score is not None:
            scores = df[mapping.score]

            if scores.isna().any():
                raise ValueError("Score column must not contain missing values.")

            if (
                not pd.api.types.is_numeric_dtype(scores)
                or pd.api.types.is_bool_dtype(scores)
            ):
                raise ValueError("Score column must contain numeric probabilities.")

            if not scores.between(0, 1).all():
                raise ValueError("Score values must be between 0 and 1.")    


def create_dataset_fingerprint(df: pd.DataFrame) -> DatasetFingerprint:
    column_names = json.dumps(
        df.columns.tolist(),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    columns_hash = hashlib.sha256(
        column_names.encode("utf-8")
    ).hexdigest()

    return DatasetFingerprint(
        row_count=len(df),
        columns_hash=columns_hash,
    )
    