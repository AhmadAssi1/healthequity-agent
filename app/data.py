import csv
from collections import Counter
from contextlib import nullcontext

import pandas as pd


def load_data(path):
    """Load a UTF-8 CSV path or seekable text stream without renaming headers."""
    with (
        nullcontext(path)
        if hasattr(path, "read")
        else open(path, encoding="utf-8-sig", newline="")
    ) as csv_file:
        position = csv_file.tell()
        reader = csv.reader(csv_file)
        # Match pandas' default behavior of skipping blank lines before the header.
        header = next(
            (row for row in reader if row and not (
                len(row) == 1 and not row[0].strip()
            )),
            [],
        )
        duplicates = [name for name, count in Counter(header).items() if count > 1]
        if duplicates:
            raise ValueError(f"Column names must be unique. Duplicates: {duplicates}")

        csv_file.seek(position)
        return pd.read_csv(
            csv_file,
            keep_default_na=False,
            na_values=["?"],
            low_memory=False,
        )


def create_target(df):
    """Create binary labels only from valid, non-missing readmission categories."""
    if "readmitted" not in df.columns:
        raise ValueError("Missing required column: 'readmitted'.")
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")

    readmitted = df["readmitted"]
    if readmitted.isna().any():
        raise ValueError("Column 'readmitted' must not contain missing values.")
    if not readmitted.isin(["<30", ">30", "NO"]).all():
        raise ValueError(
            "Column 'readmitted' must contain only '<30', '>30', or 'NO'."
        )

    return readmitted.eq("<30").astype(int).tolist()
