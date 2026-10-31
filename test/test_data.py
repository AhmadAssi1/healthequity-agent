from io import StringIO

import pandas as pd
import pytest

from app.data import create_target, load_data


def test_load_data_missing_values(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("value\n?\nNone\n", encoding="utf-8")

    df = load_data(csv_path)

    assert pd.isna(df.loc[0, "value"])
    assert df.loc[1, "value"] == "None"

def test_create_target():
    df = pd.DataFrame({
        "readmitted": ["<30", ">30", "NO"]
    })

    result = create_target(df)

    assert result == [1, 0, 0]


@pytest.mark.parametrize("missing_value", [None, float("nan"), pd.NA])
def test_create_target_missing_values_rejected(missing_value):
    df = pd.DataFrame({"readmitted": ["<30", missing_value]})

    with pytest.raises(ValueError, match="must not contain missing values"):
        create_target(df)


@pytest.mark.parametrize("invalid_value", ["TYPO", "", "?", "no", " NO ", 0, True])
def test_create_target_unknown_categories_rejected(invalid_value):
    df = pd.DataFrame({"readmitted": ["<30", invalid_value]})

    with pytest.raises(ValueError, match="must contain only"):
        create_target(df)


def test_create_target_missing_column_rejected():
    with pytest.raises(ValueError, match="Missing required column: 'readmitted'"):
        create_target(pd.DataFrame({"other": [1]}))


def test_create_target_duplicate_columns_rejected():
    df = pd.DataFrame([["NO", "<30"]], columns=["readmitted", "readmitted"])

    with pytest.raises(ValueError, match="Column names must be unique"):
        create_target(df)


def test_create_target_preserves_order_and_source():
    df = pd.DataFrame(
        {"readmitted": pd.Series(["NO", "<30", ">30", "<30"], dtype="category")}
    )
    df.index = [8, 2, 6, 1]
    original = df.copy(deep=True)

    assert create_target(df) == [0, 1, 0, 1]
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize(
    "content",
    [
        "outcome,outcome,gender\n1,0,Female\n",
        '\n  \n"out,come","out,come",gender\n1,0,Female\n',
        '\ufeff"outcome","outcome",gender\n1,0,Female\n',
    ],
)
def test_load_data_duplicate_headers_rejected(tmp_path, content):
    csv_path = tmp_path / "duplicate_headers.csv"
    csv_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Column names must be unique"):
        load_data(csv_path)


def test_load_data_preserves_distinct_headers_and_quoted_values(tmp_path):
    csv_path = tmp_path / "distinct_headers.csv"
    csv_path.write_text(
        'outcome,outcome.1,"group,name"\n1,0,"A,B"\n', encoding="utf-8"
    )

    df = load_data(csv_path)

    assert df.columns.tolist() == ["outcome", "outcome.1", "group,name"]
    assert df.loc[0, "group,name"] == "A,B"


def test_load_data_text_stream_stays_open():
    stream = StringIO("outcome,gender\n1,Female\n")

    df = load_data(stream)

    assert not stream.closed
    assert df.loc[0, "outcome"] == 1


def test_load_data_duplicate_headers_in_text_stream_rejected():
    stream = StringIO("outcome,outcome,gender\n1,0,Female\n")

    with pytest.raises(ValueError, match="Column names must be unique"):
        load_data(stream)
