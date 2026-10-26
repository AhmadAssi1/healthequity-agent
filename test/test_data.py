import pandas as pd

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