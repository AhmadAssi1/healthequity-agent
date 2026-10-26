import pandas as pd


def load_data(path):
    df = pd.read_csv(
    path,
    keep_default_na=False,
    na_values=["?"],
    low_memory=False,
)
    return df

def create_target(df):
    return [int(x == "<30") for x in df["readmitted"]]

 