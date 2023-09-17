import pandas as pd

# Read data into DataFrame
df = pd.read_csv("data/food_crises_cleaned.csv")

# Backfill values for "ipc" column and save file to csv
def backfill(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].fillna(method="bfill")
    return df

# Fills in missing values for the `ipc` and `ha` columns.
df = backfill(df, "ipc")
df = backfill(df, "ha")
