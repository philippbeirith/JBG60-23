import pandas as pd

# Read data into DataFrame
df = pd.read_csv("data/food_crises_cleaned.csv")

# Backfill values for "ipc" column and save file to csv
def backfill_ipc(df: pd.DataFrame) -> pd.DataFrame:
    df["ipc"] = df["ipc"].fillna(method="bfill")
    df.to_csv("data/food_crises_cleaned.csv", index=False)

backfill_ipc(df)
