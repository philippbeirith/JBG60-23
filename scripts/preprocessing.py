#Import  libraries
import pandas as pd
from textblob import TextBlob
import numpy as np

# Calculates delta values for the ipc column based on district and fills in nan values with 0
def calculate_delta(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f"{col}_delta"] = df.groupby("district")[col].diff()
    df[f"{col}_delta"] = df[f"{col}_delta"].fillna(0)
    return df

# Calculates the number of months since the last change in the ipc column based on district
def calculate_months_since_change(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Initialize the new column with NaN values
    df[f"{col}_months_since_change"] = float('NaN')

    # Create a dictionary to keep track of the last change date for each district
    last_change_date: dict[str, pd.Timestamp] = {}

    # Iterate through the DataFrame to calculate months_since_change for each district
    for index, row in df.iterrows():
        district = row['district']
        if district not in last_change_date:
            last_change_date[district] = row['date']

        if index > 0 and row[col] != df.at[index - 1, col]:
            last_change_date[district] = row['date']

        # Calculate the number of months since the last change
        days_since_change = (row['date'] - last_change_date[district]).days
        months_since_change = days_since_change / 30.44  # Average number of days in a month
        df.at[index, f"{col}_months_since_change"] = months_since_change

    # Convert months_since_change to int
    df[f"{col}_months_since_change"] = df[f"{col}_months_since_change"].astype(int)

    return df

# Calculates the lag value for a given list of columns based on district
def apply_lag(df: pd.DataFrame, col_list: list, lag: int) -> pd.DataFrame:

    # Loop through each column
    for col in col_list:
        # Add a lagged version of the column
        df[f"{col}_lag_{lag}"] = df.groupby("district")[col].shift(lag)

    return df

# Calculates the lead value for a given list of columns based on district
def apply_lead(df: pd.DataFrame, col_list: list, lead: int) -> pd.DataFrame:

    # Loop through each column
    for col in col_list:
        # Add a lead version of the column
        df[f"{col}_lead_{lead}"] = df.groupby("district")[col].shift(-lead)

    return df

# Calculates rolling averages for a given list of columns based on district
def apply_rolling_avg(df: pd.DataFrame, col_list: list, window: int) -> pd.DataFrame:

    # Loop through each column
    for col in col_list:
        # Add a rolling average column
        df[f"{col}_rolling_avg_{window}"] = df.groupby("district")[col].rolling(window).mean().reset_index(0, drop=True)

    return df

# Calculates rolling standard deviation for a given list of columns based on district
def apply_rolling_std(df: pd.DataFrame, col_list: list, window: int) -> pd.DataFrame:

    # Loop through each column
    for col in col_list:
        # Add a rolling standard deviation column
        df[f"{col}_rolling_std_{window}"] = df.groupby("district")[col].rolling(window).std().reset_index(0, drop=True)

    return df

# Calculates a boolean value for rain if a critical threshold of too less rain in mm is met
def calculate_rain_bool(df: pd.DataFrame) -> pd.DataFrame:
    df["rain_bool"] = df["rain_mean"] < 1.25

    # Convert boolean values to int
    df["rain_bool"] = df["rain_bool"].astype(int)

    return df

# Calculates a boolean value for ndvi if a critical threshold of too less greeenery is met
def calculate_ndvi_bool(df: pd.DataFrame) -> pd.DataFrame:
    df["ndvi_bool"] = df["ndvi_mean"] < 0.33

    # Convert boolean values to int
    df["ndvi_bool"] = df["ndvi_bool"].astype(int)

    return df

# Calculates a boolean value if et_mean is higher than rain_mean as in this case the soil would dry out
def calculate_et_bool(df: pd.DataFrame) -> pd.DataFrame:
    df["et_bool"] = df["et_mean"] > df["rain_mean"]

    # Convert boolean values to int
    df["et_bool"] = df["et_bool"].astype(int)
    
    return df

# Backfill values for "ipc" column and save file to csv
def backfill_ipc(df: pd.DataFrame) -> pd.DataFrame:
    df["ipc"] = df["ipc"].fillna(method="bfill")
    return(df)

# Apply sentiment analysis to each article within a column and set it to either 1 0 or -1
def sentiment_analysis(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df["sentiment"] = df[column].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    return(df)
    
def calculate_crises_metrics(df: pd.DataFrame):

    #Add crisis level delta (if changed, then by how many levels?)
    df = calculate_delta(df, col = "ipc")

    #Add 'months_since_change' 
    df = calculate_months_since_change(df, col = "ipc")

    #Add metrics for things like rain or price (declining trend, higher or lower, lags, rolling averages)
    df = calculate_rain_bool(df)
    df = calculate_ndvi_bool(df)
    df = calculate_et_bool(df)

    #Add leads/lags for various columns (1, 3, 6 months?)
    cols = ["ipc", "ndvi_mean", "rain_mean", "et_mean", "food_price_idx"] 
    df = apply_lag(df, col_list = cols, lag = 1)
    df = apply_lag(df, col_list = cols, lag = 3)
    df = apply_lag(df, col_list = cols, lag = 6)
    
    df = apply_lead(df, col_list = cols, lead = 1)
    df = apply_lead(df, col_list = cols, lead = 3)
    df = apply_lead(df, col_list = cols, lead = 6)
    
    df = apply_rolling_avg(df, col_list = cols, window = 3)
    df = apply_rolling_std(df, col_list = cols, window = 3)
    
    df = calculate_rain_bool(df)
    df = calculate_ndvi_bool(df)
    df = calculate_et_bool(df)

    return(df)

#This consolidates the food_crises metrics and all_africa_southsudan dataset, ready to be fed into the model.
def consolidate_data(summary: pd.DataFrame, crisis: pd.DataFrame):

    output_df = pd.merge(summary, crisis, how='left', on=['date','district'])
    return(output_df)
