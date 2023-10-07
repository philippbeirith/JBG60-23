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

def next_month_change(df):
    df['next_value'] = df.groupby('district_code')['ipc'].shift(-1)
    df['next_month_change'] = (df['ipc'] != df['next_value']).astype(int)
    df.drop(columns=['next_value'], inplace=True)
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

# This helper function was taken from the 'predicting' jupyter notebook from https://github.com/GielJW/JBG060-DC3-23-24-public/tree/main
def create_lag_df(df, columns, lag, difference=False, rolling=None, dropna=False):
    '''
    Function to add lagged colums to dataframe
    
    Inputs:
        df - Dataframe
        columns - List of columns to create lags from
        lag - The number of timesteps (in months for the default data) to lag the variable by
        difference - Whether to take the difference between each observation as new column
        rolling - The size of the rolling mean window, input None type to not use a rolling variable
        dropna - Whether to drop NaN values
        
    Output:
        df - Dataframe with the lagged columns added
    '''
    
    for column in columns:
        col = df[column].unstack()
        if rolling:
            col = col.rolling(rolling).mean()
        if difference:
            col = col.diff()
        if dropna:
            col = col.dropna(how='any')
        df[f"{column}_lag_{lag}"] = col.shift(lag).stack()
    return df

#This functions aims to quantify and parameterize the news article dataset such that it can be used in random forest.
def calculate_news_metrics(df: pd.DataFrame):
    #The final version of this dataset needs to be an aggregate per month
    
    #count(*) of all articles in a month
    df['date'] = df['date'].str[:7]
    article_count = df.groupby('date').count()
    #article_count = df.groupby(pd.Grouper(key='date', freq='M')).count()
    article_count = article_count.rename(columns={'title':'article_count'})['article_count'] 
    print(article_count)
    #count(distinct *) of unique publishers
    publisher_count = df.groupby('date')['publisher'].nunique()
    output_df = pd.merge(article_count,publisher_count,on='date')
    
    #percent_negative bounds -1:-.33
    #percent_negative = publisher_count
    #output_df = pd.merge(output_df,percent_negative,on='date')
    
    #percent_neutral bounds -.33:.33
    #percent_neutral = publisher_count
    #output_df = pd.merge(output_df,percent_neutral,on='date')
    
    #percent_positive bound .34:1
    #percent_positive = publisher_count
    #output_df = pd.merge(output_df,percent_positive,on='date')
    
    #nlp to be added later
    
    return(output_df)
    
def calculate_crises_metrics(df: pd.DataFrame):
    #Add crisis level delta (if changed, then by how many levels?)
    df = calculate_delta(df, col = "ipc")

    
    #Add 'months_since_change' 
    df = calculate_months_since_change(df, col = "ipc")
    
    #Add a tag for when the delta changes in the next entry
    df = next_month_change(df)
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

#This section prepares the nlp data
def bert_prep(df):
    # Convert the 'date' column to a datetime object
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract year and month from the 'date' column
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Group by year, month, and category and count the number of articles
    grouped = df.groupby(['year', 'month', 'predictions'])['summary'].count().reset_index()
    grouped['date'] = pd.to_datetime(grouped['year'].astype(str) + '-' + grouped['month'].astype(str) + '-01')

    # Pivot the table to have categories as columns and calculate the proportion
    pivot_table = pd.pivot_table(grouped, values='summary', index=['date'], columns='predictions', fill_value=0)
    pivot_table = pivot_table.div(pivot_table.sum(axis=1), axis=0).reset_index()
    pivot_table = pivot_table.fillna(0)
    
    pivot_table.columns = ['nlp_' + str(col) if col != 'date' else col for col in pivot_table.columns]

    
    return pivot_table

def classification_prep(df, col):
    # Convert the 'date' column to a datetime object
    df['date'] = pd.to_datetime(df['date'])
    
    grouped = df.groupby(['date', col])['summary'].count().reset_index()
    grouped['date'] = pd.to_datetime(grouped['date'].dt.year.astype(str) + '-' + grouped['date'].dt.month.astype(str) + '-01')
    
    pivot_table = pd.pivot_table(grouped, values='summary', index=['date'], columns=col, fill_value=0)
    pivot_table = pivot_table.div(pivot_table.sum(axis=1), axis=0).reset_index()
    pivot_table = pivot_table.fillna(0)
    
    pivot_table.columns = ['nlp_' + str(col) if col != 'date' else col for col in pivot_table.columns]

    
    return(pivot_table)


#This consolidates the food_crises metrics and all_africa_southsudan dataset, ready to be fed into the model.
def consolidate_data(southsudan: pd.DataFrame, crisis: pd.DataFrame, nlp):
    crisis['date'] = crisis['date'].astype('str')
    crisis['date'] = crisis['date'].str[:7]
    southsudan['date'] = southsudan['date'].astype('str')
    output_df = southsudan.merge(crisis, on='date')
    output_df['country_code'] = np.where(output_df['country'].astype('str') == 'South Sudan', 1, 2)
    
    if nlp is not None and isinstance(nlp, pd.DataFrame):
        nlp['date'] = nlp['date'].astype('str')
        nlp['date'] = nlp['date'].str[:7]
        output_df=output_df.merge(nlp, on='date')
        
    return(output_df)





