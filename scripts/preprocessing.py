#Import  libraries
import pandas as pd
from textblob import TextBlob
import numpy as np

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
    article_count = df.groupby('date').count()
    
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
    
    #Add 'months_since_change' 
    
    #Add metrics for things like rain or price (declining trend, higher or lower, lags, rolling averages)
    
    #Add leads for icp (1, 3, 6 months?)
    print('hi')
    
#This consolidates the food_crises metrics and all_africa_southsudan dataset, ready to be fed into the model.
def consolidate_data(southsudan: pd.DataFrame, crisis: pd.DataFrame):
    southsudan['date'] = southsudan['date'].astype('str')
    crisis['date'] = crisis['date'].astype('str')
    output_df = southsudan.merge(crisis, on='date')
    output_df['country_code'] = np.where(output_df['country'].astype('str') == 'South Sudan', 1, 2)

        
    
    return(output_df)
