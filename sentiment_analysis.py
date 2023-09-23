from textblob import TextBlob
import pandas as pd

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load("en_core_web_sm")

# Apply sentiment analysis to each article within a column and set it to either 1 0 or -1
def textblob_sentiment_analysis(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df["sentiment"] = df[column].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    return df

# Apply spacytextblob sentiment analysis on articles within a column and set the outcomes to either 1 0 or -1
def spacy_sentiment_analysis(df: pd.DataFrame, column: str) -> pd.DataFrame:
    nlp.add_pipe("spacytextblob")
    df["sentiment"] = df[column].apply(lambda x: nlp(x)._.polarity)
    df["subjectivity"] = df[column].apply(lambda x: nlp(x)._.subjectivity)
    df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    return df