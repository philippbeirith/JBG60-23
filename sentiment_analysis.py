from textblob import TextBlob
import pandas as pd

# Read data into DataFrame
df_all_africa: pd.DataFrame = pd.read_csv("data/all_africa_southsudan.csv")

# Apply sentiment analysis to each article within a column and set it to either 1 0 or -1
def sentiment_analysis(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df["sentiment"] = df[column].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1)
    df.to_csv("data/all_africa_southsudan.csv", index=False)

# Apply sentiment analysis to the "paragraphs" column
sentiment_analysis(df_all_africa, "paragraphs")