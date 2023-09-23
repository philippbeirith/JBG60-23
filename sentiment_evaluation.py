import pandas as pd
import matplotlib.pyplot as plt
from sentiment_analysis import textblob_sentiment_analysis
from sentiment_analysis import spacy_sentiment_analysis

# Read data into DataFrame
df_all_africa = pd.read_csv("data/all_africa_southsudan.csv")

textblob_sentiment = textblob_sentiment_analysis(df_all_africa, "paragraphs")["sentiment"]
spacy_sentiment = spacy_sentiment_analysis(df_all_africa, "paragraphs")
spacy_subjectivity = spacy_sentiment['subjectivity']

# Plot sentiment analysis comparison of both methods
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.hist(textblob_sentiment, bins=3)
ax1.set_title("TextBlob Sentiment Analysis")
ax1.set_xlabel("Sentiment")
ax1.set_ylabel("Frequency")
ax1.set_xticks([-1, 0, 1])
ax2.hist(spacy_sentiment, bins=3)
ax2.set_title("SpacyTextBlob Sentiment Analysis")
ax2.set_xlabel("Sentiment")
ax2.set_xticks([-1, 0, 1])

# Plot subjectivity analysis of SpacyTextBlob and export to png
plt.figure(figsize=(10, 5))
plt.hist(spacy_subjectivity, bins=3)
plt.title("SpacyTextBlob Subjectivity Analysis")
plt.xlabel("Subjectivity")
plt.ylabel("Frequency")
plt.xticks([0, 0.5, 1])










