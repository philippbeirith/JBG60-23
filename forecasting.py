
#Import general libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

#Import scripts
from scripts import preprocessing
from scripts import evaluation


### Begin with pre processing ###
# Read data into DataFrame
df_food_crises_cleaned: pd.DataFrame = pd.read_csv("data/food_crises_cleaned.csv")
df_southsudan: pd.DataFrame = pd.read_csv("data/all_africa_southsudan.csv")

#Modify datasets
df_food_crises_cleaned = preprocessing.backfill_ipc(df_food_crises_cleaned)
df_southsudan = preprocessing.sentiment_analysis(df_southsudan, 'paragraphs')

#Augment
df_food_crises_cleaned = preprocessing.calculate_crises_metrics(df_food_crises_cleaned)
df_southsudan = preprocessing.calculate_news_metrics(df_southsudan).reset_index()

#Consolidate
dataset: pd.DataFrame = preprocessing.consolidate_data(df_southsudan, df_food_crises_cleaned)
dataset=dataset[dataset['ipc_delta']!=0]

### Feature Selection ###
#test for longevity
#dataset_val = dataset[dataset['date'] >= '2023-01-01']

dataset.drop(columns=['date', 'district', 'country', 'year_month', 'ha'], inplace = True)

model_lead_1 = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1']), dataset['ipc_lead_1'], test_size=0.2, random_state=23)
model_lead_3 = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_3']), dataset['ipc_lead_3'], test_size=0.2, random_state=23)
model_lead_6 = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_6']), dataset['ipc_lead_6'], test_size=0.2, random_state=23)
