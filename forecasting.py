
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
df_crises = pd.read_csv("data/food_crises_cleaned.csv")
df_summary = pd.read_csv("data/articles_summary_cleaned.csv")

#Augment
df_crises = preprocessing.calculate_crises_metrics(df_crises)

#Modify datasets
df_crises = preprocessing.backfill_ipc(df_crises)
df_crises['date'] = pd.to_datetime(df_crises['date']).dt.strftime('%Y-%m')
df_crises['date'] = df_crises['date'].astype(str)
df_crises.drop(columns=["country", "district_code", "centx", "centy", "year_month", "year", "month", "ha"], inplace=True)

df_summary = preprocessing.sentiment_analysis(df_summary, "summary")
df_summary.drop(columns=["summary", "lat", "lng"], inplace=True)
df_summary.rename(columns={"location_article": "district"}, inplace=True)
df_summary['date'] = pd.to_datetime(df_summary['date']).dt.strftime('%Y-%m')
df_summary['date'] = df_summary['date'].astype(str)

#Consolidate
model_dataset = preprocessing.consolidate_data(df_summary, df_crises)
model_dataset = model_dataset.dropna(subset=['ipc'])
model_dataset['date'] = pd.to_datetime(model_dataset['date'])
model_dataset.drop(columns=["date"], inplace=True)

district_names = model_dataset["district"].unique()
district_mapping = {name: index for index, name in enumerate(district_names)}
int_data = [district_mapping[name] for name in model_dataset["district"]]
model_dataset["district"] = int_data

model_dataset = model_dataset.dropna()

### Feature Selection ###
#test for longevity
#dataset_val = dataset[dataset['date'] >= '2023-01-01']

model_lead_1 = evaluation.test_random_forest_classification_performance(model_dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), model_dataset['ipc_lead_1'], test_size=0.2, random_state=23)
print('############################################################')
#model_lead_3 = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), dataset['ipc_lead_3'], test_size=0.2, random_state=23)
#print('############################################################')
#model_lead_6 = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), dataset['ipc_lead_6'], test_size=0.2, random_state=23)
