
#Import general libraries
import pandas as pd

#Import scripts
from scripts import preprocessing
from scripts import evaluation


### Begin with pre processing ###
# Read data into DataFrame
df_food_crises_cleaned: pd.DataFrame = pd.read_csv("data/food_crises_cleaned.csv")
df_southsudan: pd.DataFrame = pd.read_csv("data/all_africa_southsudan.csv")
#bert
df_bert: pd.DataFrame = pd.read_csv("data/bert_output.csv")
#jason
df_classifications: pd.DataFrame = pd.read_csv("data/Jason_articles_with_classifications.csv")


#Modify datasets
df_food_crises_cleaned = preprocessing.backfill_ipc(df_food_crises_cleaned)
df_southsudan = preprocessing.sentiment_analysis(df_southsudan, 'paragraphs')

#Augment
df_food_crises_cleaned = preprocessing.calculate_crises_metrics(df_food_crises_cleaned)
df_southsudan = preprocessing.calculate_news_metrics(df_southsudan).reset_index()

#Prepare NLP Models
df_bert = preprocessing.bert_prep(df_bert)
df_classifications = preprocessing.classification_prep(df_classifications, 'topics_7')

#Consolidate
dataset: pd.DataFrame = preprocessing.consolidate_data(df_southsudan, df_food_crises_cleaned, df_classifications)

#dataset = dataset[dataset['next_month_change']== 1].dropna()
dataset.drop(columns=['date', 'district', 'country', 'year_month', 'ha', 'next_month_change'], inplace = True)
dataset.columns = dataset.columns.astype(str)
dataset=dataset.dropna()

model_lead_1 = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), dataset['ipc_lead_1'], test_size=0.2, random_state=23)
print('############################################################')
model_lead_3 = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), dataset['ipc_lead_3'], test_size=0.2, random_state=23)
print('############################################################')
model_lead_6 = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), dataset['ipc_lead_6'], test_size=0.2, random_state=23)
