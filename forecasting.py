#Import general libraries
import pandas as pd
import matplotlib.pyplot as plt
#Import scripts

from scripts import preprocessing
from scripts import evaluation

def shap_graph(shap_1, shap_2, shap_3):

    shap_value_df = pd.DataFrame(columns=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"])
    shap_value_df.loc[len(shap_value_df)] = shap_1
    shap_value_df.loc[len(shap_value_df)] = shap_2
    shap_value_df.loc[len(shap_value_df)] = shap_3
    shap_value_df["Model"] = ["Model Lead 1", "Model Lead 3", "Model Lead 6"]
    
    shap_value_df.plot.barh(x = "Model", stacked = True, title = "BERTopic vs ChatGPT")
    plt.gca().set_ylabel("")

### Begin with pre processing ###
# Read data into DataFrame
df_food_crises_cleaned: pd.DataFrame = pd.read_csv("data/food_crises_cleaned.csv")
df_southsudan: pd.DataFrame = pd.read_csv("data/all_africa_southsudan.csv")
#bert
df_bert: pd.DataFrame = pd.read_csv("data/bert_output.csv")
#jason
df_classifications: pd.DataFrame = pd.read_csv("data/Jason_articles_with_classifications.csv")
#grouping
df_grouping = pd.read_csv('data/topics_group_summary_results.csv')

#Modify datasets
df_food_crises_cleaned = preprocessing.backfill_ipc(df_food_crises_cleaned)

#Augment
df_food_crises_cleaned = preprocessing.calculate_crises_metrics(df_food_crises_cleaned)
df_southsudan = preprocessing.calculate_news_metrics(df_southsudan).reset_index()

#Prepare NLP Models
df_bert = preprocessing.bert_prep(df_bert)
df_classifications = preprocessing.classification_prep(df_classifications, 'topics_7')
df_grouping = preprocessing.grouped_prep(df_grouping, df_food_crises_cleaned)

#Consolidate
dataset: pd.DataFrame = preprocessing.consolidate_data(df_southsudan, df_food_crises_cleaned, df_bert, 'bert')

#Limit the data to be < 2015
#dataset = dataset[dataset['date'] <= '2015-01-01']

dataset.drop(columns=['date', 'district', 'country', 'year_month', 'ha', 'next_month_change'], inplace = True)
dataset.columns = dataset.columns.astype(str)
dataset=dataset.dropna()

# =============================================================================
model_lead_1, model_lead_1_accuracy, shap_1 = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), dataset['ipc_lead_1'], test_size=0.3, random_state=23)
# print('############################################################')
model_lead_3, model_lead_3_accuracy, shap_2 = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), dataset['ipc_lead_3'], test_size=0.3, random_state=23)
# print('############################################################')
model_lead_6, model_lead_6_accuracy, shap_3 = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), dataset['ipc_lead_6'], test_size=0.3, random_state=23)

model_lead_1_accuracy.to_csv('/Users/philippbeirith/Downloads/model_lead_1_accuracy.csv')
model_lead_3_accuracy.to_csv('/Users/philippbeirith/Downloads/model_lead_3_accuracy.csv')
model_lead_6_accuracy.to_csv('/Users/philippbeirith/Downloads/model_lead_6_accuracy_2.csv')
# =============================================================================

    
    
