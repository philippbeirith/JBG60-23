
#Import general libraries
import pandas as pd

#Import scripts

from scripts import preprocessing
from scripts import evaluation
from scripts import evaluation_test

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
#df_southsudan = preprocessing.sentiment_analysis(df_southsudan, 'paragraphs')

#Augment
df_food_crises_cleaned = preprocessing.calculate_crises_metrics(df_food_crises_cleaned)
df_southsudan = preprocessing.calculate_news_metrics(df_southsudan).reset_index()

#Prepare NLP Models
df_bert = preprocessing.bert_prep(df_bert)
df_classifications = preprocessing.classification_prep(df_classifications, 'topics_7')
df_grouping = preprocessing.grouped_prep(df_grouping, df_food_crises_cleaned)

###########################################################################################################################################################################
#Consolidate
dataset: pd.DataFrame = preprocessing.consolidate_data(df_southsudan, df_food_crises_cleaned, df_grouping, 'grouping')

#dataset = dataset[dataset['next_month_change']== 1].dropna()
#Limit the data to be < 2015
#dataset = dataset[dataset['date'] <= '2015-01-01']

dataset.drop(columns=['date', 'district', 'country', 'year_month', 'ha', 'next_month_change'], inplace = True)
dataset.columns = dataset.columns.astype(str)
dataset=dataset.dropna()

district_accuracy = pd.DataFrame()
evaluations = pd.DataFrame()

model_lead_1, model_lead_1_district_accuracy, model_1_evaluation = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), dataset['ipc_lead_1'], 'grouping', '1', random_state=23)
district_accuracy = pd.concat([district_accuracy,model_lead_1_district_accuracy], ignore_index=True)
evaluations = pd.concat([evaluations,model_1_evaluation], ignore_index=True)
print('############################################################')
model_lead_3, model_lead_3_district_accuracy, model_3_evaluation = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), dataset['ipc_lead_3'], 'grouping', '3', random_state=23)
district_accuracy = pd.concat([district_accuracy,model_lead_3_district_accuracy], ignore_index=True)
evaluations = pd.concat([evaluations,model_3_evaluation], ignore_index=True)
print('############################################################')
model_lead_6, model_lead_6_district_accuracy, model_6_evaluation = evaluation.test_random_forest_classification_performance(dataset.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']), dataset['ipc_lead_6'], 'grouping', '6', random_state=23)
district_accuracy = pd.concat([district_accuracy,model_lead_6_district_accuracy], ignore_index=True)
evaluations = pd.concat([evaluations,model_6_evaluation], ignore_index=True)

evaluations.to_csv('/evaluations.csv')
district_accuracy.to_csv('/district_accuracy.csv')



###########################################################################################################################################################################
district_accuracy = pd.DataFrame()
evaluations = pd.DataFrame()


dataset: pd.DataFrame = preprocessing.consolidate_data(df_southsudan, df_food_crises_cleaned, df_classifications, 'bert')

model_lead_1, model_lead_1_district_accuracy, model_1_evaluation = evaluation_test.test_random_forest_classification_performance_ts(dataset, 'Baseline', 'ipc_lead_1', random_state=23)
district_accuracy = pd.concat([district_accuracy,model_lead_1_district_accuracy], ignore_index=True)
evaluations = pd.concat([evaluations,model_1_evaluation], ignore_index=True)
print('############################################################')
model_lead_3, model_lead_3_district_accuracy, model_3_evaluation = evaluation_test.test_random_forest_classification_performance_ts(dataset, 'Baseline', 'ipc_lead_3', random_state=23)
district_accuracy = pd.concat([district_accuracy,model_lead_3_district_accuracy], ignore_index=True)
evaluations = pd.concat([evaluations,model_3_evaluation], ignore_index=True)
print('############################################################')
model_lead_6, model_lead_6_district_accuracy, model_6_evaluation = evaluation_test.test_random_forest_classification_performance_ts(dataset, 'Baseline', 'ipc_lead_6', random_state=23)
district_accuracy = pd.concat([district_accuracy,model_lead_6_district_accuracy], ignore_index=True)
evaluations = pd.concat([evaluations,model_6_evaluation], ignore_index=True)


evaluations.to_csv('/evaluations_test.csv')
district_accuracy.to_csv('/district_accuracy_test.csv')



model_lead_1, model_lead_1_district_accuracy, model_1_evaluation = evaluation_test.test_random_forest_classification_performance_ts(dataset, 'Baseline', 'ipc_lead_1', random_state=23)
district_accuracy = pd.concat([district_accuracy,model_lead_1_district_accuracy], ignore_index=True)
evaluations = pd.concat([evaluations,model_1_evaluation], ignore_index=True)
print('############################################################')
model_lead_3, model_lead_3_district_accuracy, model_3_evaluation = evaluation_test.test_random_forest_classification_performance_ts(dataset, 'Baseline', 'ipc_lead_3', random_state=23)
district_accuracy = pd.concat([district_accuracy,model_lead_3_district_accuracy], ignore_index=True)
evaluations = pd.concat([evaluations,model_3_evaluation], ignore_index=True)
print('############################################################')
model_lead_6, model_lead_6_district_accuracy, model_6_evaluation = evaluation_test.test_random_forest_classification_performance_ts(dataset, 'Baseline', 'ipc_lead_6', random_state=23)
district_accuracy = pd.concat([district_accuracy,model_lead_6_district_accuracy], ignore_index=True)
evaluations = pd.concat([evaluations,model_6_evaluation], ignore_index=True)


evaluations.to_csv('/evaluations_test.csv')
district_accuracy.to_csv('/district_accuracy_test.csv')


