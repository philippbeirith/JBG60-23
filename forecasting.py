
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
#df_food_crises_cleaned = preprocessing.calculate_crises_metrics(df_food_crises_cleaned)
df_southsudan = preprocessing.calculate_news_metrics(df_southsudan).reset_index()

#Consolidate
dataset: pd.DataFrame = preprocessing.consolidate_data(df_southsudan, df_food_crises_cleaned)

### Feature Selection ###
#test for longevity
dataset_val = dataset[dataset['date'] >= '2023-01-01']

dataset.drop(columns=['date', 'district', 'country', 'year_month', 'ha'], inplace = True)
X_train, X_test, y_train, y_test = train_test_split(dataset.loc[:, dataset.columns != 'ipc'], dataset['ipc'], test_size=0.2, random_state=1)

# Create and fit the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=23)
rf_classifier.fit(X_train, y_train)

# Use SelectFromModel with the fitted classifier
sel = SelectFromModel(rf_classifier)
X_train = sel.transform(X_train)

### Train Model ###
rf = RandomForestRegressor(n_estimators = 1000, random_state = 23, max_depth = 5)
rf.fit(X_train, y_train)

### Evaluation ###
predictions = rf.predict(X_test)
errors = abs(predictions - y_test)

print('MAE:', round(np.mean(errors), 2))

mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')








