import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit

def test_random_forest_classification_performance_ts(df,test_type, prediction_target, random_state=None):
    
    df['date'] = pd.to_datetime(df['date'])
    cutoff_date = df['date'].max() - pd.DateOffset(years=3)
    print(df['date'].max())
    print(cutoff_date)
    df.columns = df.columns.astype(str)
    df=df.dropna()

    nlp_columns = [col for col in df.columns if col.startswith('nlp_')]
    
    custom_feature_list1 = ['date','district_code', 'centx', 'centy', 'cropland_pct', 'pop',
           'ruggedness_mean', 'pasture_pct', 'ipc_months_since_change',
           'ipc_lag_1', 'food_price_idx_lag_1', 'ipc_lag_3', 'ndvi_mean_lag_3',
           'rain_mean_lag_3', 'et_mean_lag_3', 'food_price_idx_lag_3', 'ipc_lag_6',
           'food_price_idx_lag_6', 'ipc_rolling_avg_3',
           'food_price_idx_rolling_avg_3', 'food_price_idx_rolling_std_3', 'ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']
    custom_feature_list1.extend(nlp_columns)

    custom_feature_list3 = ['date','district_code', 'ipc_months_since_change',
           'ipc_lag_1', 'ipc_lag_3', 'food_price_idx_lag_6', 'food_price_idx_rolling_avg_3', 'food_price_idx_rolling_std_3', 
           'ipc_lag_6','food_price_idx_lag_1','food_price_idx_lag_3', 'ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6']
    custom_feature_list3.extend(nlp_columns)

    df = df[custom_feature_list1]
    
    ###
    train_set = df[df['date'] <= cutoff_date]
    test_set = df[df['date'] > cutoff_date]

    X_train = train_set.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6', 'date'])
    y_train = train_set[prediction_target]

    X_test = test_set.drop(columns=['ipc_lead_1', 'ipc_lead_3', 'ipc_lead_6', 'date'])
    y_test = test_set[prediction_target]
    
    print(len(X_test))

    split_metrics = pd.DataFrame(columns=[
        'test_type', 'prediction_target',
        'Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1 Score (Weighted)'
    ])
    all_district_accuracies = []
    
    # Create and fit the RandomForestClassifier
    rf = RandomForestClassifier(n_estimators = 5000, random_state = 23, max_depth = 7)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Store metrics for this split
    current_metrics = {
        'test_type': test_type,
        'prediction_target': prediction_target,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (Weighted)': precision_score(y_test, y_pred, average='weighted'),
        'Recall (Weighted)': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score (Weighted)': f1_score(y_test, y_pred, average='weighted')
    }
    current_metrics_df = pd.DataFrame([current_metrics])  
    split_metrics = pd.concat([split_metrics,current_metrics_df], ignore_index=True)
    
    # Calculate district accuracies for this split
    district_accuracies = {}
    unique_districts = X_test['district_code'].unique()
    
    for district in unique_districts:
        district_mask = X_test['district_code'] == district
        district_y_true = y_test[district_mask]
        district_y_pred = y_pred[district_mask]
    
        district_accuracy = accuracy_score(district_y_true, district_y_pred)
        district_accuracies[district] = district_accuracy
    
        all_district_accuracies.append((test_type, prediction_target, district, district_accuracy))

    district_accuracy_df = pd.DataFrame(all_district_accuracies, columns=['Test_Type', 'Prediction_Target', 'District', 'Accuracy'])

    importances = rf.feature_importances_
    indices = np.argsort(importances)
    features = X_train.columns
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

    return rf, district_accuracy_df, split_metrics

