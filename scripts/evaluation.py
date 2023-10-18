import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit

def test_random_forest_classification_performance(X, y, test_type, prediction_target, random_state=None):
    """
    Train and test a Random Forest classifier, then print its performance metrics.

    Parameters:
    - X: Features for training/testing.
    - y: Target variable.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Seed for reproducibility.

    Returns:
    - rf: Trained Random Forest classifier.
    """
    
    #Edit the independent variables
    nlp_columns = [col for col in X.columns if col.startswith('nlp_')]
    
    custom_feature_list1 = ['district_code', 'centx', 'centy', 'cropland_pct', 'pop',
           'ruggedness_mean', 'pasture_pct', 'ipc_months_since_change',
           'ipc_lag_1', 'food_price_idx_lag_1', 'ipc_lag_3', 'ndvi_mean_lag_3',
           'rain_mean_lag_3', 'et_mean_lag_3', 'food_price_idx_lag_3', 'ipc_lag_6',
           'food_price_idx_lag_6', 'ipc_rolling_avg_3',
           'food_price_idx_rolling_avg_3', 'food_price_idx_rolling_std_3']
    custom_feature_list1.extend(nlp_columns)

    custom_feature_list3 = ['district_code', 'ipc_months_since_change',
           'ipc_lag_1', 'ipc_lag_3', 'food_price_idx_lag_6', 'food_price_idx_rolling_avg_3', 'food_price_idx_rolling_std_3', 
           'ipc_lag_6','food_price_idx_lag_1','food_price_idx_lag_3']
    custom_feature_list3.extend(nlp_columns)

    X = X[custom_feature_list1]
    
    # Split data into training and testing sets
    tscv = TimeSeriesSplit(10)
    
    split_metrics = pd.DataFrame(columns=[
        'test_type', 'prediction_target', 'k_fold',
        'Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1 Score (Weighted)'
    ])
    all_district_accuracies = []
    k_fold=0
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Create and fit the RandomForestClassifier
    for train_index, test_index in tscv.split(X):
        k_fold +=1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
         
        rf = RandomForestClassifier(n_estimators = 5000, random_state = 23, max_depth = 7)
    
        
        rf.fit(X_train, y_train)
        # Make predictions
        y_pred = rf.predict(X_test)
        y_pred_prob = rf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
        
        # Store metrics for this split
        current_metrics = {
            'test_type': test_type,
            'prediction_target': prediction_target,
            'k_fold': k_fold,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision (Weighted)': precision_score(y_test, y_pred, average='weighted'),
            'Recall (Weighted)': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score (Weighted)': f1_score(y_test, y_pred, average='weighted')
        }
        current_metrics_df = pd.DataFrame([current_metrics])  # Note the list around the dictionary
        split_metrics = pd.concat([split_metrics,current_metrics_df], ignore_index=True)
        
        # Calculate district accuracies for this split
        district_accuracies = {}
        unique_districts = X_test['district_code'].unique()  # Assuming 'district_code' is in X_test
        
        for district in unique_districts:
            district_mask = X_test['district_code'] == district
            district_y_true = y_test[district_mask]
            district_y_pred = y_pred[district_mask]
        
            district_accuracy = accuracy_score(district_y_true, district_y_pred)
            district_accuracies[district] = district_accuracy
        
            all_district_accuracies.append((test_type, prediction_target, k_fold, district, district_accuracy))
    
    # Create a DataFrame to store the accuracy rates
    district_accuracy_df = pd.DataFrame(all_district_accuracies, columns=['Test_Type', 'Prediction_Target','K_Fold', 'District', 'Accuracy'])

    importances = rf.feature_importances_
    indices = np.argsort(importances)
    features = X.columns
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

    return rf, district_accuracy_df, split_metrics

