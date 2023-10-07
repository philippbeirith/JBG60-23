import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix, log_loss)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFromModel

def test_random_forest_classification_performance(X, y, test_size=0.2, random_state=None):
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

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create and fit the RandomForestClassifier
    rf = RandomForestClassifier(n_estimators = 5000, random_state = 23, max_depth = 7)
    
  
    # Use SelectFromModel with the fitted classifier
    #sel = SelectFromModel(rf)
    
    #Print out selected rows
    #feature_idx = sel.get_support()
    #feature_name = X_train.columns[feature_idx]
    #print(feature_name)
    
    custom_feature_list1 = ['district_code', 'centx', 'centy', 'cropland_pct', 'pop',
           'ruggedness_mean', 'pasture_pct', 'ipc_months_since_change',
           'ipc_lag_1', 'food_price_idx_lag_1', 'ipc_lag_3', 'ndvi_mean_lag_3',
           'rain_mean_lag_3', 'et_mean_lag_3', 'food_price_idx_lag_3', 'ipc_lag_6',
           'food_price_idx_lag_6', 'ipc_rolling_avg_3',
           'food_price_idx_rolling_avg_3', 'food_price_idx_rolling_std_3', 'nlp_1', 'nlp_2', 'nlp_3', 'nlp_4', 'nlp_5', 'nlp_6', 'nlp_7']
    
    custom_feature_list2 = ['district_code', 'ipc_months_since_change',
           'ipc_lag_1', 'ipc_lag_3', 
           'ipc_lag_6',
           'nlp_1', 'nlp_2', 'nlp_3', 'nlp_4', 'nlp_5', 'nlp_6', 'nlp_7']
    
    custom_feature_list3 = ['district_code', 'ipc_months_since_change',
           'ipc_lag_1', 'ipc_lag_3', 'food_price_idx_lag_6', 'food_price_idx_rolling_avg_3', 'food_price_idx_rolling_std_3', 
           'ipc_lag_6','food_price_idx_lag_1','food_price_idx_lag_3',
           'nlp_1', 'nlp_2', 'nlp_3', 'nlp_4', 'nlp_5', 'nlp_6', 'nlp_7']
    
    custom_feature_list4 = ['district_code', 'ipc_months_since_change',
           'ipc_lag_1', 'ipc_lag_3', 'food_price_idx_lag_6', 'food_price_idx_rolling_avg_3', 'food_price_idx_rolling_std_3', 
           'ipc_lag_6','food_price_idx_lag_1','food_price_idx_lag_3','ndvi_mean_lag_3',
           'nlp_1', 'nlp_2', 'nlp_3', 'nlp_4', 'nlp_5', 'nlp_6', 'nlp_7']
    X_train = X_train[custom_feature_list4]
    X_test = X_test[custom_feature_list4]
    #X_train = sel.transform(X_train)
    #X_test = sel.transform(X_test)
    print(X_train.shape)
    print(X_train.shape)

    print('kfold mean accuracy')
    print(np.mean(cross_val_score(rf, X_train, y_train, cv=10)))
    
    rf.fit(X_train, y_train)
    # Make predictions
    y_pred = rf.predict(X_test)
    y_pred_prob = rf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Calculate and print performance metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro')}")
    print(f"Precision (Micro): {precision_score(y_test, y_pred, average='micro')}")
    print(f"Precision (Weighted): {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall (Macro): {recall_score(y_test, y_pred, average='macro')}")
    print(f"Recall (Micro): {recall_score(y_test, y_pred, average='micro')}")
    print(f"Recall (Weighted): {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1 Score (Macro): {f1_score(y_test, y_pred, average='macro')}")
    print(f"F1 Score (Micro): {f1_score(y_test, y_pred, average='micro')}")
    print(f"F1 Score (Weighted): {f1_score(y_test, y_pred, average='weighted')}")

    # Only compute ROC AUC for binary classification tasks
    if len(set(y_test)) == 2:
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_prob)}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    #print(f"\nLog Loss: {log_loss(y_test, y_pred_prob)}")

    importances = rf.feature_importances_
    indices = np.argsort(importances)
    features = X.columns
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

    return rf

