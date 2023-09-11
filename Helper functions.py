import pandas as pd
from feature_engine.creation import CyclicalFeatures

def create_features(df, feature_name='theft counts', lags=1, window=3, apply_cyclical=True):
    """
    Create time series features based on time series index.

    Parameters:
    - df: Dataframe containing time series data.
    - feature_name: Name of the feature to create lags and rolling average for.
    - lags: Number of lags to be created for the specified feature.
    - window: Window size for rolling average of the specified feature.
    - apply_cyclical: Whether to apply the CyclicalFeatures transformation.

    Returns:
    - Dataframe with new features.
    """
    # Convert the index to a datetime format
    df.index = pd.to_datetime(df.index)
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year

    # Add the lags for specified feature
    for lag in range(1, lags + 1):
        col_name = f'{feature_name}_lag{lag}'
        df[col_name] = df[feature_name].shift(lag)

    # Add the moving average feature for the specified feature
    df[f'{feature_name}_moving_avg{window}'] = df[feature_name].rolling(window=window).mean()

    if apply_cyclical:
        # Assuming you've imported CyclicalFeatures class, otherwise you need to do so.
        cyclical = CyclicalFeatures(
            variables=["month"],  # The features we want to transform.
            drop_original=False,  # Whether to drop the original features.
        )
        df = cyclical.fit_transform(df)

    return df


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix, log_loss)
from sklearn.model_selection import train_test_split


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

    # Initialize and train the Random Forest classifier
    rf = RandomForestClassifier(random_state=random_state)
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

    print(f"\nLog Loss: {log_loss(y_test, y_pred_prob)}")

    return rf

