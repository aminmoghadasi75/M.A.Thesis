import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.compose import ColumnTransformer #This modification separates the encoding step for categorical columns using LabelEncoder within the ColumnTransformer.

warnings.filterwarnings('ignore')

# Function to classify return
def classify_return(return_value):
    """
    Classify the given return value into different categories based on the value range.

    Parameters:
        return_value (float): The value to be classified.

    Returns:
        str: The category that the return value belongs to. Possible categories are:
             - 'Above +5' if the value is greater than 5
             - '+2.5 to +5' if the value is between 2.5 and 5 (inclusive)
             - '+0 to +2.5' if the value is between 0 and 2.5 (inclusive)
             - '0 to -5' if the value is between -5 and 0 (inclusive)
             - 'Below -5' if the value is less than -5
    """
    if return_value > 5:
        return 'Above +5'
    elif 2.5 < return_value <= 5:
        return '+2.5 to +5'
    elif 0 <= return_value <= 2.5:
        return '+0 to +2.5'
    elif -5 <= return_value < 0:
        return '0 to -5'
    else:
        return 'Below -5'


def process_group(old_data, new_data):

    label_encoder = LabelEncoder()
    Return_Class_old = old_data['پایانی*سهم*'].pct_change() * 100
    old_data['Return_Class'] = Return_Class_old.apply(classify_return)

    old_data['Return_Class'] = label_encoder.fit_transform(old_data['Return_Class'])
    old_data = old_data.dropna(subset=['Return_Class'])

    Return_Class_new = new_data['پایانی*سهم*'].pct_change() * 100
    new_data['Return_Class'] = Return_Class_new.apply(classify_return)

    new_data['Return_Class'] = label_encoder.fit_transform(new_data['Return_Class'])
    new_data = new_data.dropna(subset=['Return_Class'])

    scaling = StandardScaler()
    # Select numeric columns except the target column
    numeric_columns = old_data.select_dtypes(include=[np.number]).drop(columns=['Return_Class']).columns.tolist()

    # Extract the subset of data with only numeric columns
    old_num_data = old_data[numeric_columns]

    

    y_old = old_data['Return_Class']
    X_old = scaling.fit_transform(old_num_data)


    # Select numeric columns except the target column
    numeric_columns = new_data.select_dtypes(include=[np.number]).drop(columns=['Return_Class']).columns.tolist()

    # Extract the subset of data with only numeric columns
    new_num_data = new_data[numeric_columns]


    y_new = new_data['Return_Class']
    X_new = scaling.fit_transform(new_num_data)


    # Train-test split for old and new data separately
    X_old_train, X_old_test, y_old_train, y_old_test = train_test_split(X_old, y_old, test_size=0.2)
    X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_new, test_size=0.2)

    # Fit the model on a combination of old and new data with 50% weightage
    X_train_combined = pd.concat([pd.DataFrame(X_old_train), pd.DataFrame(X_new_train)])
    y_train_combined = pd.concat([pd.DataFrame(y_old_train), pd.DataFrame(y_new_train)])
    sample_weights = [0.5] * len(X_old_train) + [0.5] * len(X_new_train)  # 50% weight for old and new

    model = RandomForestClassifier()
    # Fit the model
    model.fit(X_train_combined, y_train_combined, randomforestclassifier__sample_weight=sample_weights)


    # Preparing prediction part
    X_test_combined = pd.concat([pd.DataFrame(X_old_test), pd.DataFrame(X_new_test)])
    y_test_combined = pd.concat([pd.DataFrame(y_old_test), pd.DataFrame(y_new_test)])

    # Predict on the test set (either old or new)
    y_pred = model.predict(X_test_combined)

    # Evaluation metrics for classification
    accuracy = accuracy_score(y_test_combined, y_pred)
    confusion = confusion_matrix(y_test_combined, y_pred)
    report = classification_report(y_test_combined, y_pred)

    # Retrieve feature importances from the classifier
    feature_importance = model.named_steps['randomforestclassifier'].feature_importances_

    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    cumulative_importance = 0.0
    selected_features = []
    for idx in sorted_indices:
        cumulative_importance += feature_importance[idx]
        selected_features.append(X_old.columns[idx])
        if cumulative_importance >= 0.8:
            break

    return accuracy, confusion, report, selected_features


data = pd.read_excel('/home/amin/thesis/src/data/MAIN-FILES/FINNAL_DATA/فولاد.xlsx')




old_data = None

for (year, month), new_data in data.groupby([data['Date'].dt.year, data['Date'].dt.month]):
    mask = (data['Date'].dt.year == year) & (data['Date'].dt.month == month)
    filtered_data = data[mask]
    now_df = pd.DataFrame(filtered_data)
    if old_data is None:
        old_data = now_df
    else:
        mask2 = (data['Date'].dt.year <= year) & (data['Date'].dt.month <= month)
        old_data = data[mask2]

    print(f'{year}-{month} ...')
    # Process old and new data
    scores = process_group(old_data.copy(), now_df.copy())  # Ensure copies to prevent data contamination

    accuracy, confusion, report, selected_features = scores

    print(f'{year}-{month}')
    print(f'accuracy = {accuracy}')
    print(f'confusion_matrix = {confusion}')
    print(f'report = {report}')
    print(f'selected Features = {selected_features}')

