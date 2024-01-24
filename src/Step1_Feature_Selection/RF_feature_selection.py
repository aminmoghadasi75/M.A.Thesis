import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score,precision_recall_fscore_support
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


def process_group(old_data):

    label_encoder = LabelEncoder()
    Return_Class_old = old_data['پایانی*سهم*'].pct_change() * 100
    old_data['Return_Class'] = Return_Class_old.apply(classify_return)

    old_data['Return_Class'] = label_encoder.fit_transform(old_data['Return_Class'])
    old_data = old_data.dropna(subset=['Return_Class'])


    scaling = StandardScaler()
    # Select numeric columns except the target column
    numeric_columns = old_data.select_dtypes(include=[np.number]).drop(columns=['Return_Class']).columns.tolist()

    # Extract the subset of data with only numeric columns
    old_num_data = old_data[numeric_columns]



    y_old = old_data['Return_Class']
    X_old = scaling.fit_transform(old_num_data)





    # Train-test split for old and new data separately
    X_old_train, X_old_test, y_old_train, y_old_test = train_test_split(X_old, y_old, test_size=0.2)





    model = RandomForestClassifier()
    # Fit the model
    model.fit(X_old_train, y_old_train)




    # Predict on the test set (either old or new)
    y_pred = model.predict(X_old_test)

    # Evaluation metrics for classification
    accuracy = accuracy_score(y_old_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_old_test, y_pred, average='weighted')

    # Retrieve feature importances from the classifier
    feature_importance = model.feature_importances_

    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    cumulative_importance = 0.0
    selected_features = []
    for idx in sorted_indices:
        cumulative_importance += feature_importance[idx]
        selected_features.append(numeric_columns[idx])
        if cumulative_importance >= 0.8:
            break

    return accuracy, precision, recall, f1, selected_features

folder_path = '/home/amin/thesis/src/data/MAIN-FILES/FINNAL_DATA'
file_list = os.listdir(folder_path)
final_result = { }

for file in file_list:
    file_path = f'{folder_path}/{file}'
    data = pd.read_excel(file_path)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(method='ffill', inplace=True)
    old_data = None
    for (year, month), new_data in data.groupby([data['Date'].dt.year, data['Date'].dt.month]):
        mask = (data['Date'].dt.year == year) & (data['Date'].dt.month == month)
        filtered_data = data[mask]
        now_df = pd.DataFrame(filtered_data)


        if old_data is None:
            old_data = now_df

        else:
            mask2 = ((data['Date'].dt.year == year) & (data['Date'].dt.month < month)) | ((data['Date'].dt.year < year))
            old_data_filtered = old_data[mask2]
            old_data = pd.concat([old_data_filtered, now_df])
        print(f'wonk on {file} for period : {year}-{month} ...')
        scores = process_group(old_data.copy())  # Ensure copies to prevent data contamination

        accuracy, precision, recall, f1, selected_features = scores
        final_result[f'{year}-{month}'] = {'Accuracy': accuracy,
                                            'Precision': precision,
                                            'Recall' : recall,
                                            'F1': f1,
                                            'Selected Features': selected_features
                                            }
        print(f'Accuracy: {accuracy * 100:.2f}%')
    df = pd.DataFrame(final_result)
    df.to_excel(f'/home/amin/thesis/src/Step/Feature_selection_RF/RFC_{file}', index=True)





