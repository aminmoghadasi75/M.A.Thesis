import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, make_scorer,
                             precision_recall_fscore_support, precision_score, f1_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  # MODEL

warnings.filterwarnings('ignore')

def return_classifier(df: pd.DataFrame, last_month: pd.DataFrame, monthly_features: list, inx: int) -> tuple:
    scaling = StandardScaler()

    numeric_columns = df.select_dtypes(include=[np.number]).drop(columns=['Target']).columns.tolist()
    numeric_columns_last_month = last_month.select_dtypes(include=[np.number]).drop(columns=['Target']).columns.tolist()

    X_df = df[numeric_columns]
    y_df = df['Target']
    last_month_num = last_month[numeric_columns_last_month]

    my_list = feature_cleaner(monthly_features, inx)
    imp_columns = [item for item in my_list if r'\u200' not in item]

    X_df = X_df[imp_columns]
    X_last_month = last_month_num[imp_columns]

    X_df = scaling.fit_transform(X_df)
    X_last_month = scaling.transform(X_last_month)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2)
    
    # Scoring functions for accuracy and precision
    scoring = {
        'f1_weighted': make_scorer(f1_score, average='weighted')
    }

    # XGBoost parameters for grid search
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001]
    }

    xgb = XGBClassifier()
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring=scoring, refit='f1_weighted', n_jobs=-1)

    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_

    best_xgb = XGBClassifier(**best_params)
    best_xgb.fit(X_train, y_train)

    y_pred = best_xgb.predict(X_test)

    next_month_prediction = best_xgb.predict(X_last_month)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1, next_month_prediction


def feature_cleaner(df: pd.DataFrame, inx: int) -> List[str]:
    """
	This function takes in a DataFrame `df` and an index `inx` as parameters.

	It cleans the features in the DataFrame at the specified index by removing square brackets,
	replacing single quotes with empty strings, and removing leading or trailing spaces.

	If the feature 'پایانی*سهم*' is not already present in the list of cleaned features, it is added.

	The function returns the cleaned list of features.
	"""
    feat_imp = df.Features[inx].replace("[", "").replace("]", "")
    feat_imp = feat_imp.split(",")
    list_i = [ ]
    for item in feat_imp:
        list_i.append(item.replace("'", ""))
    list2 = [ ]
    for item in list_i :
        if item.startswith(' ') :
            list2.append(item[1: ])
        elif item.endswith(' ') :
            list2.append(item[:-1])
        else :
            list2.append(item)
    if 'پایانی*سهم*' not in list2 :
        list2.append('پایانی*سهم*')

    return list2

def classify_return(return_value: float) -> int:
        """
        Classifies a given return value into different categories based on its magnitude.

        Parameters:
            return_value (numeric): The value to be classified.

        Returns:
            str: The category of the given return value. Possible categories are:
                - '3' for return values greater than 5.
                - '2' for return values between 2.5 and 5 (inclusive).
                - '1' for return values between 0 and 2.5 (inclusive).
                - '-1' for return values between -5 and 0 (inclusive).
                - '-2' for return values less than -5.
        """
        if return_value >= 0:
            return 1
        elif  return_value < 0:
            return 0




data_folder_path = '/home/amin/thesis/src/data/MAIN-FILES/FINNAL_DATA'
feature_folder_path = '/home/amin/thesis/src/Step/Feature_selection_RF'
df_file_list = os.listdir(data_folder_path)
feature_file_list = os.listdir(feature_folder_path)

for file in df_file_list:
    final_result = { }
    for feature_file in feature_file_list:

        if feature_file == f'RFC_{file}' :
            print(feature_file)

            data = pd.read_excel(f'{data_folder_path}/{file}')
            feature = pd.read_excel(f'{feature_folder_path}/{feature_file}')

            # Preprocessing steps
            data.drop(columns=['تاریخ شمسی'], inplace=True)
            df = data.resample('M', on='Date').mean()
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])

            # Select monthly features
            monthly_features = pd.DataFrame(feature.iloc[-1])
            monthly_features = monthly_features.rename(columns={3: 'Features'})
            monthly_features.reset_index(inplace=True)

            # Calculate percentage price change and add return class
            pct_price = df['پایانی*سهم*'].pct_change() * 100
            df['Return_Class'] = pct_price.apply(classify_return)
            df['Shifted_Return_Class'] = df['Return_Class'].shift(periods=1).fillna(1)
            df.rename(columns={'Shifted_Return_Class' : 'Target'}, inplace=True)
            df.drop('Return_Class', axis=1, inplace=True)


            for inx in range(20 , len(df)) :
                new_df = df.iloc[ : inx - 1 ]
                last_month = df.iloc[ inx : inx + 1]


                # Perform return classification and store results
                accuracy, precision, recall, f1, next_month_prediction = return_classifier(df=new_df, last_month= last_month , monthly_features= monthly_features, inx= inx)
                final_result[f"{monthly_features['index'][inx]}"] = {'Accuracy': accuracy,'Precision': precision,'Recall': recall,'F1': f1 ,'Next_Month_Prediction' : next_month_prediction}



            final_result = pd.DataFrame(final_result, index=['Accuracy', 'Precision', 'Recall', 'F1', 'Next_Month_Prediction'])
            final_result.to_excel(f'/home/amin/thesis/src/Step2_Classification/XGB/XGB_{file}', index=True)







