# Thesis Project - Step 2

## Introduction

This script represents the second step in the thesis project, focusing on the classification of financial returns using a Random Forest classifier and 3 different classification algorithms. The goal is to predict the categorical class of financial returns based on historical data and selected features from the previous step.

## Code Overview

### Functions

#### `return_classifier(df, last_month, monthly_features, inx) -> tuple`

- **Input:**
  - `df`: DataFrame containing historical financial data.
  - `last_month`: DataFrame containing data for the last month.
  - `monthly_features`: List of selected features for the current month.
  - `inx`: Index indicating the current month.

- **Output:**
  - Tuple containing accuracy, precision, recall, F1 score, and predictions for the next month.

#### `feature_cleaner(df, inx) -> List[str]`

- **Input:**
  - `df`: DataFrame containing feature information.
  - `inx`: Index indicating the current feature set.

- **Output:**
  - List of cleaned features.

#### `classify_return(return_value) -> int`

- **Input:**
  - `return_value`: Numeric return value.

- **Output:**
  - Integer representing the class of the return value.

### Main Loop

The script iterates through financial data files (`df_file_list`) and corresponding feature files (`feature_file_list`). For each file pair, it reads the data and features, preprocesses the data, and performs classification using a Random Forest classifier.

- **Data Preprocessing:**
  - Resamples data on a monthly basis.
  - Calculates percentage price change.
  - Adds a return class based on the `classify_return` function.

- **Classification:**
  - Utilizes the `return_classifier` function to train a Random Forest model and predict the next month's return classes.
  - Conducts a grid search over various hyperparameters to find the best model.

- **Results Storage:**
  - Saves the results, including accuracy, precision, recall, F1 score, and predictions for the next month, in an Excel file.

## Instructions

### Running the Code

1. Ensure you have Python installed on your system.
2. Install the required Python libraries by running: `pip install -r requirements.txt`
3. Execute the script: `python step2_classification.py`

### Folder Structure

- **/data:** This folder should contain the financial data files in Excel format.
- **/features:** The selected features from the previous step.
- **/results:** The results of the classification analysis will be saved in this folder.

## Results

The results of the classification analysis, including evaluation metrics and predictions for the next month, are saved in Excel files within the `/results/RF` folder.

## Next Steps

This step focuses on predicting return classes for the next month using a Random Forest classifier. Subsequent steps may involve further model refinement, feature engineering, and a deeper analysis of the classification results.

Feel free to explore the code and results, and don't hesitate to reach out if you have any questions or suggestions!
