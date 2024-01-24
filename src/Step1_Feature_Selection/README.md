# Thesis Project Repository

## Overview

Welcome to the repository for the thesis project! This project focuses on the classification of financial returns and feature selection using machine learning techniques. The provided code represents the first step of the thesis, where historical financial data is processed, features are selected, and a logistic regression model is trained for classification.

## Code Structure

The code is organized into a Python script (`thesis_step1.py`). Here is an overview of the main components:

### Libraries
The necessary libraries and modules are imported, including scikit-learn for machine learning tasks and pandas for data manipulation.

### Data Processing Functions
The script contains functions for classifying return values and processing data groups. The primary function (`process_group`) performs feature scaling, selects features using Lasso regression, and trains a logistic regression model.

### Main Loop
The script iterates through financial data files, groups data by time periods, and applies the data processing function to generate evaluation metrics and selected features.

### Results Storage
The final results, including accuracy, precision, recall, F1 score, and selected features, are stored in a dictionary and saved to an Excel file.

## Instructions

### Running the Code

1. Ensure you have Python installed on your system.
2. Install the required Python libraries by running: `pip install -r requirements.txt`
3. First Method Feature Selection : Execute the script: `python Lasso_LogReg_Feature_selection.py`
3. Second Method Feature Selection : Execute the script: `python RF_feature_selection.py`

### Folder Structure

- **/data:** This folder should contain the financial data files in Excel format.

- **/results:** The results of the analysis, including evaluation metrics and selected features, will be saved in this folder.

## Results

The results of the analysis for each time period are saved in Excel files within the `/results` folder. These files provide insights into the performance of the logistic regression model and the importance of selected features.

## Next Steps

This is the first step in the thesis project. Future steps may involve refining the model, exploring additional features, and conducting a more in-depth analysis of the financial data.

Feel free to explore the code and results, and don't hesitate to reach out if you have any questions or suggestions!
