# Thesis Project - Step 3

## Introduction

This script represents the third step in the thesis project, focusing on post-processing the results from the Random Forest classification. The objective is to create a portfolio based on predicted returns and precision values, providing insights into potential investment opportunities.

## Code Overview

The script begins by reading the result files from the Random Forest classification step, located in the `./Final_result/RF-RF` folder. It then processes the data to create two dataframes, `RfRf_next_month` and `RfRf_precision`, representing predicted returns and precision values for each month.

### Processing Steps

1. **Reading Results:**
   - The script reads the result files from the specified folder path (`./Final_result/RF-RF`).

2. **Dataframe Initialization:**
   - Two dataframes, `RfRf_next_month` and `RfRf_precision`, are initialized to store predicted returns and precision values.

3. **Data Merging:**
   - The script iterates through the result files, processes the data using preprocessing functions (`preprossing_return` and `preprossing_precision`), and merges the results into the respective dataframes.

4. **Data Truncation:**
   - To align the data properly, the dataframes are truncated to contain the first 96 rows.

5. **Index Setting:**
   - The date column is set as the index for both dataframes.

6. **Stock Pool Creation:**
   - The script creates a stock pool dictionary based on positive predicted returns for each month.

7. **Portfolio Construction:**
   - Utilizing the stock pool, the script constructs a portfolio for each month based on the top-performing stocks in terms of precision values.

8. **Portfolio Saving:**
   - The final portfolio is saved as a JSON file named `my_portfolio_RFRF.json`.

## Instructions

### Running the Code

1. Ensure you have Python installed on your system.
2. Execute the script: `python step3_portfolio_construction.py`

### Folder Structure

- **/Final_result/RF-RF:** This folder should contain the result files from the Random Forest classification step.

## Results

The script generates a portfolio based on predicted returns and precision values. The resulting portfolio is saved in the file `my_portfolio_RFRF.json`.

## Next Steps

This step involves creating a portfolio based on the Random Forest classification results. Subsequent steps may include portfolio analysis, performance evaluation, and refining the portfolio construction strategy.

Feel free to explore the code and results, and don't hesitate to reach out if you have any questions or suggestions!
