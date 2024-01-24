# Thesis Project - Step 4.1: Maximizing the Sharpe Ratio

## Introduction

This script represents the fourth step in the thesis project, focusing on portfolio optimization. The objective is to calculate optimal weights for each stock in the portfolio, maximizing the Sharpe ratio. The optimization process utilizes predicted returns and constraints to ensure a balanced and efficient portfolio.

## Code Overview

The script reads the portfolio returns from a previously generated JSON file (`my__FINAL_portfolio_return_LassoLogRegANN.json`). It then calculates optimal portfolio weights by maximizing the Sharpe ratio, considering constraints such as maximum weight and the risk-free rate.

### Processing Steps

1. **Loading Portfolio Returns:**
   - The script reads the portfolio returns from the JSON file.

2. **Predicting Next Month Returns:**
   - For each stock in the portfolio, it predicts the next month's return based on a pre-trained model.

3. **Calculating Covariance Matrix:**
   - The script calculates the covariance matrix based on historical returns.

4. **Optimizing Portfolio Weights:**
   - It formulates the optimization problem to maximize the Sharpe ratio, considering constraints on weights.
   - The optimization process uses the SciPy library's `minimize` function.

5. **Results and Output:**
   - The optimal weights and corresponding Sharpe ratio are displayed for each optimization date.
   - The final optimized portfolio weights and Sharpe ratios are saved in JSON files (`OptimizedPortfolio_weight_LassoLogRegAnn.json` and `Sharpe_Ratio_Portfolio_LassoLogRegAnn.json`).

## Instructions

### Running the Code

1. Ensure you have Python installed on your system.
2. Execute the script: `python portfolio_optimization.py`

### File Paths

- **Input Files:**
  - `/home/amin/thesis/src/Step4_Portfolio_Optimization/Set_StopLoss/my__FINAL_portfolio_return_LassoLogRegANN.json`

- **Output Files:**
  - `/home/amin/thesis/src/Step4_Portfolio_Optimization/StockWeight_SharpRatio/OptimizedPortfolio_weight_LassoLogRegAnn.json`
  - `/home/amin/thesis/src/Step4_Portfolio_Optimization/StockWeight_SharpRatio/Sharpe_Ratio_Portfolio_LassoLogRegAnn.json`

## Results

The script calculates optimal portfolio weights that maximize the Sharpe ratio, ensuring an efficient and risk-adjusted allocation of assets in the portfolio.

Feel free to explore the code and results, and don't hesitate to reach out if you have any questions or suggestions!
