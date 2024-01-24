# Thesis Project - Step 4.2 - Portfolio Optimization
## Introduction

This script represents the final step in the thesis project, focusing on portfolio optimization. The optimization process involves calculating optimal weights for each stock in the portfolio to maximize the Sharpe ratio. The optimized portfolio weights are applied to the previously constructed portfolio, enhancing the overall performance and risk-adjusted returns.

## Code Overview

The script reads two JSON files: `my__FINAL_portfolio_return_LassoLogRegRF.json` contains the original portfolio returns, and `OptimizedPortfolio_weight_LassoLogRegRf.json` contains the optimized weights calculated to maximize the Sharpe ratio.

### Processing Steps

1. **Loading Portfolio Returns:**
   - The script loads the original portfolio returns from `my__FINAL_portfolio_return_LassoLogRegRF.json`.

2. **Loading Optimized Weights:**
   - The script loads the optimized weights from `OptimizedPortfolio_weight_LassoLogRegRf.json`.

3. **Calculating Optimized Portfolio Returns:**
   - For each date in the portfolio, the script matches the corresponding date in the optimized weights file.
   - It then applies the calculated weights to the stocks in the portfolio, calculating the portfolio return.

4. **Updating DataFrame:**
   - The script updates a DataFrame with the calculated optimized portfolio returns.

## Instructions

### Running the Code

1. Ensure you have Python installed on your system.
2. Execute the script: `python step4_portfolio_optimization.py`

### File Paths

- **Input Files:**
  - `./Set_StopLoss/my__FINAL_portfolio_return_LassoLogRegRF.json`
  - `/home/amin/thesis/src/Step4_Portfolio_Optimization/StockWeight_SharpRatio/OptimizedPortfolio_weight_LassoLogRegRf.json`

## Results

The script calculates optimized portfolio returns based on the previously constructed portfolio and the optimized weights to maximize the Sharpe ratio. The results are stored in the DataFrame.

## Next Steps

The final step involves evaluating the performance of the optimized portfolio, conducting risk analysis, and presenting the findings. Further steps may include refining the optimization strategy based on additional factors.

Feel free to explore the code and results, and don't hesitate to reach out if you have any questions or suggestions!
