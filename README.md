# Thesis Project: Portfolio Selection and Optimization with Machine Learning in Tehran Exchange
### * A Sustainable Portfolio Optimization and Feature Selection Approach *

## Overview

Welcome to the GitHub repository for my thesis project on portfolio optimization using machine learning techniques. This project focuses on constructing an optimized investment portfolio based on historical stock data and leveraging predictive models.

## Project Structure

The project is organized into several key steps, each contributing to the overall goal of creating an efficient and risk-adjusted investment portfolio.

### Step 1: Feature Selection with Random Forest and Lasso Regression

- **Objective:**
  - Classify stock returns into categories using two different methods: Random Forest and Lasso Regression combined with Logistic Regression.
  - Select significant features for each method.

- **Implementation:**
  - Python script: `feature_selection.py`
  - Output: Excel files with feature selection results for Random Forest and Lasso Regression.

### Step 2: Stock Classification with Multiple Methods

- **Objective:**
  - Classify stock returns into categories using four different methods: Random Forest, Support Vector Machine (SVM), Artificial Neural Network (ANN), and XGBoost.
  - Evaluate classification performance.

- **Implementation:**
  - Python script: `stock_classification_multiple_methods.py`
  - Output: Excel files with classification results for each method.

### Step 3: Monthly Portfolio Optimization

- **Objective:**
  - Create monthly optimized portfolios based on stock classifications.
  - Implement risk management strategies.

- **Implementation:**
  - Python script: `monthly_portfolio_optimization.py`
  - Output: JSON files with optimized portfolio weights and performance metrics.

### Step 4: Sharpe Ratio-Based Weight Optimization

- **Objective:**
  - Maximize the Sharpe ratio for portfolio optimization.
  - Incorporate risk-free rates into the optimization process.

- **Implementation:**
  - Python script: `sharpe_ratio_optimization.py`
  - Output: JSON files with optimized portfolio weights and Sharpe ratios.

## Getting Started

### Prerequisites

- Python (3.x)
- Required Python packages (NumPy, Pandas, Scikit-Learn, etc.)

### Execution

1. Clone the repository: `git clone https://github.com/aminmoghadasi75/M.A.Thesis.git`
2. Navigate to the project directory.
3. Execute the desired Python scripts for each step.

## Results and Documentation

Explore the respective folders for detailed results, code, and additional documentation related to each step.

Feel free to reach out if you have questions or suggestions!

Happy exploring and investing!
