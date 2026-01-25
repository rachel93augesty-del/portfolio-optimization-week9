Portfolio Optimization & Forecasting Project
Overview

This repository contains a structured workflow for analyzing and forecasting financial time series data for selected assets, with a focus on Tesla (TSLA). The project is organized using modular Python scripts in the src/ folder and complemented with Jupyter notebooks for interactive exploration and analysis.

The project is divided into two main tasks:

Task 1: Data Cleaning & Exploratory Data Analysis (EDA)

Objective: Prepare clean, long-format financial time series data and extract insights through visualization and basic statistical analysis.

Key Steps:

Data Preprocessing

Load raw asset data from CSV.

Handle missing values using forward/backward fill.

Ensure proper data types (Date as datetime, numeric columns as float).

Reshape from wide to long format to standardize multiple tickers.

Exploratory Data Analysis

Visualize closing prices over time for all tickers.

Calculate and plot daily returns (%) to assess volatility.

Analyze rolling mean and rolling standard deviation for short-term trends.

Perform outlier detection on daily returns to identify anomalous days.

Seasonality & Trend Analysis

Conduct Augmented Dickey-Fuller (ADF) tests on closing prices and daily returns.

Interpret results to identify stationary vs non-stationary series.

Non-stationary series may require differencing before applying time series models (e.g., ARIMA).

Risk Metrics

Calculate Value at Risk (VaR, 95%).

Compute Sharpe ratios (daily and annualized).

Summarize insights on the volatility and performance of each asset.

Deliverables:

eda.ipynb — Notebook containing full EDA, visualizations, and risk metrics.

eda.py — Modular Python script with reusable EDA functions.

Summary of data quality issues and resolution.

At least 3 key visualizations capturing trends, volatility, and outliers.

Task 2: Time Series Forecasting Models

Objective: Develop, train, and evaluate time series forecasting models to predict Tesla's stock prices, using both classical statistical methods and deep learning.

Key Steps:

Data Preparation

Chronologically split data into train and test sets (e.g., train: 2015–2024, test: 2025–2026).

Ensure no random shuffling, preserving temporal order.

ARIMA / SARIMA Modeling

Identify optimal (p,d,q) parameters using ACF/PACF plots or auto_arima.

Fit the ARIMA/SARIMA model on training data.

Generate forecasts for the test period.

LSTM Modeling

Scale data and prepare sequence datasets (e.g., last 60 days → next day).

Build LSTM architecture (input layer, one or more LSTM layers, dense output).

Train with optimized hyperparameters (epochs, batch size).

Generate forecasts for the test period.

Model Evaluation & Comparison

Compute performance metrics: MAE, RMSE, MAPE.

Compare ARIMA and LSTM models.

Discuss which model performs better and possible reasons.

Deliverables:

modeling.py — Python script for ARIMA and LSTM modeling.

utils.py — Helper functions for train/test split, scaling, sequence generation, and plotting.

evaluation.py — Functions to calculate and compare model performance metrics.

task2_forecasting.ipynb — Notebook demonstrating step-by-step workflow, forecasts, and performance analysis.

Forecast visualizations comparing predicted vs actual prices.

Project Structure
portfolio-optimization-week9/
│
├─ data/
│   ├─ raw/                  # Raw CSV files
│   └─ processed/            # Cleaned and reshaped data
│
├─ notebooks/
│   ├─ eda.ipynb             # Task 1 notebook
│   └─ task2_forecasting.ipynb # Task 2 notebook
│
├─ src/
│   ├─ data_cleaning.py      # Data preprocessing & reshaping
│   ├─ eda.py                # EDA functions
│   ├─ modeling.py           # Forecasting model functions (ARIMA/LSTM)
│   ├─ utils.py              # Helper functions for Task 2
│   └─ evaluation.py         # Metrics evaluation functions
│
├─ README.md                 # Project overview
└─ environment.yml / requirements.txt  # Package dependencies

Environment / Dependencies

Task 1 Required Packages:

pandas, numpy, yfinance, matplotlib, seaborn, scipy, statsmodels, pytest, jupyter


Task 2 Required Packages (in addition to Task 1):

pmdarima, tensorflow, keras, scikit-learn


You can create a conda environment with all dependencies via environment.yml or requirements.txt.

Notes / Insights

EDA (Task 1): Tesla has the highest daily return volatility compared to other assets (SPY, BND). Daily returns are stationary, while price series are not — requiring differencing for ARIMA models.

Forecasting (Task 2): ARIMA provides interpretable forecasts, while LSTM captures complex temporal patterns. Performance metrics (MAE, RMSE, MAPE) guide model selection.