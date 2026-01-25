# =========================
# evaluation.py - model metrics
# =========================

import numpy as np
import pandas as pd

# -------------------------
# Mean Absolute Error
# -------------------------
def compute_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

# -------------------------
# Root Mean Squared Error
# -------------------------
def compute_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

# -------------------------
# Mean Absolute Percentage Error
# -------------------------
def compute_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted)/actual)) * 100

# -------------------------
# Comparison table
# -------------------------
def compare_models(results_dict):
    """
    results_dict = {
        'ARIMA': {'MAE': x, 'RMSE': y, 'MAPE': z},
        'LSTM': {'MAE': x, 'RMSE': y, 'MAPE': z},
        ...
    }
    Returns a pandas DataFrame
    """
    df = pd.DataFrame(results_dict).T
    return df
