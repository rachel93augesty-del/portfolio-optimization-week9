# =========================
# utils.py - helper functions for Task 2
# =========================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -------------------------
# Chronological train/test split
# -------------------------
def train_test_split_time(df: pd.DataFrame, target_col: str, train_end_date: str):
    """
    Split dataframe into training and testing sets chronologically.
    
    Parameters:
    - df: long-format DataFrame
    - target_col: column to forecast (e.g., 'Close')
    - train_end_date: last date of training set (YYYY-MM-DD)
    
    Returns:
    - train: training DataFrame
    - test: testing DataFrame
    """
    df_sorted = df.sort_values("Date")
    train = df_sorted[df_sorted["Date"] <= train_end_date][[ "Date", target_col]].copy()
    test = df_sorted[df_sorted["Date"] > train_end_date][[ "Date", target_col]].copy()
    return train, test

# -------------------------
# Scale data for LSTM
# -------------------------
def scale_data(train: pd.DataFrame, test: pd.DataFrame, col: str):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train[[col]])
    test_scaled = scaler.transform(test[[col]])
    return train_scaled, test_scaled, scaler

# -------------------------
# Create LSTM sequences
# -------------------------
def create_lstm_sequences(data, seq_length=60):
    """
    Convert a time series array into sequences for LSTM.
    Returns X, y arrays.
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# -------------------------
# Plot forecast vs actual
# -------------------------
def plot_forecast(actual, predicted, dates, title="Forecast vs Actual"):
    plt.figure(figsize=(14,6))
    plt.plot(dates, actual, label="Actual", color="blue")
    plt.plot(dates, predicted, label="Predicted", color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
