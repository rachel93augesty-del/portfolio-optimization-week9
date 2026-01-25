# =========================
# EDA Module
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# -------------------------
# 1. Plot Closing Prices
# -------------------------
def plot_closing_prices(df: pd.DataFrame):
    plt.figure(figsize=(14,6))
    for ticker in df["Ticker"].unique():
        tmp = df[df["Ticker"] == ticker]
        plt.plot(tmp["Date"], tmp["Close"], label=ticker)
    plt.title("Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

# -------------------------
# 2. Calculate Daily Returns (%)
# -------------------------
def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change() * 100
    return df

# -------------------------
# 3. Plot Daily Returns
# -------------------------
def plot_daily_returns(df: pd.DataFrame):
    plt.figure(figsize=(14,6))
    for ticker in df["Ticker"].unique():
        tmp = df[df["Ticker"] == ticker]
        plt.plot(tmp["Date"], tmp["Daily_Return"], label=ticker, alpha=0.6)
    plt.title("Daily Returns (%) Over Time")
    plt.xlabel("Date")
    plt.ylabel("Daily Return (%)")
    plt.legend()
    plt.show()

# -------------------------
# 4. Rolling Mean & Volatility
# -------------------------
def plot_rolling_stats(df: pd.DataFrame, window=20):
    plt.figure(figsize=(14,6))
    for ticker in df["Ticker"].unique():
        tmp = df[df["Ticker"] == ticker].copy()
        tmp["Rolling_Std"] = tmp["Close"].rolling(window).std()
        tmp["Rolling_Mean"] = tmp["Close"].rolling(window).mean()
        plt.plot(tmp["Date"], tmp["Rolling_Std"], label=f"{ticker} Rolling Std")
        plt.plot(tmp["Date"], tmp["Rolling_Mean"], label=f"{ticker} Rolling Mean", linestyle='--')
    plt.title(f"Rolling {window}-Day Mean & Std Dev")
    plt.xlabel("Date")
    plt.ylabel("Price / Volatility")
    plt.legend()
    plt.show()

# -------------------------
# 5. Detect Outliers (Daily Returns)
# -------------------------
def detect_outliers(df: pd.DataFrame, threshold=3) -> pd.DataFrame:
    outliers = pd.DataFrame()
    for ticker in df["Ticker"].unique():
        tmp = df[df["Ticker"] == ticker].copy()
        mean, std = tmp["Daily_Return"].mean(), tmp["Daily_Return"].std()
        tmp_outliers = tmp[np.abs(tmp["Daily_Return"] - mean) > threshold*std]
        outliers = pd.concat([outliers, tmp_outliers])
    return outliers
# =========================
# Task 4: Seasonality & Trend Analysis
# =========================

def adf_test_series(df: pd.DataFrame, column="Close", ticker_col="Ticker"):
    """
    Perform Augmented Dickey-Fuller test on the series for each ticker.
    Returns a DataFrame summarizing the results.
    """
    results = []
    for ticker in df[ticker_col].unique():
        tmp = df[df[ticker_col] == ticker][column].dropna()
        adf_res = adfuller(tmp)
        results.append({
            "Ticker": ticker,
            "ADF Statistic": adf_res[0],
            "p-value": adf_res[1],
            "Used Lag": adf_res[2],
            "Number of Observations": adf_res[3],
            "Stationary": adf_res[1] < 0.05
        })
    return pd.DataFrame(results)

def plot_decomposition(df: pd.DataFrame, column="Close", ticker_col="Ticker", model="additive", freq=None):
    """
    Plot seasonal decomposition for each ticker.
    freq: the periodicity (e.g., 252 for daily stock prices, roughly 1 year trading days)
    """
    for ticker in df[ticker_col].unique():
        tmp = df[df[ticker_col] == ticker].set_index("Date")[column].dropna()
        decomposition = seasonal_decompose(tmp, model=model, period=freq)
        fig = decomposition.plot()
        fig.set_size_inches(14, 8)
        fig.suptitle(f"{ticker} - {model.capitalize()} Seasonal Decomposition", fontsize=16)
        plt.show()
        
# =========================
# Risk Metrics: VaR & Sharpe
# =========================

def calculate_var(df: pd.DataFrame, confidence_level=0.05) -> pd.DataFrame:
    """
    Calculate historical Value at Risk (VaR) at the specified confidence level for each ticker.
    """
    var_df = pd.DataFrame()
    for ticker in df["Ticker"].unique():
        tmp = df[df["Ticker"] == ticker]
        var = -np.percentile(tmp["Daily_Return"].dropna(), confidence_level*100)
        var_df = pd.concat([var_df, pd.DataFrame({"Ticker": [ticker], "VaR_95": [var]})])
    return var_df

def calculate_sharpe(df: pd.DataFrame, risk_free_rate=0.0) -> pd.DataFrame:
    """
    Calculate daily and annualized Sharpe Ratio for each ticker.
    Assumes daily returns are in percentage form.
    """
    sharpe_df = pd.DataFrame()
    for ticker in df["Ticker"].unique():
        tmp = df[df["Ticker"] == ticker]
        mean_ret = tmp["Daily_Return"].mean() - risk_free_rate
        std_ret = tmp["Daily_Return"].std()
        sharpe_daily = mean_ret / std_ret
        sharpe_annual = sharpe_daily * np.sqrt(252)  # annualize assuming 252 trading days
        sharpe_df = pd.concat([sharpe_df, pd.DataFrame({
            "Ticker": [ticker],
            "Sharpe_Daily": [sharpe_daily],
            "Sharpe_Annual": [sharpe_annual]
        })])
    return sharpe_df
