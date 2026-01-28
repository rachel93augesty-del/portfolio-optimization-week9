# src/metrics.py

import numpy as np
import pandas as pd

# -----------------------------
# Portfolio Metrics
# -----------------------------

def portfolio_daily_returns(prices: pd.DataFrame, weights: list) -> pd.Series:
    """
    Compute portfolio daily returns.
    """
    weights = np.array(weights)
    daily_returns = prices.pct_change().dropna()
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    return portfolio_returns


def portfolio_cumulative_returns(daily_returns: pd.Series) -> pd.Series:
    """
    Compute cumulative returns from daily returns.
    """
    return (1 + daily_returns).cumprod()


def portfolio_performance(daily_returns: pd.Series) -> tuple:
    """
    Compute annualized return, volatility, and Sharpe ratio.
    """
    ann_return = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    return ann_return, ann_vol, sharpe


# -----------------------------
# Backtest / Performance Metrics
# -----------------------------

def total_return(cum_returns: pd.Series) -> float:
    """
    Total return from cumulative returns
    """
    return cum_returns.iloc[-1] / cum_returns.iloc[0] - 1


def annualized_return(daily_returns: pd.Series) -> float:
    """
    Annualized return from daily returns
    """
    return daily_returns.mean() * 252


def annualized_volatility(daily_returns: pd.Series) -> float:
    """
    Annualized volatility from daily returns
    """
    return daily_returns.std() * np.sqrt(252)


def sharpe_ratio(daily_returns: pd.Series) -> float:
    """
    Sharpe ratio assuming risk-free = 0
    """
    return annualized_return(daily_returns) / annualized_volatility(daily_returns)


def max_drawdown(cum_returns: pd.Series) -> float:
    """
    Maximum drawdown from cumulative returns
    """
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    return drawdown.min()
