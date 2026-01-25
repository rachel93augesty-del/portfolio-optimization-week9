# Time Series Forecasting for Portfolio Optimization

## Project Overview
This project is part of the **10 Academy Artificial Intelligence Mastery – Week 9 Challenge**.  
It focuses on applying **time series forecasting and modern portfolio theory (MPT)** to support data-driven investment decisions for a fictional financial advisory firm, **Guide Me in Finance (GMF) Investments**.

The objective is to analyze historical financial market data, forecast asset behavior, and construct an optimized investment portfolio that balances return and risk.

---

## Business Context
GMF Investments provides personalized portfolio management using data-driven insights.  
Due to market efficiency (Efficient Market Hypothesis), exact price prediction is difficult. Instead, forecasting models are used as **decision-support tools** to:
- Understand market trends
- Estimate risk and volatility
- Improve asset allocation strategies

---

## Assets Analyzed
| Asset | Ticker | Description | Risk Profile |
|------|-------|------------|-------------|
| Tesla | TSLA | High-growth equity | High risk / high return |
| Vanguard Total Bond Market ETF | BND | Bond market exposure | Low risk |
| S&P 500 ETF | SPY | Broad market index | Moderate risk |

Data is sourced using the **YFinance API** from **2015-01-01 to 2026-01-15**.

---

## Project Objectives
- Perform exploratory data analysis (EDA) on financial time series
- Analyze volatility, returns, and risk metrics
- Test stationarity and prepare data for modeling
- Build and compare forecasting models (ARIMA/SARIMA and LSTM)
- Forecast future market behavior
- Optimize a portfolio using Modern Portfolio Theory
- Backtest the optimized strategy against a benchmark

---

## Project Structure
portfolio-optimization/
├── data/
│ └── processed/
├── notebooks/
├── src/
├── tests/
├── scripts/
├── requirements.txt
├── README.md
└── .github/workflows/