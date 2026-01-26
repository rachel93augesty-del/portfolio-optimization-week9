import pandas as pd
import yfinance as yf
from typing import List
from pathlib import Path

# Project-level data paths
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_and_save_market_data(
    tickers: List[str],
    start_date: str = "2015-01-01",
    end_date: str = "2026-01-15"
) -> pd.DataFrame:
    """
    Fetch historical adjusted close prices using YFinance
    and persist raw data to data/raw.

    Returns:
        pd.DataFrame: Long-format dataframe with Date, Ticker, Price
    """

    raw_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    df = (
        raw_data["Close"]
        .reset_index()
        .melt(id_vars="Date", var_name="Ticker", value_name="Price")
        .dropna()
    )

    output_path = RAW_DATA_DIR / "market_prices_raw.csv"
    df.to_csv(output_path, index=False)

    print("Data extraction completed successfully")
    print(f"Saved to: {output_path}")
    print(f"Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"Tickers: {df['Ticker'].unique().tolist()}")
    print(f"Total rows: {len(df)}")

    return df
