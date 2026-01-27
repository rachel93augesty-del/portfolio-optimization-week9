# src/processed_data.py
import pandas as pd
from pathlib import Path

def load_tsla_cleaned(processed_dir="../data/processed"):
    """
    Load processed Tesla stock data for forecasting (Task 3)
    
    Args:
        processed_dir (str or Path): Path to the 'data/processed' folder
    Returns:
        tsla_df (pd.DataFrame): Tesla stock data indexed by Date with 'Close' column
    """
    processed_dir = Path(processed_dir)
    filepath = processed_dir / "all_assets_clean.csv"

    # Load CSV
    df = pd.read_csv(filepath, parse_dates=["Date"])

    # Strip any spaces from column names (safety)
    df.columns = df.columns.str.strip()

    # Filter Tesla only
    tsla_df = df[df["Ticker"] == "TSLA"].copy()
    tsla_df.set_index("Date", inplace=True)

    # Keep only Price column and rename to Close
    if "Price" in tsla_df.columns:
        tsla_df = tsla_df[["Price"]]
        tsla_df.rename(columns={"Price": "Close"}, inplace=True)
    else:
        raise ValueError("No 'Price' column found in CSV")

    return tsla_df
