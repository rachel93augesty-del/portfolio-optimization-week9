# src/data_reshape.py
import pandas as pd

def reshape_wide_to_long(df):
    frames = []
    tickers = ["TSLA", "BND", "SPY"]
    for ticker in tickers:
        expected_cols = [f"Open_{ticker}", f"High_{ticker}", f"Low_{ticker}", f"Close_{ticker}", f"Volume_{ticker}"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print(f"⚠️ Missing columns for {ticker}: {missing}, skipping")
            continue
        tmp = df[["Date"] + expected_cols].copy()
        tmp.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        tmp["Ticker"] = ticker
        frames.append(tmp)
    if not frames:
        raise ValueError("No valid ticker columns found to reshape!")
    df_long = pd.concat(frames, ignore_index=True)
    return df_long
