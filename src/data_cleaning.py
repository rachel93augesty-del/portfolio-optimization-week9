import pandas as pd

def clean_data(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str)
    
    # Convert numeric columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    
    # Sort and forward-fill
    df = df.sort_values(["Ticker", "Date"])
    df[numeric_cols] = df.groupby("Ticker")[numeric_cols].ffill()
    
    # Drop remaining NAs
    df = df.dropna()
    return df

def scale_data(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    df_scaled = df.copy()
    for col in numeric_cols:
        min_val, max_val = df_scaled[col].min(), df_scaled[col].max()
        if max_val != min_val:
            df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)
    return df_scaled
