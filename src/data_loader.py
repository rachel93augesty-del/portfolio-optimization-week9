import pandas as pd
from pathlib import Path

def load_raw_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={df.columns[0]: "Date", df.columns[-1]: "Ticker"}, inplace=True)
    return df
