"""
data_loader.py
================
Utilities for loading and engineering features from Limit Order Book (LOB) CSVs.
Assumes columns like bid_price_1, ask_price_1, bid_size_1, ask_size_1 for the top level.
Extend to top-N by adding more columns (e.g., bid_price_2, ...).
"""
import pandas as pd
import numpy as np

class LOBDataLoader:
    def __init__(self, csv_path: str, top_n_levels: int = 1):
        self.csv_path = csv_path
        self.top_n_levels = top_n_levels
        self.df = None

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path)
        if "timestamp" in self.df.columns:
            self.df = self.df.sort_values("timestamp").reset_index(drop=True)
        return self.df

    def compute_features(self) -> pd.DataFrame:
        df = self.df.copy()
        # Basic features
        df["spread"] = df["ask_price_1"] - df["bid_price_1"]
        df["mid_price"] = (df["ask_price_1"] + df["bid_price_1"]) / 2.0
        # Order Flow Imbalance at top level
        df["ofi"] = (df["bid_size_1"] - df["ask_size_1"]) / (df["bid_size_1"] + df["ask_size_1"] + 1e-9)
        # Depth imbalance (top level)
        df["depth_imb"] = (df["bid_size_1"] - df["ask_size_1"]) / (df["bid_size_1"] + df["ask_size_1"] + 1e-9)
        # Rolling volatility (of mid)
        df["rolling_vol"] = df["mid_price"].pct_change().rolling(100, min_periods=10).std().fillna(0.0)
        self.df = df
        return self.df

    def label_data(self, horizon: int = 10) -> pd.DataFrame:
        df = self.df.copy()
        future_mid = df["mid_price"].shift(-horizon)
        diff = future_mid - df["mid_price"]
        df["label"] = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
        self.df = df.dropna()
        return self.df

if __name__ == "__main__":
    loader = LOBDataLoader("data/LOB_sample.csv")
    loader.load_data()
    loader.compute_features()
    loader.label_data(horizon=10)
    print(loader.df.head())
