"""
enhanced_data_loader.py
========================
Enhanced data loader supporting FI-2010, LOBSTER, and custom LOB formats.
Integrates with DeepLOB, TransLOB, and LOBCAST preprocessing pipelines.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import pickle

@dataclass
class LOBConfig:
    """Configuration for LOB data processing"""
    dataset_type: str = "fi2010"  # "fi2010", "lobster", "custom"
    n_levels: int = 10
    feature_set: str = "full"  # "raw", "basic", "full"
    normalize: bool = True
    horizon: int = 10  # prediction horizon in events
    train_days: List[int] = None
    test_days: List[int] = None
    stocks: List[str] = None
    
    def __post_init__(self):
        if self.dataset_type == "fi2010":
            self.train_days = self.train_days or [1, 2, 3, 4, 5, 6, 7]
            self.test_days = self.test_days or [8, 9, 10]
            self.stocks = self.stocks or ["AAPL", "GOOG", "MSFT", "AMZN", "FB"]  # placeholder names

class FI2010Loader:
    """Specialized loader for FI-2010 dataset"""
    
    def __init__(self, data_dir: str, config: LOBConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self.n_features = 144  # FI-2010 has 144 hand-crafted features
        
    def load_raw_data(self, file_pattern: str = "Train_Dst_NoAuction_DecPre_CF_*.txt") -> np.ndarray:
        """Load FI-2010 raw files"""
        files = sorted(self.data_dir.glob(file_pattern))
        if not files:
            raise FileNotFoundError(f"No FI-2010 files found matching {file_pattern} in {self.data_dir}")
        
        data_list = []
        for f in files:
            # FI-2010 files are space-delimited with 144 features + labels
            df = pd.read_csv(f, sep='\s+', header=None)
            data_list.append(df.values)
        
        return np.concatenate(data_list, axis=0) if data_list else np.array([])
    
    def extract_features_labels(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from FI-2010 format"""
        # FI-2010: first 144 columns are features, remaining are labels for different horizons
        features = data[:, :self.n_features]
        
        # Labels are in columns 144, 145, 146, 147, 148 for horizons 10, 20, 30, 50, 100
        horizon_map = {10: 144, 20: 145, 30: 146, 50: 147, 100: 148}
        label_col = horizon_map.get(self.config.horizon, 144)
        
        if data.shape[1] > label_col:
            labels = data[:, label_col].astype(int) - 1  # Convert to 0-indexed
        else:
            # Compute labels from price movements if not provided
            labels = self._compute_labels(features)
        
        return features, labels
    
    def _compute_labels(self, features: np.ndarray) -> np.ndarray:
        """Compute price movement labels from features"""
        # Extract mid-price from features (assuming standard FI-2010 structure)
        # Mid-price is typically computed from best bid/ask
        bid_price = features[:, 0]  # First bid price
        ask_price = features[:, self.config.n_levels]  # First ask price
        mid_price = (bid_price + ask_price) / 2
        
        # Compute future returns
        future_mid = np.roll(mid_price, -self.config.horizon)
        returns = (future_mid - mid_price) / mid_price
        
        # Threshold for movement (typically 0.0002 for FI-2010)
        alpha = 0.0002
        labels = np.where(returns > alpha, 1, 
                         np.where(returns < -alpha, -1, 0))
        
        # Handle edge cases
        labels[-self.config.horizon:] = 0
        
        return labels

class LOBSTERLoader:
    """Loader for LOBSTER data format"""
    
    def __init__(self, data_dir: str, config: LOBConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        
    def load_orderbook_message_pair(self, 
                                   orderbook_file: str, 
                                   message_file: str) -> pd.DataFrame:
        """Load LOBSTER orderbook and message files"""
        # LOBSTER format: orderbook has price/size pairs, messages have events
        orderbook = pd.read_csv(orderbook_file, header=None)
        messages = pd.read_csv(message_file, header=None)
        
        # Combine and format
        n_levels = self.config.n_levels
        col_names = []
        for i in range(1, n_levels + 1):
            col_names.extend([f'ask_price_{i}', f'ask_size_{i}', 
                            f'bid_price_{i}', f'bid_size_{i}'])
        
        orderbook.columns = col_names[:orderbook.shape[1]]
        orderbook['timestamp'] = messages[0]  # First column is typically timestamp
        
        return orderbook

class UnifiedLOBLoader:
    """Unified loader supporting multiple dataset formats"""
    
    def __init__(self, config: LOBConfig):
        self.config = config
        self.data = None
        self.features = None
        self.labels = None
        self.scaler = StandardScaler() if config.normalize else None
        
    def load_data(self, data_path: str) -> 'UnifiedLOBLoader':
        """Load data based on configured dataset type"""
        if self.config.dataset_type == "fi2010":
            loader = FI2010Loader(data_path, self.config)
            raw_data = loader.load_raw_data()
            self.features, self.labels = loader.extract_features_labels(raw_data)
            
        elif self.config.dataset_type == "lobster":
            loader = LOBSTERLoader(data_path, self.config)
            # Implement LOBSTER loading logic
            raise NotImplementedError("LOBSTER loader integration pending")
            
        elif self.config.dataset_type == "custom":
            # Load custom CSV format
            self.data = pd.read_csv(data_path)
            self.features, self.labels = self._extract_custom_features()
            
        else:
            raise ValueError(f"Unknown dataset type: {self.config.dataset_type}")
        
        return self
    
    def _extract_custom_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from custom CSV format"""
        feature_cols = []
        
        if self.config.feature_set in ["raw", "full"]:
            # Raw LOB features
            for i in range(1, self.config.n_levels + 1):
                for side in ['bid', 'ask']:
                    for typ in ['price', 'size']:
                        col = f'{side}_{typ}_{i}'
                        if col in self.data.columns:
                            feature_cols.append(col)
        
        if self.config.feature_set in ["basic", "full"]:
            # Compute basic features
            self.data = self._compute_basic_features(self.data)
            feature_cols.extend(['spread', 'mid_price', 'ofi', 'depth_imb', 'rolling_vol'])
        
        if self.config.feature_set == "full":
            # Add advanced features
            self.data = self._compute_advanced_features(self.data)
            feature_cols.extend(['wap', 'price_spread', 'volume_imb', 'bid_ask_spread_derivative'])
        
        features = self.data[feature_cols].values
        labels = self._compute_labels_from_df(self.data)
        
        return features, labels
    
    def _compute_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute basic LOB features"""
        # Spread
        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        
        # Mid price
        df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2
        
        # Order flow imbalance
        bid_vol = df[[f'bid_size_{i}' for i in range(1, self.config.n_levels + 1)]].sum(axis=1)
        ask_vol = df[[f'ask_size_{i}' for i in range(1, self.config.n_levels + 1)]].sum(axis=1)
        df['ofi'] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        
        # Depth imbalance
        df['depth_imb'] = df['bid_size_1'] - df['ask_size_1']
        
        # Rolling volatility
        df['rolling_vol'] = df['mid_price'].pct_change().rolling(100, min_periods=10).std()
        
        return df.fillna(0)
    
    def _compute_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute advanced LOB features inspired by DeepLOB and TransLOB"""
        # Weighted average price (WAP)
        df['wap'] = (df['bid_price_1'] * df['ask_size_1'] + 
                    df['ask_price_1'] * df['bid_size_1']) / (df['bid_size_1'] + df['ask_size_1'] + 1e-9)
        
        # Price spread across levels
        df['price_spread'] = df['ask_price_5'] - df['bid_price_5'] if 'ask_price_5' in df.columns else df['spread']
        
        # Volume imbalance ratio
        for i in range(1, min(6, self.config.n_levels + 1)):
            df[f'vol_imb_{i}'] = (df[f'bid_size_{i}'] - df[f'ask_size_{i}']) / (
                df[f'bid_size_{i}'] + df[f'ask_size_{i}'] + 1e-9)
        
        # Spread derivative
        df['bid_ask_spread_derivative'] = df['spread'].diff()
        
        # Queue imbalance (if available)
        # This would require order-level data
        
        return df.fillna(0)
    
    def _compute_labels_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """Compute movement labels from dataframe"""
        if 'label' in df.columns:
            return df['label'].values
        
        mid_price = df['mid_price']
        future_mid = mid_price.shift(-self.config.horizon)
        returns = (future_mid - mid_price) / mid_price
        
        # Dynamic threshold based on volatility
        vol = returns.rolling(100, min_periods=10).std().fillna(returns.std())
        threshold = vol * 0.5  # Adaptive threshold
        
        labels = np.where(returns > threshold, 1,
                         np.where(returns < -threshold, -1, 0))
        
        return labels
    
    def prepare_sequences(self, 
                         seq_length: int = 100,
                         stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for deep learning models"""
        if self.features is None:
            raise ValueError("No features loaded. Call load_data() first.")
        
        n_samples = len(self.features)
        sequences = []
        seq_labels = []
        
        for i in range(0, n_samples - seq_length - self.config.horizon, stride):
            seq = self.features[i:i + seq_length]
            label = self.labels[i + seq_length]
            
            sequences.append(seq)
            seq_labels.append(label)
        
        return np.array(sequences), np.array(seq_labels)
    
    def train_test_split(self, 
                        test_ratio: float = 0.2,
                        val_ratio: float = 0.1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data into train, validation, and test sets"""
        n_samples = len(self.features)
        
        # Temporal split (no shuffling for time series)
        train_end = int(n_samples * (1 - test_ratio - val_ratio))
        val_end = int(n_samples * (1 - test_ratio))
        
        splits = {
            'train': (self.features[:train_end], self.labels[:train_end]),
            'val': (self.features[train_end:val_end], self.labels[train_end:val_end]),
            'test': (self.features[val_end:], self.labels[val_end:])
        }
        
        # Normalize if configured
        if self.scaler:
            X_train, y_train = splits['train']
            self.scaler.fit(X_train)
            
            for split_name in splits:
                X, y = splits[split_name]
                X_normalized = self.scaler.transform(X)
                splits[split_name] = (X_normalized, y)
        
        return splits
    
    def save_processed(self, output_path: str):
        """Save processed features and labels"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_path,
            features=self.features,
            labels=self.labels,
            config=self.config.__dict__
        )
        
        if self.scaler:
            with open(output_path.with_suffix('.scaler'), 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def load_processed(self, input_path: str) -> 'UnifiedLOBLoader':
        """Load preprocessed data"""
        data = np.load(input_path)
        self.features = data['features']
        self.labels = data['labels']
        
        scaler_path = Path(input_path).with_suffix('.scaler')
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        return self


# Example usage function
def prepare_fi2010_data(data_dir: str = "data/fi2010/raw") -> Dict:
    """Complete pipeline for FI-2010 data preparation"""
    
    # Configure for FI-2010
    config = LOBConfig(
        dataset_type="fi2010",
        n_levels=10,
        feature_set="full",
        normalize=True,
        horizon=10  # 10 events ahead prediction
    )
    
    # Load and process
    loader = UnifiedLOBLoader(config)
    loader.load_data(data_dir)
    
    # Prepare sequences for deep learning
    sequences, labels = loader.prepare_sequences(seq_length=100, stride=10)
    
    # Split data
    splits = loader.train_test_split(test_ratio=0.2, val_ratio=0.1)
    
    # Save processed data
    loader.save_processed("data/fi2010/processed/fi2010_processed.npz")
    
    print(f"Data prepared successfully!")
    print(f"Training samples: {len(splits['train'][0])}")
    print(f"Validation samples: {len(splits['val'][0])}")
    print(f"Test samples: {len(splits['test'][0])}")
    print(f"Feature dimension: {splits['train'][0].shape[1]}")
    
    return splits
