"""
baseline.py
===========
Baselines for mid-price direction prediction on LOB features.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

def _naive_strategy_returns(mid_prices: pd.Series, preds: pd.Series) -> pd.Series:
    preds = preds.reindex(mid_prices.index).fillna(0)
    rets = mid_prices.pct_change().fillna(0.0)
    pos = preds.shift(1).fillna(0.0)
    return pos * rets

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252*6*60) -> float:
    r = returns.dropna()
    if r.std() == 0: return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())

@dataclass
class BaselineResults:
    acc: float
    f1: float
    sharpe: float
    preds: pd.Series

class Baselines:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.lr: Optional[LogisticRegression] = None
        self.lgbm = None

    def _split(self, X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size, shuffle=False)

    def train_logreg(self, X: pd.DataFrame, y: pd.Series, mid_prices: pd.Series) -> BaselineResults:
        Xtr, Xte, ytr, yte = self._split(X, y)
        self.lr = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced")
        self.lr.fit(Xtr, ytr)
        yhat = pd.Series(self.lr.predict(Xte), index=Xte.index)
        acc = accuracy_score(yte, yhat)
        f1 = f1_score(yte, yhat, average="macro")
        sr = sharpe_ratio(_naive_strategy_returns(mid_prices.loc[Xte.index], yhat))
        return BaselineResults(acc=float(acc), f1=float(f1), sharpe=sr, preds=yhat)

    def train_lightgbm(self, X: pd.DataFrame, y: pd.Series, mid_prices: pd.Series, params: Optional[Dict]=None) -> BaselineResults:
        if not _HAS_LGB:
            raise RuntimeError("LightGBM not installed. `pip install lightgbm`.")
        Xtr, Xte, ytr, yte = self._split(X, y)
        params = params or {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "max_depth": -1,
            "min_data_in_leaf": 50,
            "verbosity": -1,
            "metric": "multi_logloss",
        }
        dtr = lgb.Dataset(Xtr, label=(ytr + 1))
        dve = lgb.Dataset(Xte, label=(yte + 1), reference=dtr)
        self.lgbm = lgb.train(params, dtr, valid_sets=[dtr, dve], num_boost_round=400, early_stopping_rounds=40, verbose_eval=False)
        cls = self.lgbm.predict(Xte).argmax(axis=1) - 1
        yhat = pd.Series(cls, index=Xte.index)
        acc = accuracy_score(yte, yhat)
        f1 = f1_score(yte, yhat, average="macro")
        sr = sharpe_ratio(_naive_strategy_returns(mid_prices.loc[Xte.index], yhat))
        return BaselineResults(acc=float(acc), f1=float(f1), sharpe=sr, preds=yhat)
