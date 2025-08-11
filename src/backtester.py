"""
backtester.py
=============
Event-driven backtesting engine for evaluating trading strategies on LOB data.
"""
import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, initial_cash=1_000_000, transaction_cost=0.0):
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.position = 0.0
        self.equity_curve = []

    def run(self, prices: pd.Series, signals: pd.Series):
        self.reset()
        prices = prices.reindex(signals.index).ffill()
        for price, sig in zip(prices, signals):
            # flatten if signal sign flips
            if np.sign(self.position) != np.sign(sig):
                self.cash += self.position * price * (1 - self.transaction_cost)
                self.position = 0.0
            # enter new position
            if sig == 1:
                units = self.cash / (price * (1 + self.transaction_cost))
                self.position += units
                self.cash -= units * price * (1 + self.transaction_cost)
            elif sig == -1:
                units = self.cash / (price * (1 + self.transaction_cost))
                self.position -= units
                self.cash += units * price * (1 - self.transaction_cost)
            # mark-to-market
            equity = self.cash + self.position * price
            self.equity_curve.append(equity)
        return pd.Series(self.equity_curve, index=signals.index)

    def compute_metrics(self):
        import numpy as np
        ec = np.array(self.equity_curve, dtype=float)
        rets = pd.Series(ec).pct_change().dropna()
        sharpe = 0.0 if rets.std() == 0 else (np.sqrt(252*6*60) * rets.mean() / rets.std())
        peak = np.maximum.accumulate(ec)
        mdd = np.max(peak - ec) if len(ec) else 0.0
        return {"Sharpe": float(sharpe), "MaxDrawdown": float(mdd)}
