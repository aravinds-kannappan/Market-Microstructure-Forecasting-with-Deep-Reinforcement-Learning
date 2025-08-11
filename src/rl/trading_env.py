"""
trading_env.py
===============
Gymnasium env for event-driven trading on LOB features.
"""
from typing import Optional, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class LOBTradingEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, features: np.ndarray, mid_prices: np.ndarray, preds: Optional[np.ndarray] = None,
                 fee_bps: float = 1.0, slip_bps_mkt: float = 2.0, slip_bps_lmt: float = 0.2,
                 max_position: float = 1.0, inventory_penalty: float = 0.0, episode_length: Optional[int] = None, seed: int = 42):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.features = features; self.mid = mid_prices; self.preds = preds
        self.fee_bps = fee_bps; self.slip_bps_mkt = slip_bps_mkt; self.slip_bps_lmt = slip_bps_lmt
        self.max_position = max_position; self.inventory_penalty = inventory_penalty
        self.max_t = episode_length or (len(self.mid) - 1)
        fdim = features.shape[1]; add = 1 if preds is not None else 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(fdim+add,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self._reset_state()
    def _reset_state(self):
        self.t = 0; self.pos = 0.0; self.equity = 1.0
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed); self._reset_state(); return self._obs(), {}
    def _obs(self):
        f = self.features[self.t]
        return (np.concatenate([f, np.array([self.preds[self.t]], dtype=np.float32)]) if self.preds is not None else f).astype(np.float32)
    def step(self, action: int):
        assert self.action_space.contains(action); done = False
        mid_now = float(self.mid[self.t]); exec_price = mid_now; fee = 0.0; dpos = 0.0
        mkt_slip = self.slip_bps_mkt * 1e-4; lmt_slip = self.slip_bps_lmt * 1e-4; fee_frac = self.fee_bps * 1e-4
        if action == 1: exec_price = mid_now*(1+mkt_slip); dpos = +0.1; fee = fee_frac
        elif action == 2: exec_price = mid_now*(1-mkt_slip); dpos = -0.1; fee = fee_frac
        elif action == 3: exec_price = mid_now*(1+lmt_slip); dpos = +0.05
        elif action == 4: exec_price = mid_now*(1-lmt_slip); dpos = -0.05
        self.pos = float(np.clip(self.pos + dpos, -self.max_position, self.max_position))
        self.t += 1; 
        if self.t >= self.max_t: done = True
        mid_next = float(self.mid[self.t])
        ret = (mid_next - mid_now) / mid_now
        pnl = self.pos * ret
        costs = fee
        inv_pen = self.inventory_penalty * (self.pos ** 2)
        reward = pnl - costs - inv_pen
        self.equity *= (1.0 + reward)
        return self._obs(), float(reward), done, False, {"equity": self.equity, "pos": self.pos}
    def render(self): pass
