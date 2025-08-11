"""
train_agent.py
===============
PPO training wrapper for the LOBTradingEnv (stable-baselines3).
"""
from typing import Optional
import numpy as np
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    _HAS_SB3 = True
except Exception:
    _HAS_SB3 = False

def train_ppo_agent(features: np.ndarray, mid_prices: np.ndarray, preds: Optional[np.ndarray] = None,
                    total_timesteps: int = 50_000, learning_rate: float = 3e-4, n_steps: int = 1024,
                    gamma: float = 0.99, clip_range: float = 0.2, seed: int = 42):
    if not _HAS_SB3:
        raise RuntimeError("stable-baselines3 not installed. `pip install stable-baselines3`.")
    from .trading_env import LOBTradingEnv
    def make_env():
        return LOBTradingEnv(features=features, mid_prices=mid_prices, preds=preds,
                             fee_bps=1.0, slip_bps_mkt=2.0, slip_bps_lmt=0.2,
                             max_position=1.0, inventory_penalty=0.0,
                             episode_length=len(mid_prices)-1, seed=seed)
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=learning_rate, n_steps=n_steps,
                gamma=gamma, clip_range=clip_range, seed=seed, tensorboard_log=None)
    model.learn(total_timesteps=total_timesteps)
    return model
