# Market Microstructure Forecasting with Deep Reinforcement Learning

This repository contains a full implementation of a high-frequency trading research project. The goal is to forecast short-horizon price movements from limit order book (LOB) data using deep learning and to exploit those predictions with a reinforcement learning (RL) execution agent. The project is modular and includes data ingestion, feature engineering, baseline and advanced models, a custom trading environment, a C++ execution engine, an event-driven backtester, and a demonstration notebook.

## Dataset

I target the FI-2010 limit order book dataset, which is publicly available and consists of approximately four million events from five Finnish stocks over ten consecutive trading days ([arxiv.org](https://ar5iv.labs.arxiv.org)). The data is derived from the NASDAQ Nordic ITCH feed and provides top-of-book snapshots at high frequency ([arxiv.org](https://ar5iv.labs.arxiv.org)).

You must manually download the dataset from the official source (see Section 3 of the paper for the link) and extract the CSV files. Place the CSVs under the `data/` directory. Each CSV should have columns named `time`, `bid_price_i`, `bid_size_i`, `ask_price_i` and `ask_size_i` for i = 1..N where N is the depth of the book.

If you do not have access to the FI-2010 data, any similar LOB dataset or even simpler bid/ask quote data will work provided it follows the same column convention. The notebook includes a small synthetic fallback example for testing.

## Installation

1. Clone this repository and change into the project directory.

2. Create a Python virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies are:
   - `pandas`, `numpy` for data processing
   - `scikit-learn` and optionally `lightgbm` for baseline models
   - `torch` for deep learning
   - `gymnasium` and `stable-baselines3` for RL
   - `jupyter` for running the notebook

4. Compile the C++ execution library (optional but recommended for realistic execution):

   The execution engine is implemented in `src/execution_engine.cpp`. To compile it into a shared library run:
   ```bash
   g++ -O3 -std=c++17 -shared -fPIC src/execution_engine.cpp -o src/libexecution_engine.so
   ```

   You need a C++17 compiler (e.g. g++). If compilation succeeds you will find `libexecution_engine.so` in the `src/` directory. The Python wrapper in `src/execution_bridge.py` will load this library at run time.

5. Launch JupyterLab or Jupyter Notebook to run the example notebook:
   ```bash
   jupyter notebook notebooks/lob_rl_demo.ipynb
   ```

## Directory Structure

```
├── src/                    # Python and C++ source code
│   ├── data_loader.py      # LOB data ingestion and feature engineering
│   ├── models/             # Machine learning models
│   │   ├── baseline.py     # Logistic regression and LightGBM
│   │   └── deep.py         # Temporal convolutional network (TCN)
│   ├── rl/                 # Reinforcement learning components
│   │   ├── trading_env.py  # Custom Gymnasium environment
│   │   └── train_agent.py  # PPO training and evaluation
│   ├── backtester.py       # Event-driven backtesting engine
│   ├── execution_engine.cpp# Low-latency C++ order book implementation
│   └── execution_bridge.py # ctypes wrapper around the C++ library
├── data/                   # Place your LOB CSV files here
├── notebooks/
│   └── lob_rl_demo.ipynb   # Demonstration notebook
└── README.md               # This file
```

## Usage Overview

The typical workflow is:

1. **Data ingestion**: Use `LOBDataLoader` from `data_loader.py` to load your CSVs and compute engineered features and labels. Adjust the prediction horizon (`delta_t`) and rolling volatility window according to your sampling frequency.

2. **Baseline modelling**: Train a logistic regression or LightGBM model using `train_logistic_regression` and evaluate it with `evaluate_model`. This provides a performance benchmark.

3. **Deep learning**: Train a temporal convolutional network (TCN) using `train_tcn_model` in `models/deep.py`. TCNs can capture temporal dependencies in the order book history and often outperform simpler baselines.

4. **Reinforcement learning**: Set up the trading environment with your engineered features, mid price series and ML predictions. Then train a PPO agent using `train_ppo_agent` from `rl/train_agent.py`. Evaluate the agent on held-out data with `evaluate_rl_agent`.

5. **Backtesting**: Use `backtest_ml_strategy` in `backtester.py` to simulate a naive long/short strategy based on your ML predictions. Use `compute_performance_metrics` to compute equity curves, drawdowns and Sharpe ratios. For RL agents, you can simulate the environment directly and accumulate PnL from the rewards.

## Extending the Project

This framework is designed to be modular and extensible. Possible enhancements include:

- Incorporating additional features such as queue imbalance, order age and higher-order interactions
- Implementing a transformer architecture for better long-range dependency modelling
- Adding distributed training and backtesting via Dask or MPI
- Integrating the C++ engine via pybind11 for even lower latency
- Extending the backtester to model partial fills, slippage and complex transaction cost models

I hope this repository provides a solid foundation for exploring high-frequency trading research with modern machine learning and reinforcement learning.
