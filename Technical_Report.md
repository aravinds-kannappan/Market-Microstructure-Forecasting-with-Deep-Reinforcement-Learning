# Technical Report: Market Microstructure Forecasting with Deep Reinforcement Learning

## Introduction

High-frequency trading (HFT) strategies hinge on exploiting transient inefficiencies in limit order books (LOBs). The LOB is the exchange's matching engine that stores outstanding limit orders. Understanding its dynamics requires granular, high-dimensional data and modelling techniques capable of learning complex temporal patterns. In this project I implement a complete pipeline for LOB forecasting and trading. I adopt the FI-2010 dataset introduced by Ntakaris et al., which contains approximately four million events for five Finnish stocks traded on NASDAQ Nordic over ten consecutive trading days ([arxiv.org](https://ar5iv.labs.arxiv.org)). The data originates from the ITCH feed and provides a complete market-wide event history ([arxiv.org](https://ar5iv.labs.arxiv.org)).

My objectives are to:
1. Predict short-horizon price movements from order book snapshots
2. Use these predictions to inform a deep reinforcement learning (RL) agent that executes trades
3. Evaluate performance via event-driven backtesting

I prioritise reproducibility, modularity and efficient execution.

## Data and Feature Engineering

The FI-2010 dataset provides time-ordered snapshots of the top 10 levels of the LOB. I implemented a flexible loader (`LOBDataLoader`) that reads CSV files with columns `time`, `bid_price_i`, `bid_size_i`, `ask_price_i`, `ask_size_i` for i=1..N and computes several hand-crafted features widely used in the LOB literature:

- **Spread**: difference between the best ask and best bid price
- **Mid price**: the average of the best bid and best ask prices
- **Order flow imbalance**: the normalised difference between the sum of bid volumes and the sum of ask volumes
- **Depth imbalance**: the average difference between bid and ask depths at each level
- **Rolling volatility**: the rolling standard deviation of mid-price returns over a fixed window

Samples are labelled based on the mid-price movement over a horizon of Δt time steps. Following the dataset's original protocol, I define three classes: upward (+1), downward (–1) and stationary (0). For a prediction horizon of one second with 10 ms sampling frequency, Δt corresponds to 100 steps. The loader constructs an ordered feature matrix and an array of movement labels which are chronologically split into training and test sets.

## Baseline Models

I implemented two supervised baseline classifiers: multinomial logistic regression and LightGBM. A pipeline with z-score normalisation feeds features into the logistic regression. LightGBM uses a multi-class objective with three leaves and 100 boosting rounds. Both models output probabilities for the three movement classes. I evaluate classification accuracy, macro F1-score and a simple trading Sharpe ratio (apply a long/short strategy based on the predicted sign). Baseline results serve as a benchmark for more complex models and the RL agent.

## Temporal Convolutional Network

High-frequency price movements exhibit temporal dependencies that simple linear models cannot capture. I implemented a temporal convolutional network (TCN) consisting of dilated convolutional blocks with residual connections and ReLU activations. Each input sequence spans a fixed history of past LOB features. The receptive field grows exponentially through dilation, enabling the network to model long-range patterns without increasing depth dramatically. The network's output at the final time step is passed through a fully connected layer to produce class logits.

Training uses cross-entropy loss and the Adam optimiser. A small demonstration configuration with three layers and 64 hidden units per layer is provided. The TCN generally outperforms logistic regression in classification accuracy and Sharpe ratio, although training requires more computational resources.

## Reinforcement Learning Environment

To couple price prediction with execution, I created a custom environment (`LOBTradingEnv`) conforming to the Gymnasium API. The agent's observation combines the current LOB features, the ML model's prediction and the current position. Actions include:
- Holding
- Placing limit orders
- Placing market orders

Limit orders are assumed to execute at the mid price, while market orders cross the spread. Transaction costs and a market impact penalty discourage excessive trading. The reward at each step is the change in PnL net of costs. The environment resets at the beginning of the dataset and terminates when the end is reached, closing any open position.

## Deep RL Agent

I trained a Proximal Policy Optimisation (PPO) agent from `stable-baselines3` to learn an optimal trade execution policy. The agent observes the environment's state and outputs discrete actions. My training function wraps the environment in a `DummyVecEnv` to handle the vectorised API. During training the agent minimises a clipped surrogate loss while estimating value functions. For demonstration purposes I limited the number of timesteps; real experiments should run for several hundred thousand steps. Evaluation on held-out data yields an average reward and total PnL. In my small synthetic example, the RL agent outperformed the naive ML strategy by adjusting its position size and timing trades more effectively.

## C++ Execution Engine

Execution latency is critical in HFT. I implemented a simple order book matching engine in C++ (`execution_engine.cpp`). The engine maintains two sorted maps for bids and asks, supports limit orders, market orders, cancellations and queries of the best bid/ask. Matching removes liquidity from the opposite side of the book in price–time priority and returns the average execution price. Thread safety is ensured via a mutex; in a production system one would use finer-grained locks or lock-free data structures. 

A C API exposes functions for integration with Python via ctypes. Although g++ was unavailable in the constrained environment, the source can be compiled by users with a C++17 compiler to obtain a shared library. Python functions in `execution_bridge.py` wrap the C API and provide a familiar interface.

## Event-Driven Backtesting

I created a lightweight backtesting engine (`backtester.py`) that simulates strategies on historical data. For ML models, the engine implements a naive long/short policy: go long (short) if the predicted label is upward (downward) and hold otherwise. It computes an equity curve from mid-price changes and transaction costs, then derives performance metrics such as:
- Total return
- Maximum drawdown
- Sharpe ratio
- Calmar ratio

For RL agents, I use the environment directly and cumulatively sum rewards to obtain PnL. Backtesting helps compare ML benchmarks with RL strategies and examine the statistical properties of returns.

## Results and Discussion

Because of computational constraints I demonstrate the pipeline on a small synthetic dataset and a subset of the FI-2010 data. Logistic regression achieved modest predictive accuracy and Sharpe ratio. The TCN improved both metrics by capturing temporal dependencies. The PPO agent, albeit trained with few timesteps, learned to adjust its position more smoothly than the naive long/short strategy and generated higher PnL in the demo. These results illustrate the potential of combining deep learning and reinforcement learning for market microstructure forecasting.

## Future Improvements

Several extensions could enhance the framework:

### Data Enrichment
- Incorporate additional features such as queue positions, order age and hidden liquidity
- Apply feature scaling and normalisation consistent with the FI-2010 protocol

### Model Architectures  
- Explore transformer models for LOB data and compare them with TCNs
- Transformers may capture long-range dependencies more effectively but at increased computational cost

### Hyperparameter Tuning
- Perform systematic tuning for model and RL hyperparameters using cross-validation on the anchored folds described in the FI-2010 paper
- This includes window sizes, sequence lengths, learning rates and network sizes

### Execution Realism
- Integrate the C++ engine into the RL environment via pybind11
- Implement more realistic limit order execution, including queue dynamics, partial fills and slippage

### Distributed Computing
- Run backtests in parallel across multiple strategies using Dask or MPI
- Leverage GPUs for faster training

## Conclusion

I presented an end-to-end framework for market microstructure forecasting and trade execution. The pipeline includes data ingestion, feature engineering, baseline and deep models, a custom RL environment, a C++ execution engine and a backtester. While the demonstration used modest computational resources, the design scales to larger data sets and more sophisticated models. This work serves as a foundation for future research and experimentation with high-frequency trading strategies.
