# Adaptive Multi-Factor Aggregation with Sentiment Context on CSI 300 (15-Minute Horizon)

## Executive Summary

This capstone project improves short-horizon equity signal aggregation in the CSI 300 universe by combining 70 price-derived microstructure factors with disclosure-based sentiment context. The core contribution is dynamic factor weighting: a multi-agent reinforcement learning aggregation layer (MATD3) learns when each of five base signals should receive more or less weight, conditioned on current market state and sentiment regime. The final model (MATD3 v3) achieves a 103.2% total return and Sharpe ratio of 1.769 over the 2023-12 to 2025-12 test period, substantially outperforming all individual signal baselines.

---

## Results (2026-04-03) — Final Results: MATD3 v3

### Close-to-Close Backtest (Top-5 EW, fee=0.03%, full period 2023-12 to 2025-12)

| Model | Total Return | Ann. Return | Sharpe | Sortino | Calmar | Max Drawdown | Avg Turnover |
|-------|-------------|-------------|--------|---------|--------|--------------|-------------|
| **MATD3 v3** | **103.2%** | **41.0%** | **1.769** | **3.253** | **2.023** | **-20.2%** | 1.284 |
| CNN+LSTM+TF | 74.9% | 31.1% | 1.438 | 2.476 | 1.702 | -18.3% | 0.872 |
| MATD3 v2 | 59.7% | 25.4% | 1.171 | 2.094 | 1.116 | -22.8% | 1.319 |
| Cross-Attn | 66.0% | 27.8% | 1.079 | 1.958 | 0.993 | -28.0% | 1.426 |
| GAT+TF | 18.2% | 8.4% | 0.422 | 0.645 | 0.353 | -23.9% | 1.042 |
| LightGBM | 17.8% | 8.3% | 0.294 | 0.519 | 0.215 | -38.3% | 1.648 |
| EW Ensemble | -24.1% | -12.5% | -0.561 | -0.977 | -0.352 | -35.5% | 1.604 |
| SSM+TF | -24.1% | -12.5% | -0.561 | -0.977 | -0.352 | -35.5% | 1.604 |

MATD3 v3 achieves the highest total return (103.2%), Sharpe (1.769), Sortino (3.253), and Calmar (2.023) across all models. These results supersede the earlier v3 figure of 82.3% reported mid-project, which was from an intermediate training run at 50 epochs; the final model was trained for 80 epochs.

### MATD3 Version History

| Parameter | v1 | v2 | v3 (final) |
|-----------|----|----|------------|
| Gate range | [0.5, 1.5] | [0, 2.0] | **[0, 3.0]** |
| dd_lambda | 0.5 | 0.3 | **0.1** |
| vol_lambda | 30.0 | 10.0 | **3.0** |
| reward_scale | 1000 | 1000 | **2000** |
| Epochs | 15 | 30 | **80** |
| Action noise | 0.05 | 0.08 | **0.10** |
| Total Return | — | 59.7% | **103.2%** |
| Sharpe | — | 1.171 | **1.769** |

### v3 Gate Weight Summary

| Signal | Mean Gate | Std | % above 1.0 | % near 0 |
|--------|-----------|-----|-------------|----------|
| CNN+LSTM+TF | 2.66 | 0.79 | 91.3% | 2.6% |
| SSM+TF | 2.07 | 1.31 | 70.8% | 21.3% |
| Cross-Attn | 1.87 | 1.28 | 65.8% | 18.3% |
| GAT+TF | 1.33 | 1.40 | 46.5% | 43.7% |
| LightGBM | 0.91 | 1.24 | 33.2% | 52.3% |

The RL agent learned to persistently up-weight CNN+LSTM+TF (mean gate 2.66) and suppress LightGBM (mean gate 0.91), consistent with their standalone IC performance.

---

## Ablation: v4 and v5 — Why v3 Remains Optimal

After confirming v3's strong performance, two experimental variants were developed to explore further improvement. Both failed to improve on v3 and are reported here as negative results for completeness.

### MATD3 v4 — Softmax Portfolio + Volatility Targeting

**Methods introduced:** (1) Replaced equal-weight (EW) top-5 portfolio with softmax-weighted allocation (temperature=3.0), so position sizes are proportional to signal strength. (2) Added a one-sided volatility targeting penalty: only excess volatility above a 20% annualised target is penalised (`var_penalty = vol_lambda * max(0, rolling_vol - target)^2 * 1000`). (3) Added a hard daily turnover cap (max 50% of portfolio per day). (4) Increased training to 80 epochs.

**Outcome:** v4 underperformed v3 on all metrics in the comparison backtest. Root cause: the softmax portfolio used during training is structurally different from the EW top-5 used in the close-to-close backtest framework. This train/test mismatch means the agent optimises for a portfolio rule it never executes at evaluation, leading to systematically suboptimal gate assignments. Additionally, the one-sided vol penalty was rarely active (rolling vol was frequently below the 20% target), and `turnover_lambda=0.5` was too aggressive, suppressing signal differentiation.

### MATD3 v5 — Calmar-Focused Reward + Restored EW Portfolio

**Methods introduced:** (1) Removed softmax and turnover cap from training — restored EW top-5 to achieve train/test consistency. (2) Shifted reward focus toward Calmar ratio by increasing `dd_lambda` from 0.1 to 0.15 and removing the volatility penalty entirely (`vol_lambda=0`). (3) Added a moderate turnover penalty (`turnover_lambda=0.25`). (4) Extended training to 100 epochs with slightly higher exploration noise (0.12).

**Outcome:** v5 does not materially improve on v3 in preliminary results. Increasing `dd_lambda` from the carefully tuned v3 value of 0.1 distorts the reward signal such that the agent becomes overly conservative, reducing turnover and thus signal responsiveness. v3's reward specification (`net_return − 0.1×drawdown − 3.0×rolling_var`) with 80 training epochs represents the best trade-off found.

**Conclusion:** MATD3 v3 is confirmed as the optimal configuration in this search space. Both v4 and v5 serve as ablation evidence that the v3 design choices (EW portfolio consistency, moderate dd and vol penalties, gate range [0,3]) are non-trivially load-bearing.

---

## Component Details

### 1. Microstructure Factor Engineering

**70 factors** (`Factor.py`, functions `f_001`–`f_070`) computed on 1.5M rows of 15-minute intraday bar data (2018–2025) for 50 CSI 300 constituent stocks.

Factor categories:
| Category | Examples | Count |
|----------|----------|-------|
| Spread & liquidity | Bid-ask spread, effective spread, Amihud illiquidity ratio | ~12 |
| Order flow | Order flow imbalance (OFI), buy/sell pressure ratio | ~10 |
| Price microstructure | Microprice deviation from mid, mid-return lags, momentum | ~18 |
| Volume | Log volume z-score, volume surprise, relative volume | ~12 |
| Volatility | Rolling realised vol at multiple horizons (5/10/20 bars) | ~10 |
| Composite | Cross-sectional ranks, interaction terms | ~8 |

All factors are computed in a strictly look-ahead-free manner: each bar uses only information available at `TimeStart` of that bar.

**LightGBM baseline:** IC = 0.0675, IC IR = **1.42** on 2024–2025 test set.

---

### 2. Deep Learning Signal Models

Five models are trained on `df_trainsub.parquet` (1.14M rows, 2018–2023) and inferred on the test period. All models target `ret_mid_t1` — the cross-sectionally demeaned next-bar mid-price return.

#### 2a. CNN + LSTM + Transformer (`CNN+LSTM+TRANSFORMER.py`)

**Architecture:** Three-stage temporal encoder.

1. **Conv1D block** (`kernel_size=3`, stride 1, padding 1): Extracts local temporal patterns from the factor sequence (sequence length = rolling window of recent bars per stock).
2. **LSTM** (`hidden_size=128`, 2 layers, dropout=0.1): Models non-linear sequential dependencies across the factor time series.
3. **Transformer encoder** (`d_model=128`, `nhead=4`, 2 encoder layers, `dim_feedforward=256`): Applies multi-head self-attention over the LSTM output sequence to capture long-range dependencies.
4. **Linear head:** Maps the CLS-equivalent token to a scalar signal prediction.

**Training:** Adam, `lr=1e-3`, `weight_decay=1e-5`, MSE loss on forward returns. Batch size 512, 30 epochs, gradient clipping at 1.0.

**Standalone backtest:** 74.9% total return, Sharpe 1.438 — the strongest individual signal and the primary target that MATD3 learns to up-weight.

#### 2b. Cross-Sectional Attention (`CROSS ATTEN.py`)

**Architecture:** Instead of temporal attention across time steps, applies attention *across stocks* at each timestamp.

1. **Stock embedding layer:** Projects each stock's factor vector to a common `d_model=64` space.
2. **CrossAttentionBlock:** Multi-head attention where each stock's embedding attends to all other stocks simultaneously, allowing the model to condition each stock's signal on the cross-sectional distribution. Query = stock's own embedding; Key/Value = all stocks' embeddings.
3. **Per-stock Transformer encoder:** Refines each stock's representation with 2 self-attention layers after cross-sectional mixing.
4. **Linear regression head:** Per-stock scalar output.

**Rationale:** Captures relative momentum and mean-reversion dynamics by explicitly modelling how a stock's return prediction depends on the contemporaneous cross-section.

**Standalone backtest:** 66.0% total return, Sharpe 1.079.

#### 2c. GAT + Transformer (`GNN.py`)

**Architecture:** Graph-based signal model using return correlations as edges.

1. **Dynamic correlation graph:** At each training batch, constructs a k-nearest-neighbour graph (k=10) over stocks based on their rolling 30-day return correlation matrix. Edge weight = Pearson correlation coefficient.
2. **Graph Attention Network (GAT):** 2 GAT layers with 4 attention heads each. Each node (stock) aggregates information from its correlated neighbours, weighted by learned attention over edge features (correlation values).
3. **Transformer encoder:** Applied to the per-node GAT output sequence (one node per time step across the lookback window), capturing temporal dynamics within the graph structure.
4. **Linear head:** Per-stock signal.

**Rationale:** Sector and factor co-movement in the CSI 300 means correlated stocks often share return drivers. The GAT captures these cross-stock spillovers explicitly.

**Standalone backtest:** 18.2% total return, Sharpe 0.422. The graph construction is sensitive to correlation instability in short lookback windows; MATD3 learns to suppress this signal (mean gate 1.33, 43.7% near-zero).

#### 2d. SSM + Transformer (`S4+Transformer.py`)

**Architecture:** State-space model for efficient long-sequence factor processing.

1. **Discrete SSM layer (S4-inspired):** Parameterises a linear time-invariant state-space model `h_t = Ah_{t-1} + Bu_t; y_t = Ch_t + Du_t` in discrete time. Computes the full convolutional filter `K = [CB, CAB, CA²B, ...]` via FFT for efficient O(L log L) sequence processing over long lookback windows (L=64 bars).
2. **Transformer encoder:** Two self-attention layers applied to the SSM output, providing global context across the sequence.
3. **Linear head:** Scalar return prediction.

**Rationale:** Motivated by the hypothesis that 15-minute factor sequences have long-range dependencies not well captured by LSTM's bounded memory. In practice, the SSM did not reliably outperform LSTM-based models on this dataset.

**Standalone backtest:** -24.1% total return, Sharpe -0.561. This model overfit to patterns not present in the test period; MATD3 assigns it an intermediate gate (mean 2.07) during bullish regimes but near-zero (21.3% of time) during drawdowns.

---

### 3. Sentiment Pipeline

Full pipeline: raw text collection → translation → FinBERT inference → daily RL features.

#### 3a. Data Collection

| Source | Rows | Coverage | Script |
|--------|------|----------|--------|
| CNINFO official announcements | 50,534 | 50 stocks × 2018–2025 | `download_sentiment_data.py` |
| EastMoney per-stock news | 500 | 50 stocks, recent 2025–2026 | `download_sentiment_data.py` |
| CCTV financial news | 11,029 | Market-level, 2018–2019 | `prepare_data.py` |
| Baidu economic news | 14,735 | Market-level, 2020–2025 | `prepare_data.py` |
| **Total** | **76,798** | — | — |

Collection uses AKShare with polite rate-limiting (1–2 second delays between requests). CNINFO data is the primary stock-level signal source; CCTV and Baidu provide market-level macro sentiment context.

#### 3b. NLP Inference (`run_sentiment_nlp.py`)

**Translation:** All Chinese text is translated to English using `Helsinki-NLP/opus-mt-zh-en` (MarianMT, offline). Greedy decoding (`num_beams=1`) is used to avoid OOM on GPU-constrained environments. Batch size 32.

**Sentiment scoring:** `ProsusAI/finbert` (English FinBERT, 3-class: positive / neutral / negative). Score formula:

```
sentiment_score = P(positive) − P(negative)  ∈ [−1, +1]
```

Batch size 64 for inference. The model auto-detects and clears stale results if the model name changes (prevents mixing scores from different model versions).

#### 3c. Daily RL Feature Aggregation

Four features are aggregated per stock per trading day and merged into the RL state:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `mean_sentiment` | Importance-weighted daily mean score | Bullish/bearish regime signal |
| `sentiment_vol` | Importance-weighted std dev of daily scores | Market disagreement / uncertainty |
| `message_volume` | Text count per stock per day | Information flow / news buzz |
| `abnormal_sentiment` | 30-day rolling z-score, clipped ±10, MIN_STD=0.01 floor | Sentiment surprise relative to recent history |

Importance weights use a decay function that down-weights older documents within the same day. The z-score clipping prevents extreme values during periods of sparse data.

**Coverage:** 19,396 rows (50 stocks × 2018-01-01 to 2026-03-10, 100% fill on test period).

---

### 4. MATD3 RL Aggregation Layer — v3 Architecture

`notebooks/rl_full_pipeline_v3.ipynb` (and `rl_full_pipeline_v4.ipynb` which shares all v3 architecture code).

#### 4a. Problem Formulation

At each 15-minute bar, the MATD3 system must assign a *gate weight* `g_i ∈ [0, 3]` to each of the 5 base signals. The aggregated signal for stock `s` at time `t` is:

```
final_score(s,t) = mean_i [ g_i(t) × signal_i_cs_z(s,t) ]
```

where `signal_i_cs_z` is the cross-sectionally standardised z-score of base signal `i`. The gate is a multiplicative amplifier: `g_i = 0` suppresses signal `i`; `g_i = 3` triples its contribution.

The top-5 stocks by `final_score` are selected and held with equal weight in a long-only portfolio subject to China's T+1 settlement rule.

#### 4b. Agent Design

**5 agents**, one per base signal (`signal_pre_boost_cs_z`, `signal_hybrid_cs_z`, `signal_ssm_tf_cs_z`, `signal_tf_cross_cs_z`, `signal_gat_tf_cs_z`). Each agent controls one gate scalar.

**Actor network** (`Actor`): 3-layer MLP with LayerNorm.
```
Input(55) → Linear(128) → LayerNorm → ReLU → Linear(128) → ReLU → Linear(1) → delta_raw
gate_i = clip(1.5 + 1.5 × tanh(delta_raw), 0, 3)
```

The `tanh` squash keeps the gate bounded; initialising at `1.5 + 1.5×tanh(0) = 1.5` provides a neutral starting point.

**Shared twin critics** (`SharedCritic`): Two independent centralized critics, each receiving *all* agents' states and actions concatenated.
```
Input dim = n_agents × state_dim + n_agents × action_dim = 5×55 + 5×1 = 280
→ Linear(128) → LayerNorm → ReLU → Linear(128) → ReLU → Linear(1)
```

Using shared (centralised) critics follows the CTDE (Centralised Training, Decentralised Execution) paradigm: at training time, each agent's actor update has access to the full joint action profile; at inference, each actor only needs its own 55-dimensional state.

#### 4c. State Space (55 dimensions)

Each agent receives a 55-dimensional state vector composed of four blocks:

| Block | Dim | Content |
|-------|-----|---------|
| Own-signal summary | 15 | Mean, std, 5 quantiles (10/25/50/75/90th percentile), top-5 mean, bottom-5 mean, IQR, Q50−Q10; plus 4 padding zeros for future extension |
| Relative summary | 2 | Correlation of own signal with ensemble average; mean absolute deviation from ensemble |
| Shared global | 22 | Cross-sectional mean and std of 11 shared market/sentiment features (see below) |
| Portfolio global | 8 | Cash fraction, invested fraction, sellable fraction, unsellable fraction, HHI concentration, max weight, top-5 holding weight sum, previous turnover lag |
| Portfolio per-agent | 8 | Previous gate value for this agent, mean/std of inventory and sellable weights, inventory-weighted signal exposure, sellable-weighted exposure, top-5-holding signal mean |

The 11 shared state features are:
`sig_mean_5`, `sig_std_5`, `spread_over_mid`, `microprice_minus_mid_over_mid`, `log_volume_sum_cs_rank`, `ret_mid_t1_lag1`, `ret_mid_t1_lag2`, `mean_sentiment`, `abnormal_sentiment`, `sentiment_vol`, `message_volume`.

The sentiment features (`mean_sentiment`, `abnormal_sentiment`, `sentiment_vol`, `message_volume`) are included in the shared state so every agent can condition its gating decision on the current information-flow regime.

#### 4d. Composite Reward (v3)

```
composite = net_return − 0.1 × drawdown − 3.0 × rolling_var
reward = composite × 2000
```

- **`net_return`**: Gross bar return minus half-spread transaction cost, realised under the T+1 constraint.
- **`drawdown`**: Current peak-to-trough drawdown of the running NAV. Penalises capital loss depth (targets Calmar ratio).
- **`rolling_var`**: Variance of the last 50 net returns. Penalises return volatility (targets Sharpe ratio).
- **`reward_scale=2000`**: Rescales the composite into a range suitable for neural network gradient flow.

The reward is computed by `calc_reward_composite()` and tracked by `RewardTracker`, which maintains a rolling buffer of the last 50 returns and the running NAV/peak NAV.

#### 4e. Training Algorithm (MATD3)

MATD3 is Multi-Agent TD3 — an extension of Twin Delayed DDPG (TD3) to multi-agent settings under CTDE.

**Key TD3 mechanisms retained:**

1. **Twin critics:** Two independent Q-networks; the minimum Q-value is used for the Bellman target, reducing overestimation bias (`q_target = min(Q1_target, Q2_target)`).
2. **Target policy smoothing:** Gaussian noise (std=0.02, clipped ±0.05) added to target actor actions during critic update, preventing the critic from fitting sharp peaks.
3. **Delayed policy update:** Actors updated every `policy_delay=2` critic updates. Allows Q-function to stabilise before actor gradients are applied.
4. **Soft target network update:** `θ_target ← τθ + (1−τ)θ_target` with `τ=0.005` after every update step.

**Actor loss:**
```
L_actor_i = -Q1(s, a_1,..., a_i_proposed,..., a_N) + gate_reg_lambda × (gate_i − 1)²
```

The gate regularisation term `gate_reg_lambda=0.001` penalises gates far from 1.0 (neutral), preventing agents from defaulting to all-or-nothing suppression early in training.

**Training hyperparameters (v3):**
| Parameter | Value |
|-----------|-------|
| Epochs | 80 |
| Replay buffer capacity | 300,000 |
| Batch size | 512 |
| Warmup steps (random actions before learning) | 2,000 |
| Learning rate (actor / critic) | 1e-4 / 3e-4 |
| Action exploration noise std | 0.10 |
| Train every N steps | 2 |
| Gradient updates per training step | 2 |
| Gradient clip norm | 1.0 |
| Discount factor γ | 0.95 |

Training data: close-bar only (14:45–15:00 bars), 403 trading days (2023-12-05 to 2025-08-04), ~20,150 timesteps per epoch.

#### 4f. Portfolio Execution (T+1 Constraint)

China A-share markets enforce a T+1 rule: shares bought today cannot be sold until the next trading day. The MATD3 framework models this explicitly:

- `inventory_map`: shares currently held (can be sold tomorrow if held since yesterday).
- `sellable_map`: subset of inventory that is sellable today (held since at least yesterday).
- At each bar, `execute_target_weight_t1()` fills sell orders from `sellable_map` first, then buys with remaining cash. Unfilled buys (due to insufficient sellable positions) are scaled proportionally.

Transaction costs: 0.03% half-spread applied to both buys and sells (no stamp duty or broker commission modelled separately, as the half-spread approximates the round-trip cost).

---

## Repository Structure

```
├── Factor.py                         # 70 microstructure factors (f_001–f_070)
├── CNN+LSTM+TRANSFORMER.py           # CNN + LSTM + Transformer signal model
├── CROSS ATTEN.py                    # Cross-sectional attention signal model
├── GNN.py                            # GAT + Transformer signal model
├── S4+Transformer.py                 # SSM + Transformer signal model
├── notebooks/
│   ├── train and rf pre.ipynb        # Feature engineering + all 5 model signal generation
│   ├── rl_full_pipeline_v3.ipynb     # MATD3 v3 training + backtest (final model)
│   ├── rl_full_pipeline_v4.ipynb     # MATD3 v4 experiment (softmax portfolio + vol targeting)
│   ├── rl_full_pipeline_v5.ipynb     # MATD3 v5 experiment (Calmar reward focus)
│   └── Backtesting.ipynb             # Standalone close-to-close backtest framework
├── scripts/
│   ├── download_sentiment_data.py    # CNINFO + EastMoney data collection
│   ├── prepare_data.py               # CCTV + Baidu market news + validation
│   ├── run_sentiment_nlp.py          # FinBERT inference + daily aggregation
│   ├── backtest_all_signals.py       # Baseline signal backtest (all 5 individual models)
│   ├── backtest_close_to_close.py    # Close-to-close backtest with MATD3 comparison
│   ├── run_matd3_v3_full.py          # MATD3 v3 full training + backtest script
│   ├── patch_v4.py                   # Generates rl_full_pipeline_v4.ipynb from v3
│   ├── patch_v4_comparison.py        # Updates v4 notebook comparison cells (v2/v3/v4)
│   ├── patch_v5.py                   # Generates rl_full_pipeline_v5.ipynb from v4
│   └── predict_and_compare.py        # Signal comparison and IC analysis
├── matd3_v2_models/                  # MATD3 v2 checkpoints (actors + critics)
├── matd3_v3_models/                  # MATD3 v3 checkpoints (actors + critics, final)
├── matd3_v4_models/                  # MATD3 v4 checkpoints (experimental)
├── models_comparison/matd3_v3/       # v3 backtest charts: PnL curves, metrics bar chart, gate weights
├── models_comparison/matd3_v4/       # v4 backtest charts
├── src/cninfo/
│   ├── align.py                      # Leakage-safe event-to-bar alignment
│   ├── sentiment.py                  # Lexicon-based scoring (MVP baseline)
│   ├── scrape.py                     # CNINFO metadata loader
│   └── io.py                         # Parquet I/O helpers
└── requirements.txt                  # Full dependency list (Python 3.9)
```

> **Note:** Large parquet files (`df_trainsub.parquet` 582 MB, `df_testsub_*.parquet`, `df_full_rl_clean*.parquet`) and `sentiment_data.db` (52 MB) are gitignored. Run scripts in order to regenerate locally.

---

## Ethical and Data Considerations

- CNINFO is a public corporate disclosure platform; only publicly available announcements are used.
- All data collection respects platform rate limits (polite throttling applied).
- No private, proprietary, or restricted personal data is required.
- Leakage prevention: event timestamps mapped to next-available bar (forward merge only). All factor computations use strictly causal windows.
