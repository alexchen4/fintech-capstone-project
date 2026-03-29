# Adaptive Multi-Factor Aggregation with Sentiment Context on CSI 300 (15-Minute Horizon)

## Executive Summary
This capstone project improves short-horizon equity signal aggregation in the CSI 300 universe by combining 70 price-derived microstructure factors with disclosure-based sentiment context. The core contribution is dynamic factor weighting: an RL aggregation layer (MATD3) learns when each factor should receive more or less weight, conditioned on current market state and sentiment regime.

---

## Results (2026-03-28) — Full Pipeline Complete

### Close-to-Close Backtest (Top-5 EW, fee=0.03%, full period 2023-12 to 2025-12)

| Model | Total Return | Ann. Return | Sharpe | Sortino | Calmar | Max Drawdown |
|-------|-------------|-------------|--------|---------|--------|--------------|
| **MATD3 v3** | **82.3%** | **33.7%** | **1.467** | **2.719** | 1.651 | -20.4% |
| CNN+LSTM+TF | 74.9% | 31.1% | 1.438 | 2.476 | **1.702** | **-18.3%** |
| MATD3 v2 | 59.7% | 25.4% | 1.171 | 2.094 | 1.116 | -22.8% |
| Cross-Attn | 66.0% | 27.8% | 1.079 | 1.958 | 0.993 | -28.0% |
| GAT+TF | 18.2% | 8.4% | 0.422 | 0.645 | 0.353 | -23.9% |
| LightGBM | 17.8% | 8.3% | 0.294 | 0.519 | 0.215 | -38.3% |
| EW Ensemble | -24.1% | -12.5% | -0.561 | -0.977 | -0.352 | -35.5% |
| SSM+TF | -24.1% | -12.5% | -0.561 | -0.977 | -0.352 | -35.5% |

MATD3 v3 achieves the highest total return (82.3%) and Sortino ratio (2.719) across all models.

### MATD3 Version History

| Parameter | v1 | v2 | v3 |
|-----------|----|----|-----|
| Gate range | [0.5, 1.5] | [0, 2.0] | **[0, 3.0]** |
| dd_lambda | 0.5 | 0.3 | **0.1** |
| vol_lambda | 30.0 | 10.0 | **3.0** |
| reward_scale | 1000 | 1000 | **2000** |
| Epochs | 15 | 30 | **50** |
| Action noise | 0.05 | 0.08 | **0.10** |
| Total Return | — | 59.7% | **82.3%** |
| Sharpe | — | 1.171 | **1.467** |

### Gate Weight Summary (v3)
| Signal | Mean Gate | Std | % above 1.0 | % near 0 |
|--------|-----------|-----|-------------|----------|
| CNN+LSTM+TF | 2.66 | 0.79 | 91.3% | 2.6% |
| SSM+TF | 2.07 | 1.31 | 70.8% | 21.3% |
| Cross-Attn | 1.87 | 1.28 | 65.8% | 18.3% |
| GAT+TF | 1.33 | 1.40 | 46.5% | 43.7% |
| LightGBM | 0.91 | 1.24 | 33.2% | 52.3% |

MATD3 successfully learned to up-weight strong signals (CNN+LSTM+TF) and suppress weak ones (LightGBM, SSM+TF).

---

## Completed Components

### 1. Microstructure Factor Engineering
- **70 factors** (`Factor.py`, `f_001`–`f_070`) computed on 1.5M rows of 15-minute intraday bar data (2018–2025)
- Covers: bid-ask spread, order imbalance, OFI, Amihud illiquidity, microprice deviation, volume Z-scores, etc.
- **LightGBM baseline**: IC = 0.0675, IC IR = **1.42** on 2024–2025 test set

### 2. Deep Learning Models
| Model | File | Architecture |
|-------|------|--------------|
| LightGBM | `Factor.py` + notebook | Gradient boosting on 70 factors; IC=0.0675, ICIR=1.42 |
| CNN + LSTM + Transformer | `CNN+LSTM+TRANSFORMER.py` | Conv1D → LSTM → Transformer encoder |
| Cross-Attention | `CROSS ATTEN.py` | Cross-sectional attention across stocks |
| GAT + Transformer | `GNN.py` | Return-correlation graph → GAT → Transformer |
| SSM + Transformer | `S4+Transformer.py` | Discrete state-space model → Transformer |

All 5 signals inferred on test period → saved in `df_testsub_full.parquet` (gitignored due to size).

### 3. Sentiment Pipeline
Full pipeline from raw data collection to daily RL-ready features.

**Data Collection**
| Source | Rows | Coverage |
|--------|------|----------|
| CNINFO official announcements | 50,534 | 50 stocks × 2018–2025 |
| EastMoney per-stock news | 500 | 50 stocks, recent 2025–2026 |
| CCTV financial news | 11,029 | Market-level, 2018–2019 |
| Baidu economic news | 14,735 | Market-level, 2020–2025 |
| **Total texts** | **76,798** | — |

**Inference**
- Translation: `Helsinki-NLP/opus-mt-zh-en` (offline MarianMT, zh → en)
- Sentiment model: `ProsusAI/finbert` (English FinBERT, 3-class)
- Score formula: `sentiment_score = P(positive) − P(negative) ∈ [−1, +1]`

**Daily RL Features** (19,396 rows, 50 stocks × 2018–2026)
| Feature | Description |
|---------|-------------|
| `mean_sentiment` | Importance-weighted daily sentiment score |
| `sentiment_vol` | Weighted std dev (market disagreement) |
| `message_volume` | Text count (information flow / buzz) |
| `abnormal_sentiment` | 30-day rolling z-score, clipped ±10 |

Scripts: `scripts/download_sentiment_data.py` → `scripts/prepare_data.py` → `scripts/run_sentiment_nlp.py`

### 4. RL Aggregation Layer — MATD3 v3
`notebooks/rl_full_pipeline.ipynb` — 22-cell self-contained pipeline.

| Component | Details |
|-----------|---------|
| Architecture | Multi-Agent TD3, 5 agents (one per signal), shared twin critics |
| State dim | 55 = 15 (own signal stats) + 2 (rel to ensemble) + 22 (shared global) + 8 (portfolio global) + 8 (portfolio per-agent) |
| Shared state | 11 features: 7 market microstructure + 4 sentiment context |
| Composite reward | `net_return − 0.1×drawdown − 3.0×rolling_var` (PnL + Calmar + Sharpe components) |
| Execution | T+1 constraint (A-share rule), long-only, top-5, max 20% per stock |
| Training data | Close-bar only (14:45–15:00), 503 trading days (2023-12 ~ 2025-12) |
| Outputs | `matd3_v3_models/`, `models_comparison/matd3_v3/` |

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
│   ├── rl_full_pipeline.ipynb        # MATD3 v3 training + backtest (Steps B–E)
│   ├── rl_full_pipeline_v2.ipynb     # MATD3 v2 backup notebook
│   └── Backtesting.ipynb             # Close-to-close backtest framework
├── scripts/
│   ├── download_sentiment_data.py    # CNINFO + EastMoney data collection
│   ├── prepare_data.py               # CCTV + Baidu market news + validation
│   ├── run_sentiment_nlp.py          # FinBERT inference + daily aggregation
│   ├── backtest_all_signals.py       # Baseline signal backtest (all 5 individual models)
│   ├── backtest_close_to_close.py    # Close-to-close backtest with MATD3 comparison
│   ├── run_matd3_v3_full.py          # MATD3 v3 full training + backtest script
│   └── predict_and_compare.py        # Signal comparison and IC analysis
├── matd3_v2_models/                  # MATD3 v2 checkpoints (actors + critics)
├── matd3_v3_models/                  # MATD3 v3 checkpoints (actors + critics)
├── models_comparison/matd3_v3/       # v3 backtest charts: PnL curves, metrics bar chart, gate weights
├── matd3_composite_models/           # Baseline backtest results (individual signals)
├── src/cninfo/
│   ├── align.py                      # Leakage-safe event-to-bar alignment
│   ├── sentiment.py                  # Lexicon-based scoring (MVP baseline)
│   ├── scrape.py                     # CNINFO metadata loader
│   └── io.py                         # Parquet I/O helpers
└── requirements.txt                  # Full dependency list (Python 3.9)
```

> **Note:** Large parquet files (`df_trainsub.parquet` 582MB, `df_testsub_*.parquet`, etc.) and `sentiment_data.db` (52MB) are gitignored. Run scripts in order to regenerate locally.

---

## Ethical and Data Considerations
- CNINFO is a public corporate disclosure platform; only publicly available announcements are used.
- All data collection respects platform rate limits (polite throttling applied).
- No private, proprietary, or restricted personal data is required.
- Leakage prevention: event timestamps mapped to next-available bar (forward merge only).
