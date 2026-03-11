# Adaptive Multi-Factor Aggregation with Sentiment Context on CSI 300 (15-Minute Horizon)

## Executive Summary
This capstone project improves short-horizon equity signal aggregation in the CSI 300 universe by combining 70 price-derived microstructure factors with disclosure-based sentiment context. The core contribution is dynamic factor weighting: an RL aggregation layer learns when each factor should receive more or less weight, conditioned on current market state and sentiment regime.

---

## Progress Update (2026-03-11)

### ✅ Completed

#### 1. Microstructure Factor Engineering
- **70 factors** (`Factor.py`, `f_001`–`f_070`) computed on 1.5M rows of 15-minute intraday bar data (2018–2025)
- Covers: bid-ask spread, order imbalance, OFI, Amihud illiquidity, microprice deviation, volume Z-scores, etc.
- **LightGBM baseline**: IC = 0.0675, IC IR = **1.42** on 2024–2025 test set

#### 2. Deep Learning Models (trained, checkpoints saved)
| Model | Checkpoint | Architecture |
|-------|-----------|--------------|
| CNN + LSTM + Transformer | `best_hybrid.pt` | Conv1D → LSTM → Transformer encoder |
| GAT + Transformer | `best_gat_tf.pt` | Return-correlation graph → GAT → Transformer |
| SSM + Transformer | `best_ssm_tf.pt` | Discrete state-space model → Transformer |

> All three trained on 70 microstructure factors only (sentiment not yet integrated).

#### 3. Sentiment Pipeline (end-to-end, production-complete)
Full pipeline from raw data collection to daily RL-ready features:

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
- Label distribution: neutral 92.3% / negative 4.3% / positive 3.5%
- Score formula: `sentiment_score = P(positive) − P(negative) ∈ [−1, +1]`

**Daily RL Features** (`daily_sentiment_features` table, 19,396 rows)
| Feature | Description |
|---------|-------------|
| `mean_sentiment` | Importance-weighted daily sentiment score |
| `sentiment_vol` | Weighted std dev (market disagreement) |
| `message_volume` | Text count (information flow / buzz) |
| `abnormal_sentiment` | 30-day rolling z-score (sentiment shock detector, clipped ±10) |

Scripts: `scripts/download_sentiment_data.py` → `scripts/prepare_data.py` → `scripts/run_sentiment_nlp.py`

---

### 🔲 In Progress / Next Steps

#### Step 1 — Integrate Sentiment into Training Data (immediate)
Merge `daily_sentiment_features` with 15-minute bar data on `(SecuCode, TradingDay)`.
Sentiment features become `f_071`–`f_074`, expanding the feature set from 70 → 74 columns.

```python
df_merged = df_bars.merge(df_sentiment,
    left_on=["SecuCode", "TradingDay"],
    right_on=["SecuCode", "trade_date"], how="left")
df_merged[sentiment_cols] = df_merged.groupby("SecuCode")[sentiment_cols].ffill()
```

#### Step 2 — Retrain Models with Sentiment Features
Re-run CNN+LSTM+TF, GAT+TF, SSM+TF on 74-feature set.
Measure IC improvement from adding sentiment context.

#### Step 3 — Train Cross-Attention Model
`CROSS ATTEN.py` is written but no checkpoint exists (`best_tf_cross.pt` missing).
This model captures cross-sectional dependencies by attending across stocks at the same timestamp.

#### Step 4 — Build RL Aggregation Layer (core innovation)
Design and implement the RL policy that dynamically weights the four model predictions
as a function of current market state + sentiment regime. No code exists for this yet.

#### Step 5 — Unified Backtest
Compare: LightGBM baseline → individual deep models → RL-weighted ensemble
Metrics: IC, IC IR, annual return, Sharpe ratio, max drawdown (2024–2025 test period)

---

## Repository Structure
```
├── Factor.py                    # 70 microstructure factors (f_001–f_070)
├── CNN+LSTM+TRANSFORMER.py      # Hybrid deep learning model
├── CROSS ATTEN.py               # Cross-sectional attention model
├── GNN.py                       # Graph attention network model
├── S4+Transformer.py            # State space model + Transformer
├── scripts/
│   ├── download_sentiment_data.py   # CNINFO + EastMoney data collection
│   ├── prepare_data.py              # CCTV + Baidu market news + validation
│   ├── run_sentiment_nlp.py         # FinBERT inference + daily aggregation
│   └── pipeline_chain.py            # Auto-chained pipeline executor
├── src/cninfo/
│   ├── align.py                 # Leakage-safe event-to-bar alignment
│   ├── sentiment.py             # Lexicon-based scoring (MVP baseline)
│   ├── scrape.py                # CNINFO metadata loader
│   └── io.py                    # Parquet I/O helpers
├── notebooks/
│   ├── sentiment_pipeline_mvp.ipynb
│   └── Factor and boosting.ipynb
├── SENTIMENT_PIPELINE_REPORT.md # Full sentiment pipeline results & SQL examples
└── requirements.txt             # Full dependency list (Python 3.9)
```

> **Note:** `sentiment_data.db` (52MB SQLite) is gitignored. Run the scripts in order to regenerate locally.

---

## Ethical and Data Considerations
- CNINFO is a public corporate disclosure platform; only publicly available announcements are used.
- All data collection respects platform rate limits (polite throttling applied).
- No private, proprietary, or restricted personal data is required.
- Leakage prevention: event timestamps mapped to next-available bar (forward merge only).
