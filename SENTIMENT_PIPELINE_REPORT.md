# Sentiment Pipeline Results Report
**Generated:** 2026-03-11
**Project:** CSI 300 Adaptive Multi-Factor RL Trading System
**Database:** `sentiment_data.db`

---

## 1. Pipeline Overview

```
CNINFO (AKShare) ─────────────────────────┐
EastMoney News (AKShare) ─────────────────┤──► sentiment_data.db
CCTV Financial News (AKShare) ────────────┤        │
Baidu Economic News (AKShare) ────────────┘        │
                                                    ▼
                                        MarianMT (zh → en)
                                                    │
                                                    ▼
                                        ProsusAI/finbert
                                        (3-class: pos/neu/neg)
                                                    │
                                                    ▼
                                    daily_sentiment_features
                                    (RL input features, per stock per day)
```

---

## 2. Raw Data Collection

| Table | Rows | Coverage | Source |
|-------|------|----------|--------|
| `announcements` | **50,534** | 50 stocks × 2018-01-01 ~ 2025-12-31 | CNINFO official disclosures (6-month segments) |
| `news_em` | **500** | 50 stocks, recent 2025-2026 | EastMoney per-stock news |
| `market_news` | **25,764** | 2018-01-01 ~ 2025-12-31 | CCTV financial news (2018-2019) + Baidu economic news (2020-2025) |
| **Total texts** | **76,798** | — | — |

### Stock Universe (50 stocks)
```
000400 000415 000423 000425 000503 000559 000581 000629 000738 000778
000826 000831 000858 001979 002129 002142 002153 002236 002353 002456
002594 300024 300058 300146 600009 600030 600038 600085 600111 600309
600332 600362 600398 600570 600585 600642 600663 600688 600999 601018
601169 601231 601328 601555 601718 601818 601857 601933 601988 603000
```

### Announcement Importance Ranking
CNINFO announcements are weighted by title keyword importance:

| Rank | Category | Weight (1/rank) |
|------|----------|-----------------|
| 1 | Annual Report | 1.000 |
| 2 | Semi-annual Report | 0.500 |
| 3 | Quarterly Report | 0.333 |
| 4 | Major Event / M&A | 0.250 |
| 6 | Dividend | 0.167 |
| 99 | General / Other | 0.010 |

---

## 3. Sentiment Inference

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Sentiment model | `ProsusAI/finbert` (English FinBERT, 3-class) |
| Translation model | `Helsinki-NLP/opus-mt-zh-en` (MarianMT, offline) |
| Inference device | CPU |
| Batch size (inference) | 64 |
| Batch size (translation) | 32 |
| Translation mode | Greedy decoding (`num_beams=1`) |
| Max token length | 128 |

### Label Distribution

| Label | Count | Percentage |
|-------|-------|-----------|
| neutral | 70,862 | **92.3%** |
| negative | 3,276 | 4.3% |
| positive | 2,660 | 3.5% |

> **Note:** High neutral ratio is expected — official regulatory disclosures (CNINFO)
> are factual in nature and do not carry strong sentiment signal. The ~8% pos/neg
> texts are the key signal carriers for the RL agent.

### Sentiment Score by Source

| Source | Rows | Avg Sentiment Score |
|--------|------|---------------------|
| `announcements` | 50,534 | -0.0029 (slightly negative) |
| `market_news` | 25,764 | -0.0123 (market news skews negative) |
| `news_em` | 500 | +0.0562 (EastMoney skews slightly positive) |

**Sentiment score formula:** `score = P(positive) − P(negative) ∈ [−1, +1]`

---

## 4. Daily RL Feature Table (`daily_sentiment_features`)

### Coverage
| Metric | Value |
|--------|-------|
| Total rows | **19,396** |
| Stocks covered | 51 (50 stocks + MARKET) |
| Date range | **2018-01-01 ~ 2026-03-10** |
| Avg texts per stock-day | 4.0 |

### Feature Definitions

| Feature | Formula | Description |
|---------|---------|-------------|
| `mean_sentiment` | Weighted avg of `P(pos)−P(neg)` | Daily sentiment signal; range `[−1, +1]` |
| `sentiment_vol` | Weighted std dev of sentiment scores | Market disagreement / uncertainty |
| `message_volume` | Count of texts on that day | Information flow / buzz level |
| `abnormal_sentiment` | z-score vs 30-day rolling mean | Sentiment shock detector |
| `roll_mean_30d` | 30-day rolling mean of `mean_sentiment` | Trend baseline |
| `roll_std_30d` | 30-day rolling std | Historical volatility of sentiment |
| `pos_ratio` | Fraction of texts with `P(pos) > 0.5` | Bullish signal fraction |
| `neg_ratio` | Fraction of texts with `P(neg) > 0.5` | Bearish signal fraction |
| `ann_volume` | Count from `announcements` table | Official disclosure activity |
| `news_volume` | Count from `news_em` table | Media coverage activity |

### Feature Statistics

| Feature | Mean | Notes |
|---------|------|-------|
| `mean_sentiment` | 0.0005 | Near-zero mean, range [−0.966, +0.934] |
| `sentiment_vol` | 0.0411 | Low volatility on most days |
| `message_volume` | 4.0 | Average 4 texts per stock per trading day |
| `\|abnormal_sentiment\|` | 0.7387 | Healthy z-score distribution |
| rows with `\|z\| > 3` | 900 | 4.6% of rows — genuine sentiment shocks |

---

## 5. Data Quality & Bug Fixes

### Issues Found and Fixed

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| `abnormal_sentiment` avg = 352 (exploding z-score) | `rolling_std ≈ 0` → division by `1e-8` → values up to ±6,800,000 | Added `MIN_STD = 0.01` floor + `clip(−10, 10)` |
| Garbled output `鈥?` in terminal | UTF-8 em dash `—` misread by Windows GBK terminal | Replaced with ASCII `--` |
| `bad allocation` OOM crash | `TRANS_BATCH_SIZE=128` + `num_beams=2` exhausted RAM | Reduced to `TRANS_BATCH_SIZE=32`, `num_beams=1` |
| Inference speed: 20-25s/batch | CPU thermal throttling after 15hr run + old batch size | Restarted with High Performance power plan + `BATCH_SIZE=64` |

---

## 6. How to Use in RL Agent

### Query Example
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('sentiment_data.db')

# Get all sentiment features for a specific stock
df = pd.read_sql("""
    SELECT SecuCode, trade_date,
           mean_sentiment,
           sentiment_vol,
           message_volume,
           abnormal_sentiment,
           pos_ratio,
           neg_ratio,
           ann_volume
    FROM daily_sentiment_features
    WHERE SecuCode = '000858'
    ORDER BY trade_date
""", conn)

conn.close()
print(df.tail())
```

### Merge with Price Data
```python
# Align sentiment features with 15-min bar data
# sentiment features are daily -> forward-fill within each trading day
df_bars['trade_date'] = df_bars['TradingDay'].astype(str)

df_merged = df_bars.merge(
    df_sentiment,
    on=['SecuCode', 'trade_date'],
    how='left'
)

# Forward-fill intraday (same sentiment applies all day)
sentiment_cols = ['mean_sentiment', 'sentiment_vol', 'message_volume',
                  'abnormal_sentiment', 'pos_ratio', 'neg_ratio']
df_merged[sentiment_cols] = df_merged.groupby('SecuCode')[sentiment_cols].ffill()
```

### Recommended RL Features
For the RL state vector, use:
- `mean_sentiment` — primary directional signal
- `abnormal_sentiment` — detects regime shifts / shocks
- `message_volume` — attention/information flow proxy
- `sentiment_vol` — uncertainty / disagreement measure

---

## 7. Next Steps

- [ ] Merge `daily_sentiment_features` with 15-min OHLCV bar data (`qfq_15min_all.csv`)
- [ ] Compute 70 microstructure factors (`Factor.py`, factors 001-070)
- [ ] Add sentiment features to factor matrix as additional columns
- [ ] Train hybrid models: CNN+LSTM+Transformer, GNN, S4+Transformer
- [ ] Backtest RL agent with combined factor + sentiment state vector
