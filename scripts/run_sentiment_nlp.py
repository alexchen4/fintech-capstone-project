"""
NolBERT / FinBERT Sentiment Inference + Daily RL Feature Aggregation
=====================================================================
Input  : sentiment_data.db  (tables: announcements, news_em, market_news)
Output : sentiment_data.db  (tables: sentiment_raw, daily_sentiment_features)

Daily RL input features (per SecuCode per trade_date):
  mean_sentiment      - weighted average sentiment score in [-1, 1]
  sentiment_vol       - sentiment standard deviation (market disagreement)
  message_volume      - total text count (attention / buzz level)
  abnormal_sentiment  - z-score vs 30-day rolling historical mean

Translation strategy:
  Raw texts are in Chinese; NolBERT only supports English.
  Uses Helsinki-NLP/opus-mt-zh-en (offline MarianMT) to translate
  each batch before passing to NolBERT/FinBERT.

Model options:
  Default : ProsusAI/finbert  (English FinBERT, 3-class)
  Replace : set MODEL_NAME to any English HuggingFace model path
  NolBERT : set MODEL_NAME = "path/to/nolbert" for local weights
  Alt     : "yiyanghkust/finbert-tone" (English FinBERT tone)

Weighting strategy:
  announcements : weighted by importance_rank (annual report = highest)
  news_em       : weighted by source_rank
  market_news   : uniform weight (importance_w = 1.0)

Python 3.9 compatible (no X|Y union type hints)
"""

from __future__ import annotations

import io
import math
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "sentiment_data.db"

# Sentiment model (English NolBERT / FinBERT)
# Original NolBERT is English-only; Chinese texts are translated first
MODEL_NAME = "ProsusAI/finbert"
# For local NolBERT weights:  MODEL_NAME = r"path\to\nolbert"
# Alternative English models:
#   "yiyanghkust/finbert-tone"                     (English FinBERT tone, 3-class)
#   "ahmedrachid/FinancialBERT-Sentiment-Analysis"  (English financial BERT)

# Chinese -> English translation (offline MarianMT, no external API required)
TRANSLATE_ENABLED = True
TRANS_MODEL_NAME  = "Helsinki-NLP/opus-mt-zh-en"
TRANS_BATCH_SIZE  = 32    # translation batch size (keep small to avoid OOM with beam search)

BATCH_SIZE     = 64   # inference batch size (FinBERT inference only, safe at 64)
MAX_LEN        = 128  # max token length (announcement titles are usually <64)
ROLLING_WINDOW = 30   # rolling window in calendar days for abnormal_sentiment
WEIGHT_BY_RANK = True # whether to weight by importance_rank / source_rank

# CPU throughput estimate:
#   batch_size=64 reduces per-item overhead vs batch_size=32
#   Translation + inference combined: ~8-12s per batch of 64 on CPU

# Label -> sentiment score mapping (positive=+1, neutral=0, negative=-1)
# ProsusAI/finbert labels: positive / negative / neutral
LABEL_SCORE_MAP: Dict[str, float] = {
    "positive": +1.0, "pos": +1.0,
    "neutral":   0.0, "neu":  0.0,
    "negative": -1.0, "neg": -1.0,
    # Generic LABEL_x fallback (index order varies by model)
    "LABEL_0": +1.0,   # finbert: positive
    "LABEL_1":  0.0,   # finbert: neutral
    "LABEL_2": -1.0,   # finbert: negative
}

# ──────────────────────────────────────────────────────────────
# Database initialization
# ──────────────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sentiment_raw (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_table    TEXT    NOT NULL,   -- 'announcements', 'news_em', or 'market_news'
    source_id       INTEGER NOT NULL,   -- primary key from source table
    SecuCode        TEXT    NOT NULL,
    text_date       TEXT    NOT NULL,   -- YYYY-MM-DD
    text_input      TEXT,               -- translated English text used for inference
    label           TEXT,               -- raw model label
    score_raw       REAL,               -- argmax confidence score
    prob_pos        REAL,               -- P(positive)
    prob_neu        REAL,               -- P(neutral)
    prob_neg        REAL,               -- P(negative)
    sentiment_score REAL,               -- final score = P(pos) - P(neg) in [-1, 1]
    importance_w    REAL,               -- weight (1/importance_rank or 1/source_rank)
    model_name      TEXT,
    UNIQUE(source_table, source_id)
);

CREATE TABLE IF NOT EXISTS daily_sentiment_features (
    SecuCode              TEXT    NOT NULL,
    trade_date            TEXT    NOT NULL,
    mean_sentiment        REAL,   -- weighted mean daily sentiment
    sentiment_vol         REAL,   -- sentiment standard deviation (disagreement)
    message_volume        INTEGER,-- number of texts on this day
    abnormal_sentiment    REAL,   -- z-score vs 30-day rolling mean
    roll_mean_30d         REAL,   -- 30-day rolling mean (intermediate)
    roll_std_30d          REAL,   -- 30-day rolling std dev
    weighted_vol          REAL,   -- unweighted sentiment std dev (supplement)
    pos_ratio             REAL,   -- fraction of positive texts
    neg_ratio             REAL,   -- fraction of negative texts
    ann_volume            INTEGER,-- count from announcements table
    news_volume           INTEGER,-- count from news_em table
    PRIMARY KEY (SecuCode, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_raw_code  ON sentiment_raw(SecuCode);
CREATE INDEX IF NOT EXISTS idx_raw_date  ON sentiment_raw(text_date);
CREATE INDEX IF NOT EXISTS idx_feat_code ON daily_sentiment_features(SecuCode);
CREATE INDEX IF NOT EXISTS idx_feat_date ON daily_sentiment_features(trade_date);
"""


def init_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    print("[DB] Schema ready")


# ──────────────────────────────────────────────────────────────
# Load pending texts (skip already-processed ids)
# ──────────────────────────────────────────────────────────────
def load_pending(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load unprocessed records from announcements + news_em + market_news."""
    done_ann = set(
        r[0] for r in conn.execute(
            "SELECT source_id FROM sentiment_raw WHERE source_table='announcements'"
        )
    )
    done_news = set(
        r[0] for r in conn.execute(
            "SELECT source_id FROM sentiment_raw WHERE source_table='news_em'"
        )
    )
    done_mkt = set(
        r[0] for r in conn.execute(
            "SELECT source_id FROM sentiment_raw WHERE source_table='market_news'"
        )
    )

    # -- announcements --
    df_ann = pd.read_sql(
        "SELECT id, SecuCode, announce_dt, title, importance_rank FROM announcements",
        conn
    )
    df_ann = df_ann[~df_ann["id"].isin(done_ann)].copy()
    df_ann["source_table"] = "announcements"
    df_ann["source_id"]    = df_ann["id"]
    df_ann["text_date"]    = pd.to_datetime(df_ann["announce_dt"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_ann["text_input"]   = df_ann["title"].fillna("").str[:MAX_LEN * 2]
    # Lower importance_rank = higher importance -> higher weight
    df_ann["importance_w"] = 1.0 / df_ann["importance_rank"].clip(lower=1)

    # -- news_em --
    df_news = pd.read_sql(
        "SELECT id, SecuCode, news_dt, title, content, source_rank FROM news_em",
        conn
    )
    if not df_news.empty:
        df_news = df_news[~df_news["id"].isin(done_news)].copy()
        df_news["source_table"] = "news_em"
        df_news["source_id"]    = df_news["id"]
        df_news["text_date"]    = pd.to_datetime(df_news["news_dt"], errors="coerce").dt.strftime("%Y-%m-%d")
        # Concatenate title + content, truncate to MAX_LEN*2 chars
        df_news["text_input"]   = (
            df_news["title"].fillna("") + " " + df_news["content"].fillna("")
        ).str[:MAX_LEN * 2]
        df_news["importance_w"] = 1.0 / df_news["source_rank"].clip(lower=1)
    else:
        df_news = pd.DataFrame(columns=df_ann.columns)

    # -- market_news (Baidu/CCTV market-level news, SecuCode='MARKET') --
    # Check table exists before querying
    tbl_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='market_news'"
    ).fetchone()
    if tbl_exists:
        df_mkt = pd.read_sql(
            "SELECT id, news_dt, title, event_tag FROM market_news",
            conn
        )
    else:
        df_mkt = pd.DataFrame()

    if not df_mkt.empty:
        df_mkt = df_mkt[~df_mkt["id"].isin(done_mkt)].copy()
        df_mkt["source_table"] = "market_news"
        df_mkt["source_id"]    = df_mkt["id"]
        df_mkt["SecuCode"]     = "MARKET"
        df_mkt["text_date"]    = pd.to_datetime(df_mkt["news_dt"], errors="coerce").dt.strftime("%Y-%m-%d")
        # Concatenate title + event_tag
        df_mkt["text_input"]   = (
            df_mkt["title"].fillna("") + " " + df_mkt["event_tag"].fillna("")
        ).str[:MAX_LEN * 2]
        df_mkt["importance_w"] = 1.0   # uniform weight for market-level news
    else:
        df_mkt = pd.DataFrame(columns=df_ann.columns)

    # Merge all sources
    keep_cols = ["source_table","source_id","SecuCode","text_date","text_input","importance_w"]
    df = pd.concat([df_ann[keep_cols], df_news[keep_cols], df_mkt[keep_cols]], ignore_index=True)
    df = df[df["text_input"].str.strip().str.len() > 0]
    df = df.dropna(subset=["text_date"])
    print(f"[DATA] Pending: announcements={len(df_ann):,}  news_em={len(df_news):,}  "
          f"market_news={len(df_mkt):,}  total={len(df):,}")
    return df


# ──────────────────────────────────────────────────────────────
# Sentiment model (batch inference + automatic GPU detection)
# ──────────────────────────────────────────────────────────────
def load_model(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    device_name = f"GPU cuda:{device}" if device >= 0 else "CPU"
    print(f"[MODEL] Loading sentiment model: {model_name}  |  device: {device_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,        # return probabilities for all classes
        truncation=True,
        max_length=MAX_LEN,
    )
    return clf


# ──────────────────────────────────────────────────────────────
# Chinese -> English translation (Helsinki-NLP/opus-mt-zh-en)
# ──────────────────────────────────────────────────────────────
def load_translator():
    """Load the zh->en translation model. Returns None if TRANSLATE_ENABLED=False."""
    if not TRANSLATE_ENABLED:
        print("[TRANS] Translation disabled — using raw text directly")
        return None
    from transformers import MarianMTModel, MarianTokenizer
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"[TRANS] Loading translation model: {TRANS_MODEL_NAME}  |  device: {device_name}")
    tok = MarianTokenizer.from_pretrained(TRANS_MODEL_NAME)
    mdl = MarianMTModel.from_pretrained(TRANS_MODEL_NAME)
    mdl.eval()
    if torch.cuda.is_available():
        mdl = mdl.cuda()
    print("[TRANS] Translation model ready")
    return (tok, mdl)


def translate_texts(
    translator,
    texts: List[str],
    batch_size: int = TRANS_BATCH_SIZE,
) -> List[str]:
    """
    Batch-translate Chinese texts to English.
    - Returns texts unchanged when translator is None (translation disabled).
    - Empty strings pass through as-is.
    """
    if translator is None:
        return texts

    tok, mdl = translator
    device = next(mdl.parameters()).device
    results: List[str] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out_ids = mdl.generate(**encoded, num_beams=1, max_new_tokens=128)
        decoded = tok.batch_decode(out_ids, skip_special_tokens=True)
        results.extend(decoded)

    return results


def _parse_probs(output: List[Dict]) -> Tuple[float, float, float, str, float, float]:
    """
    Parse pipeline output (top_k=None) into component probabilities.
    Returns: (prob_pos, prob_neu, prob_neg, label, score_raw, sentiment_score)
    """
    label_map: Dict[str, float] = {}
    for item in output:
        label_map[item["label"].lower()] = item["score"]

    # Try multiple label naming conventions
    prob_pos = (label_map.get("label_2") or label_map.get("positive") or
                label_map.get("pos") or 0.0)
    prob_neu = (label_map.get("label_1") or label_map.get("neutral") or
                label_map.get("neu") or 0.0)
    prob_neg = (label_map.get("label_0") or label_map.get("negative") or
                label_map.get("neg") or 0.0)

    # Normalize to guard against floating-point drift
    total = prob_pos + prob_neu + prob_neg
    if total > 0:
        prob_pos /= total; prob_neu /= total; prob_neg /= total

    # Final sentiment score = P(positive) - P(negative) in [-1, 1]
    sentiment_score = float(prob_pos - prob_neg)

    # Dominant label (highest probability class)
    best      = max(output, key=lambda x: x["score"])
    label     = best["label"]
    score_raw = best["score"]

    return prob_pos, prob_neu, prob_neg, label, score_raw, sentiment_score


def run_inference(
    clf,
    translator,
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    batch_size: int = BATCH_SIZE,
):
    """Batch translate (zh->en) then run sentiment inference, writing to sentiment_raw."""
    texts_zh  = df["text_input"].tolist()
    n         = len(texts_zh)
    n_batches = math.ceil(n / batch_size)

    insert_sql = """
    INSERT OR IGNORE INTO sentiment_raw
        (source_table, source_id, SecuCode, text_date, text_input,
         label, score_raw, prob_pos, prob_neu, prob_neg,
         sentiment_score, importance_w, model_name)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """

    rows_written = 0
    pbar = tqdm(range(n_batches), desc="Inference", unit="batch", ascii=True)

    for b in pbar:
        start = b * batch_size
        end   = min(start + batch_size, n)
        batch_texts_zh = texts_zh[start:end]
        batch_meta     = df.iloc[start:end]

        # Translate Chinese -> English
        batch_texts_en = translate_texts(translator, batch_texts_zh,
                                         batch_size=len(batch_texts_zh))

        # Run sentiment inference
        try:
            results = clf(batch_texts_en)
        except Exception as e:
            print(f"\n  [WARN] batch {b} inference failed: {e}, skipping")
            continue

        rows = []
        for i, (out, (_, row)) in enumerate(zip(results, batch_meta.iterrows())):
            prob_pos, prob_neu, prob_neg, label, score_raw, sent_score = _parse_probs(out)
            rows.append((
                row["source_table"],
                int(row["source_id"]),
                row["SecuCode"],
                row["text_date"],
                batch_texts_en[i][:512],   # store translated English text
                label,
                float(score_raw),
                float(prob_pos),
                float(prob_neu),
                float(prob_neg),
                float(sent_score),
                float(row["importance_w"]),
                MODEL_NAME,
            ))

        cur = conn.cursor()
        cur.executemany(insert_sql, rows)
        conn.commit()
        rows_written += len(rows)
        pbar.set_postfix(written=rows_written)

    print(f"\n[INFER] Done: {rows_written:,} rows written to sentiment_raw")
    return rows_written


# ──────────────────────────────────────────────────────────────
# Daily aggregation -> RL features
# ──────────────────────────────────────────────────────────────
def _weighted_mean(scores: np.ndarray, weights: np.ndarray) -> float:
    w = weights / (weights.sum() + 1e-12)
    return float(np.dot(w, scores))


def _weighted_std(scores: np.ndarray, weights: np.ndarray) -> float:
    if len(scores) <= 1:
        return 0.0
    w   = weights / (weights.sum() + 1e-12)
    mu  = np.dot(w, scores)
    var = np.dot(w, (scores - mu) ** 2)
    return float(math.sqrt(max(var, 0.0)))


def aggregate_daily(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Aggregate sentiment_raw into daily features per (SecuCode, trade_date).
    Returns a complete daily_sentiment_features DataFrame.
    """
    print("[AGG] Reading sentiment_raw ...")
    df = pd.read_sql(
        """SELECT SecuCode, text_date, sentiment_score, importance_w,
                  prob_pos, prob_neg, source_table
           FROM sentiment_raw
           ORDER BY SecuCode, text_date""",
        conn,
    )
    if df.empty:
        print("[AGG] sentiment_raw is empty, skipping aggregation")
        return pd.DataFrame()

    df["text_date"] = pd.to_datetime(df["text_date"])
    print(f"[AGG] {len(df):,} raw records — starting daily aggregation ...")

    records = []
    for (code, date), grp in tqdm(
        df.groupby(["SecuCode", "text_date"], sort=True),
        desc="Daily aggregation",
        unit="(stock,day)",
        ascii=True
    ):
        scores  = grp["sentiment_score"].values.astype(float)
        weights = grp["importance_w"].values.astype(float) if WEIGHT_BY_RANK else np.ones(len(grp))

        mean_s = _weighted_mean(scores, weights)
        vol_s  = _weighted_std(scores, weights)
        vol_uw = float(np.std(scores))          # unweighted std as supplement
        pos_r  = float((grp["prob_pos"] > 0.5).mean())
        neg_r  = float((grp["prob_neg"] > 0.5).mean())
        ann_n  = int((grp["source_table"] == "announcements").sum())
        news_n = int((grp["source_table"] == "news_em").sum())

        records.append({
            "SecuCode":       code,
            "trade_date":     date,
            "mean_sentiment": mean_s,
            "sentiment_vol":  vol_s,
            "weighted_vol":   vol_uw,
            "message_volume": len(grp),
            "pos_ratio":      pos_r,
            "neg_ratio":      neg_r,
            "ann_volume":     ann_n,
            "news_volume":    news_n,
        })

    feat = pd.DataFrame(records)
    feat = feat.sort_values(["SecuCode", "trade_date"]).reset_index(drop=True)

    # 30-day rolling z-score for abnormal_sentiment
    print("[AGG] Computing 30-day rolling abnormal sentiment z-score ...")
    EPS     = 1e-8
    MIN_STD = 0.01   # floor to prevent z-score explosion when all hist values are identical
    roll_mean_list, roll_std_list, abnormal_list = [], [], []

    for code, grp in feat.groupby("SecuCode", sort=False):
        s = grp["mean_sentiment"].values.astype(float)
        n = len(s)
        r_mean = np.full(n, np.nan)
        r_std  = np.full(n, np.nan)
        r_abno = np.full(n, np.nan)

        for i in range(n):
            window_start = max(0, i - ROLLING_WINDOW)
            hist = s[window_start:i]   # exclude current day
            if len(hist) >= 2:
                mu        = hist.mean()
                sig       = hist.std()
                r_mean[i] = mu
                r_std[i]  = sig
                r_abno[i] = float(np.clip((s[i] - mu) / (max(sig, MIN_STD) + EPS), -10.0, 10.0))
            elif len(hist) == 1:
                r_mean[i] = hist[0]
                r_std[i]  = 0.0
                r_abno[i] = 0.0

        roll_mean_list.extend(r_mean.tolist())
        roll_std_list.extend(r_std.tolist())
        abnormal_list.extend(r_abno.tolist())

    feat["roll_mean_30d"]      = roll_mean_list
    feat["roll_std_30d"]       = roll_std_list
    feat["abnormal_sentiment"] = abnormal_list
    feat["trade_date"]         = feat["trade_date"].dt.strftime("%Y-%m-%d")

    return feat


def write_daily_features(conn: sqlite3.Connection, feat: pd.DataFrame):
    if feat.empty:
        return 0
    # Full recompute: clear old data to ensure consistency
    conn.execute("DELETE FROM daily_sentiment_features")
    conn.commit()

    cols = [
        "SecuCode","trade_date","mean_sentiment","sentiment_vol",
        "message_volume","abnormal_sentiment","roll_mean_30d","roll_std_30d",
        "weighted_vol","pos_ratio","neg_ratio","ann_volume","news_volume",
    ]
    feat_out = feat[cols].where(pd.notnull(feat[cols]), None)
    feat_out.to_sql("daily_sentiment_features", conn, if_exists="append", index=False)
    conn.commit()
    return len(feat_out)


# ──────────────────────────────────────────────────────────────
# Sample output (example RL query)
# ──────────────────────────────────────────────────────────────
def print_rl_sample(conn: sqlite3.Connection):
    sample = conn.execute("""
        SELECT SecuCode, trade_date,
               ROUND(mean_sentiment,4)      AS mean_sent,
               ROUND(sentiment_vol,4)       AS sent_vol,
               message_volume               AS vol,
               ROUND(abnormal_sentiment,4)  AS abnormal
        FROM daily_sentiment_features
        WHERE mean_sentiment IS NOT NULL
          AND message_volume >= 1
        ORDER BY RANDOM()
        LIMIT 8
    """).fetchall()

    print("\n" + "=" * 72)
    print("RL Input Feature Sample (daily_sentiment_features)")
    print(f"{'SecuCode':>10} {'date':>12} {'mean_sent':>10} {'sent_vol':>9} "
          f"{'volume':>7} {'abnormal':>10}")
    print("-" * 72)
    for r in sample:
        print(f"{r[0]:>10} {r[1]:>12} {str(r[2]):>10} {str(r[3]):>9} "
              f"{r[4]:>7} {str(r[5]):>10}")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────
def print_summary(conn: sqlite3.Connection):
    n_raw  = conn.execute("SELECT COUNT(*) FROM sentiment_raw").fetchone()[0]
    n_feat = conn.execute("SELECT COUNT(*) FROM daily_sentiment_features").fetchone()[0]
    stocks = conn.execute("SELECT COUNT(DISTINCT SecuCode) FROM daily_sentiment_features").fetchone()[0]
    dates  = conn.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily_sentiment_features").fetchone()
    nulls  = conn.execute("SELECT COUNT(*) FROM daily_sentiment_features WHERE mean_sentiment IS NULL").fetchone()[0]

    print("\n" + "=" * 72)
    print("Collection & Inference Complete -- Final Statistics")
    print(f"  sentiment_raw              : {n_raw:>10,} rows")
    print(f"  daily_sentiment_features   : {n_feat:>10,} rows  ({stocks} stocks)")
    print(f"  Date coverage              : {dates[0]} ~ {dates[1]}")
    print(f"  NULL rows (no texts)       : {nulls:>10,}")

    dist = conn.execute("""
        SELECT
            ROUND(AVG(mean_sentiment),4) AS avg_sent,
            ROUND(AVG(sentiment_vol),4)  AS avg_vol,
            ROUND(AVG(message_volume),1) AS avg_msg,
            ROUND(AVG(ABS(abnormal_sentiment)),4) AS avg_abs_abnormal
        FROM daily_sentiment_features
        WHERE mean_sentiment IS NOT NULL
    """).fetchone()
    if dist:
        print(f"\n  Cross-stock avg sentiment  : {dist[0]}")
        print(f"  Cross-stock avg volatility : {dist[1]}")
        print(f"  Avg daily message volume   : {dist[2]}")
        print(f"  Avg |abnormal sentiment|   : {dist[3]}")

    print(f"\n  Database path              : {DB_PATH}")
    print("\n  RL usage example:")
    print("    conn = sqlite3.connect('sentiment_data.db')")
    print("    df = pd.read_sql(\"\"\"")
    print("        SELECT SecuCode, trade_date,")
    print("               mean_sentiment, sentiment_vol,")
    print("               message_volume, abnormal_sentiment")
    print("        FROM daily_sentiment_features")
    print("        WHERE SecuCode='000858'")
    print("        ORDER BY trade_date\"\"\", conn)")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("NolBERT Sentiment Inference + Daily RL Feature Aggregation")
    print(f"  Sentiment model : {MODEL_NAME}")
    print(f"  Translation     : {TRANS_MODEL_NAME if TRANSLATE_ENABLED else 'disabled (raw text)'}")
    print(f"  Database        : {DB_PATH}")
    print(f"  Batch size      : {BATCH_SIZE}  |  Max length: {MAX_LEN}  |  Rolling window: {ROLLING_WINDOW}d")
    print("=" * 72)

    if not DB_PATH.exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        print("  Please run scripts/download_sentiment_data.py first")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # Check model consistency — clear old results if model has changed
    existing_models = [r[0] for r in conn.execute(
        "SELECT DISTINCT model_name FROM sentiment_raw WHERE model_name IS NOT NULL"
    ).fetchall()]
    if existing_models and MODEL_NAME not in existing_models:
        old_models = ", ".join(existing_models)
        print(f"[WARN] Existing results from different model: {old_models}")
        print(f"[WARN] Current model: {MODEL_NAME}")
        print(f"[WARN] Clearing old inference results to ensure model consistency ...")
        conn.execute("DELETE FROM sentiment_raw")
        conn.execute("DELETE FROM daily_sentiment_features")
        conn.commit()
        print(f"[WARN] Cleared — will re-infer all texts")

    # Step 1: Load pending texts
    df_pending = load_pending(conn)
    if df_pending.empty:
        print("[INFO] No pending texts — jumping to aggregation")
    else:
        n_pending   = len(df_pending)
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        # Translate ~0.05s + infer ~0.15s per item on CPU; GPU ~0.02s total
        secs_per_item = 0.02 if torch.cuda.is_available() else 0.20
        eta_min = n_pending * secs_per_item / 60
        print(f"[INFO] {n_pending:,} texts pending  |  device: {device_name}"
              f"  |  estimated time (with translation): {eta_min:.0f} min")
        if not torch.cuda.is_available() and n_pending > 5000:
            print("[HINT] Consider running in background; or pre-filter by importance_rank<=5")

        # Step 2: Load translation model
        translator = load_translator()

        # Step 3: Load sentiment model
        clf = load_model(MODEL_NAME)

        # Step 4: Translate + infer
        run_inference(clf, translator, df_pending, conn, batch_size=BATCH_SIZE)

        # Free memory
        del clf, translator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 5: Daily aggregation -> RL features
    feat = aggregate_daily(conn)
    if not feat.empty:
        n = write_daily_features(conn, feat)
        print(f"[AGG] Wrote {n:,} rows to daily_sentiment_features")

    # Step 6: Print sample and summary
    print_rl_sample(conn)
    print_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
