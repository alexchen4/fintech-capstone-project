#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.secu import normalize_secu_series
from common.universe import validate_universe
from sentiment.align_time import build_universe_bar_index, load_universe_set

DEFAULT_DATASET = ROOT / "data" / "processed" / "sentiment" / "sentiment_dataset_2025-12.parquet"
DEFAULT_PRICE = ROOT / "data" / "qfq_15min_all.csv"
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "universe.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "sentiment"


def _infer_out_path(ds_path: Path) -> Path:
    m = re.search(r"(\d{4}-\d{2})", ds_path.name)
    tag = m.group(1) if m else "unknown"
    return DEFAULT_OUT_DIR / f"sentiment_features_15m_{tag}.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate announcement-level sentiment to 15m feature panel.")
    p.add_argument("--dataset_parquet", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--price_csv", type=Path, default=DEFAULT_PRICE)
    p.add_argument("--universe_csv", type=Path, default=DEFAULT_UNIVERSE)
    p.add_argument("--out_parquet", type=Path, default=None)
    p.add_argument("--chunksize", type=int, default=500_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_parquet.exists():
        raise FileNotFoundError(f"dataset parquet not found: {args.dataset_parquet}")
    if not args.price_csv.exists():
        raise FileNotFoundError(f"price_csv not found: {args.price_csv}")
    if not args.universe_csv.exists():
        raise FileNotFoundError(f"universe_csv not found: {args.universe_csv}")

    out_parquet = args.out_parquet or _infer_out_path(args.dataset_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    ds = pd.read_parquet(args.dataset_parquet)
    required = {"ann_id", "SecuCode", "t_event_bar"}
    miss = sorted(required - set(ds.columns))
    if miss:
        raise ValueError(f"dataset missing columns: {miss}")

    ds = ds.copy()
    ds["SecuCode"] = normalize_secu_series(ds["SecuCode"])
    ds["bar_ts"] = pd.to_datetime(ds["t_event_bar"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai")

    # Placeholder sentiment score until model outputs are available.
    if "sent_score" not in ds.columns:
        ds["sent_score"] = 0.0
    ds["sent_score"] = pd.to_numeric(ds["sent_score"], errors="coerce")

    universe_set = load_universe_set(args.universe_csv)
    validate_universe(ds, universe_set, col="SecuCode")

    bars = build_universe_bar_index(args.price_csv, universe_set, chunksize=args.chunksize)
    valid_pairs = set(zip(bars["SecuCode"].tolist(), pd.to_datetime(bars["bar_ts_local"], errors="coerce").tolist()))

    feats = (
        ds.dropna(subset=["bar_ts"])
        .groupby(["SecuCode", "bar_ts"], dropna=False)
        .agg(
            sent_count=("ann_id", "count"),
            sent_sum=("sent_score", "sum"),
            sent_mean=("sent_score", "mean"),
        )
        .reset_index()
    )

    # Validate each (SecuCode, bar_ts) exists in source-of-truth bars.
    check_pairs = list(zip(feats["SecuCode"].tolist(), pd.to_datetime(feats["bar_ts"], errors="coerce").tolist()))
    missing_pairs = [p for p in check_pairs if p not in valid_pairs]
    if missing_pairs:
        preview = ", ".join([f"({a},{b})" for a, b in missing_pairs[:10]])
        raise ValueError(f"Found {len(missing_pairs)} feature rows with non-existent bar_ts in price data: {preview}")

    feats = feats.sort_values(["SecuCode", "bar_ts"], kind="stable").reset_index(drop=True)
    feats.to_parquet(out_parquet, index=False)

    print("[build_sentiment_features_15m] summary")
    print(f"dataset_parquet={args.dataset_parquet}")
    print(f"out_parquet={out_parquet}")
    print(f"rows={len(feats)}")
    print(f"unique_SecuCode={int(feats['SecuCode'].nunique()) if len(feats) else 0}")
    print(f"bar_ts_min={None if len(feats)==0 else str(feats['bar_ts'].min())}")
    print(f"bar_ts_max={None if len(feats)==0 else str(feats['bar_ts'].max())}")


if __name__ == "__main__":
    main()
