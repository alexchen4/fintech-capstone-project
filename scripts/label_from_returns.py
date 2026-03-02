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
from sentiment.align_time import load_universe_set

DEFAULT_DATASET = ROOT / "data" / "processed" / "sentiment" / "sentiment_dataset_2025-12.parquet"
DEFAULT_PRICE = ROOT / "data" / "qfq_15min_all.csv"
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "universe.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "sentiment"


def _infer_out_path(ds_path: Path) -> Path:
    m = re.search(r"(\d{4}-\d{2})", ds_path.name)
    tag = m.group(1) if m else "unknown"
    return DEFAULT_OUT_DIR / f"sentiment_labeled_{tag}.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create weak sentiment labels from future 15m returns.")
    p.add_argument("--dataset_parquet", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--price_csv", type=Path, default=DEFAULT_PRICE)
    p.add_argument("--universe_csv", type=Path, default=DEFAULT_UNIVERSE)
    p.add_argument("--out_parquet", type=Path, default=None)
    p.add_argument("--h", type=int, default=8, help="Forward bars horizon")
    p.add_argument("--tau", type=float, default=0.002, help="Label threshold")
    p.add_argument("--chunksize", type=int, default=500_000)
    return p.parse_args()


def _load_close_series(price_csv: Path, universe_set: set[str], chunksize: int) -> pd.DataFrame:
    close_parts: list[pd.DataFrame] = []
    for ch in pd.read_csv(
        price_csv,
        usecols=["SecuCode", "TradingDay", "TimeStart", "ClosePrice"],
        dtype={"SecuCode": str, "TradingDay": str, "TimeStart": str, "ClosePrice": float},
        chunksize=chunksize,
    ):
        c = ch.copy()
        c["SecuCode"] = normalize_secu_series(c["SecuCode"])
        c = c[c["SecuCode"].isin(universe_set)]
        if c.empty:
            continue
        c["bar_ts_local"] = pd.to_datetime(
            c["TradingDay"].astype(str).str.strip() + " " + c["TimeStart"].astype(str).str.strip().str.zfill(4),
            format="%Y%m%d %H%M",
            errors="coerce",
        ).dt.tz_localize("Asia/Shanghai")
        c = c.rename(columns={"ClosePrice": "close"})
        c = c.dropna(subset=["bar_ts_local", "close"])
        close_parts.append(c[["SecuCode", "bar_ts_local", "close"]])

    if not close_parts:
        return pd.DataFrame(columns=["SecuCode", "bar_ts_local", "close"])

    out = pd.concat(close_parts, ignore_index=True)
    out = out.drop_duplicates(["SecuCode", "bar_ts_local"], keep="last")
    out = out.sort_values(["SecuCode", "bar_ts_local"], kind="stable").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    if not args.dataset_parquet.exists():
        raise FileNotFoundError(f"dataset parquet not found: {args.dataset_parquet}")
    if not args.price_csv.exists():
        raise FileNotFoundError(f"price_csv not found: {args.price_csv}")

    out_parquet = args.out_parquet or _infer_out_path(args.dataset_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    ds = pd.read_parquet(args.dataset_parquet)
    required = {"ann_id", "SecuCode", "t_event_bar"}
    miss = sorted(required - set(ds.columns))
    if miss:
        raise ValueError(f"dataset missing columns: {miss}")

    ds["SecuCode"] = normalize_secu_series(ds["SecuCode"])
    ds["t_event_bar"] = pd.to_datetime(ds["t_event_bar"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai")

    universe_set = load_universe_set(args.universe_csv)
    bars = _load_close_series(args.price_csv, universe_set=universe_set, chunksize=args.chunksize)
    if bars.empty:
        raise ValueError("No bars loaded for universe.")

    bars = bars.sort_values(["SecuCode", "bar_ts_local"], kind="stable").reset_index(drop=True)
    bars[f"close_fwd_{args.h}"] = bars.groupby("SecuCode")["close"].shift(-int(args.h))
    bars[f"ret_fwd_{args.h}"] = bars[f"close_fwd_{args.h}"] / bars["close"] - 1.0

    join_cols = ["SecuCode", "bar_ts_local", f"ret_fwd_{args.h}"]
    merged = ds.merge(
        bars[join_cols],
        left_on=["SecuCode", "t_event_bar"],
        right_on=["SecuCode", "bar_ts_local"],
        how="left",
    ).drop(columns=["bar_ts_local"])

    ret_col = f"ret_fwd_{args.h}"
    merged["y"] = pd.NA
    merged.loc[merged[ret_col] > args.tau, "y"] = 1
    merged.loc[merged[ret_col] < -args.tau, "y"] = -1
    merged.loc[(merged[ret_col] >= -args.tau) & (merged[ret_col] <= args.tau), "y"] = 0

    merged.to_parquet(out_parquet, index=False)

    total = len(merged)
    covered = int(merged[ret_col].notna().sum())
    y_counts = merged["y"].value_counts(dropna=False).to_dict()

    print("[label_from_returns] summary")
    print(f"dataset_parquet={args.dataset_parquet}")
    print(f"out_parquet={out_parquet}")
    print(f"h={args.h}")
    print(f"tau={args.tau}")
    print(f"rows={total}")
    print(f"label_coverage={covered}/{total} ({round(100.0*covered/total,2) if total else 0.0}%)")
    print(f"class_balance={y_counts}")


if __name__ == "__main__":
    main()
