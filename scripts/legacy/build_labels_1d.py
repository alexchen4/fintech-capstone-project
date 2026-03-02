#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BARS = ROOT / "data" / "processed" / "bars_1d.parquet"
DEFAULT_LABELS = ROOT / "data" / "processed" / "labels_1d.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build forward return labels from daily bars backbone.")
    p.add_argument("--bars_path", type=Path, default=DEFAULT_BARS)
    p.add_argument("--output_path", type=Path, default=DEFAULT_LABELS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.bars_path.exists():
        raise FileNotFoundError(f"missing bars file: {args.bars_path}")

    df = pd.read_parquet(args.bars_path)
    required = {"asset_key", "date", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"bars file missing required columns: {sorted(missing)}")

    out = df[["asset_key", "date", "close"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["asset_key", "date", "close"]).sort_values(["asset_key", "date"]).reset_index(drop=True)

    out["ret_1d_fwd"] = out.groupby("asset_key")["close"].shift(-1) / out["close"] - 1.0
    out["ret_5d_fwd"] = out.groupby("asset_key")["close"].shift(-5) / out["close"] - 1.0
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out[["asset_key", "date", "ret_1d_fwd", "ret_5d_fwd"]]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_path, index=False)

    nan_tail_1d = int(out["ret_1d_fwd"].isna().sum())
    nan_tail_5d = int(out["ret_5d_fwd"].isna().sum())
    print(f"wrote: {args.output_path}")
    print(f"rows={len(out)} nan_ret_1d={nan_tail_1d} nan_ret_5d={nan_tail_5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
