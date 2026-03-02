#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data" / "processed" / "panel_with_sentiment_1d.parquet"
DEFAULT_LABELS = ROOT / "data" / "processed" / "labels_1d.parquet"
DEFAULT_SIGNALS_DIR = ROOT / "data" / "processed" / "signals"
DEFAULT_OUT = ROOT / "data" / "processed" / "training_table_1d.parquet"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge panel + labels + model signals into one daily training table.")
    p.add_argument("--panel_path", type=Path, default=DEFAULT_PANEL)
    p.add_argument("--labels_path", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--signals_dir", type=Path, default=DEFAULT_SIGNALS_DIR)
    p.add_argument("--output_path", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.panel_path.exists():
        raise FileNotFoundError(f"missing panel file: {args.panel_path}")
    if not args.labels_path.exists():
        raise FileNotFoundError(f"missing labels file: {args.labels_path}")

    base = _read_table(args.panel_path)
    labels = _read_table(args.labels_path)
    required = {"asset_key", "date"}
    for name, df in [("panel", base), ("labels", labels)]:
        miss = required.difference(df.columns)
        if miss:
            raise ValueError(f"{name} missing required keys: {sorted(miss)}")

    out = base.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.merge(labels, on=["asset_key", "date"], how="left")

    signal_files = []
    if args.signals_dir.exists():
        signal_files = sorted([p for p in args.signals_dir.iterdir() if p.suffix.lower() in {".csv", ".parquet"}])

    merged_signals = 0
    for path in signal_files:
        sig = _read_table(path)
        miss = required.difference(sig.columns)
        if miss:
            continue
        value_cols = [c for c in sig.columns if c.startswith("signal_")]
        if not value_cols:
            continue
        keep = ["asset_key", "date"] + value_cols
        sig = sig[keep].copy()
        sig["date"] = pd.to_datetime(sig["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out = out.merge(sig, on=["asset_key", "date"], how="left")
        merged_signals += 1

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_path, index=False)
    print(f"wrote: {args.output_path}")
    print(f"rows={len(out)} signal_files_merged={merged_signals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
