#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BY_STOCK_DIR = ROOT / "data" / "raw" / "meta" / "by_stock"
DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "meta"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split per-stock CNINFO meta CSVs into monthly meta CSVs.")
    p.add_argument("--by_stock_dir", type=Path, default=DEFAULT_BY_STOCK_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--start_month", default="2018-01")
    p.add_argument("--end_month", default="2025-12")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def month_range(start_month: str, end_month: str) -> list[str]:
    start = datetime.strptime(start_month, "%Y-%m")
    end = datetime.strptime(end_month, "%Y-%m")
    if start > end:
        raise ValueError(f"start_month > end_month: {start_month} > {end_month}")

    out: list[str] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def main() -> int:
    args = parse_args()

    if not args.by_stock_dir.exists():
        raise FileNotFoundError(f"by_stock_dir not found: {args.by_stock_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stock_files = sorted(args.by_stock_dir.glob("announcements_*.csv"))
    if not stock_files:
        raise FileNotFoundError(f"No announcements_*.csv found in {args.by_stock_dir}")

    frames: list[pd.DataFrame] = []
    for f in stock_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            frames.append(df)
        except Exception:
            continue

    if not frames:
        print("[split] no rows found in by-stock files")
        return 0

    all_df = pd.concat(frames, ignore_index=True)

    required = ["SecuCode", "publish_ts", "title", "pdf_url", "source", "column", "orgId"]
    for c in required:
        if c not in all_df.columns:
            all_df[c] = ""

    dt = pd.to_datetime(all_df["publish_ts"], utc=True, errors="coerce")
    all_df = all_df.loc[~dt.isna()].copy()
    all_df["month_tag"] = dt.dt.strftime("%Y-%m")

    months = month_range(args.start_month, args.end_month)
    months_written = 0
    months_skipped = 0
    months_zero_rows = 0
    total_rows_processed = 0

    for month in months:
        out_file = args.out_dir / f"announcements_meta_{month}.csv"

        if out_file.exists() and not args.overwrite:
            print(f"[split] month={month} SKIP existing file={out_file}")
            months_skipped += 1
            continue

        sub = all_df.loc[all_df["month_tag"] == month, required].copy()
        if sub.empty:
            months_zero_rows += 1
            continue

        sub = sub.rename(columns={"SecuCode": "ticker"})
        sub["ticker"] = sub["ticker"].astype(str).str.zfill(6)
        sub = sub.sort_values(["ticker", "publish_ts", "title"], kind="stable").reset_index(drop=True)

        sub.to_csv(out_file, index=False, encoding="utf-8")
        months_written += 1
        total_rows_processed += int(len(sub))

        print(f"[split] month={month} rows={len(sub)} file={out_file}")

    print("[split] summary")
    print(f"months_written={months_written}")
    print(f"months_skipped={months_skipped}")
    print(f"months_with_0_rows_not_written={months_zero_rows}")
    print(f"total_rows_processed={total_rows_processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
