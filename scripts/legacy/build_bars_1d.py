#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "df_clean.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "bars_1d.parquet"
DEFAULT_META = ROOT / "data" / "processed" / "bars_1d_meta.json"


def normalize_asset_key(x: object) -> str:
    s = str(x or "").strip().upper()
    if not s:
        return ""
    m = re.match(r"^(SHSE|SSE|XSHG)[\.\-_]?(\d{1,6})$", s)
    if m:
        return f"SHSE.{m.group(2).zfill(6)}"
    m = re.match(r"^(SZSE|SZ|XSHE)[\.\-_]?(\d{1,6})$", s)
    if m:
        return f"SZSE.{m.group(2).zfill(6)}"
    d = re.search(r"(\d+)", s)
    if not d:
        return ""
    code = d.group(1).zfill(6)
    exch = "SHSE" if code.startswith(("5", "6", "9")) else "SZSE"
    return f"{exch}.{code}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build daily bars backbone from df_clean.csv")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output_parquet", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--meta_json", type=Path, default=DEFAULT_META)
    p.add_argument("--date_min", type=str, default="2018-01-01")
    p.add_argument("--date_max", type=str, default="2025-12-31")
    p.add_argument(
        "--keep_cols",
        type=str,
        default="open,high,low,close,volume,amount",
        help="Comma-separated additional columns to retain (besides asset_key/date).",
    )
    p.add_argument("--chunksize", type=int, default=200000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"missing input csv: {args.input_csv}")

    start = pd.to_datetime(args.date_min, errors="raise")
    end = pd.to_datetime(args.date_max, errors="raise")

    header = pd.read_csv(args.input_csv, nrows=0)
    cols = list(header.columns)
    required = {"ticker", "frequency", "eob"}
    missing = required.difference(cols)
    if missing:
        raise ValueError(f"df_clean missing required columns: {sorted(missing)}")

    keep_extra = [c.strip() for c in args.keep_cols.split(",") if c.strip()]
    usecols = [c for c in cols if c in set(["ticker", "frequency", "eob", "bob"] + keep_extra)]

    parts = []
    for chunk in pd.read_csv(args.input_csv, usecols=usecols, chunksize=args.chunksize):
        freq = chunk["frequency"].astype(str).str.lower()
        dt = pd.to_datetime(chunk["eob"], errors="coerce")
        asset_key = chunk["ticker"].map(normalize_asset_key)
        mask = freq.eq("1d") & dt.between(start, end, inclusive="both") & asset_key.ne("")
        if not mask.any():
            continue

        sub = chunk.loc[mask].copy()
        sub["asset_key"] = asset_key.loc[mask].astype(str)
        sub["date"] = dt.loc[mask].dt.strftime("%Y-%m-%d")

        ordered = ["asset_key", "date"] + [c for c in keep_extra if c in sub.columns]
        parts.append(sub[ordered])

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["asset_key", "date"] + keep_extra)
    out = out.dropna(subset=["asset_key", "date"]).sort_values(["asset_key", "date"]).reset_index(drop=True)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_parquet, index=False)

    meta = {
        "input_csv": str(args.input_csv),
        "output_parquet": str(args.output_parquet),
        "rows": int(len(out)),
        "unique_tickers": int(out["asset_key"].nunique()) if len(out) else 0,
        "date_min": str(out["date"].min()) if len(out) else "",
        "date_max": str(out["date"].max()) if len(out) else "",
        "columns": out.columns.tolist(),
    }
    with args.meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"wrote: {args.output_parquet}")
    print(f"wrote: {args.meta_json}")
    print(f"rows={meta['rows']} unique_tickers={meta['unique_tickers']} date_range=[{meta['date_min']}, {meta['date_max']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
