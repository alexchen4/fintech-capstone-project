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
from sentiment.align_time import (
    align_events_to_next_bar,
    build_universe_bar_index,
    load_universe_set,
    validate_events_universe,
)
from sentiment.truncation import truncate_text

DEFAULT_ANN = ROOT / "data" / "processed" / "cninfo_text" / "ann_text_2025-12.parquet"
DEFAULT_PRICE = ROOT / "data" / "qfq_15min_all.csv"
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "universe.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "sentiment"


def _infer_out_path(ann_path: Path) -> Path:
    m = re.search(r"(\d{4}-\d{2})", ann_path.name)
    tag = m.group(1) if m else "unknown"
    return DEFAULT_OUT_DIR / f"sentiment_dataset_{tag}.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sentiment dataset with deterministic 15m t_event_bar alignment.")
    parser.add_argument("--ann_text_parquet", type=Path, default=DEFAULT_ANN)
    parser.add_argument("--price_csv", type=Path, default=DEFAULT_PRICE)
    parser.add_argument("--universe_csv", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--out_parquet", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--sample_n", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_chars_total", type=int, default=4000)
    parser.add_argument("--max_chars_body", type=int, default=3500)
    return parser.parse_args()


def _assign_time_split(publish_ts: pd.Series, train_ratio: float, val_ratio: float) -> pd.Series:
    if train_ratio <= 0 or val_ratio < 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("Invalid split ratios. Require train_ratio>0, val_ratio>=0, train_ratio+val_ratio<1.")

    n = len(publish_ts)
    if n == 0:
        return pd.Series([], dtype="object")

    order = publish_ts.sort_values(kind="stable").index.tolist()
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    split = pd.Series(index=publish_ts.index, dtype="object")
    split.loc[order[:train_end]] = "train"
    split.loc[order[train_end:val_end]] = "val"
    split.loc[order[val_end:]] = "test"
    return split


def _run_validations(df: pd.DataFrame, universe_set: set[str]) -> None:
    bad_len = df["SecuCode"].astype(str).str.len() != 6
    if bool(bad_len.any()):
        raise ValueError(f"SecuCode length validation failed for {int(bad_len.sum())} rows.")

    out_of_uni = sorted(set(df["SecuCode"].tolist()) - universe_set)
    if out_of_uni:
        raise ValueError(f"Found {len(out_of_uni)} codes outside universe: {out_of_uni[:20]}")

    dupes = int(df["ann_id"].astype(str).duplicated().sum())
    if dupes > 0:
        raise ValueError(f"Duplicate ann_id found: {dupes}")


def main() -> None:
    args = parse_args()

    if not args.ann_text_parquet.exists():
        raise FileNotFoundError(f"ann_text parquet not found: {args.ann_text_parquet}")
    if not args.price_csv.exists():
        raise FileNotFoundError(f"price_csv not found: {args.price_csv}")
    if not args.universe_csv.exists():
        raise FileNotFoundError(f"universe_csv not found: {args.universe_csv}")

    out_parquet = args.out_parquet or _infer_out_path(args.ann_text_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    ann = pd.read_parquet(args.ann_text_parquet)
    required = {"ann_id", "SecuCode", "publish_dt_utc", "clean_title", "clean_body", "text_len", "parse_status"}
    missing = sorted(required - set(ann.columns))
    if missing:
        raise ValueError(f"ann_text parquet missing required columns: {missing}")

    ds = ann.copy()
    ds["SecuCode"] = normalize_secu_series(ds["SecuCode"])

    universe_set = load_universe_set(args.universe_csv)
    validate_events_universe(ds, universe_set)

    bars = build_universe_bar_index(args.price_csv, universe_set=universe_set, chunksize=args.chunksize)
    aligned = align_events_to_next_bar(ds, bars)

    # Phase-6 model table fields.
    aligned["publish_dt_utc"] = pd.to_datetime(aligned["publish_dt_utc"], errors="coerce", utc=True)
    trunc = [
        truncate_text(
            t,
            b,
            max_chars_total=int(args.max_chars_total),
            max_chars_body=int(args.max_chars_body),
        )
        for t, b in zip(aligned["clean_title"], aligned["clean_body"])
    ]
    aligned["text"] = [x.text for x in trunc]
    aligned["text_len_chars"] = [int(x.text_len_chars) for x in trunc]
    aligned["was_truncated"] = [bool(x.was_truncated) for x in trunc]
    # Backward-compatible alias for prior checks/scripts.
    aligned["text_len"] = aligned["text_len_chars"]
    aligned["split"] = _assign_time_split(aligned["publish_dt_utc"], args.train_ratio, args.val_ratio)

    out = aligned[
        [
            "ann_id",
            "SecuCode",
            "publish_dt_utc",
            "t_event_bar",
            "text",
            "text_len_chars",
            "was_truncated",
            "text_len",
            "parse_status",
            "split",
        ]
    ].copy()

    _run_validations(out, universe_set)
    out = out.sort_values(["publish_dt_utc", "ann_id"], kind="stable").reset_index(drop=True)
    out.to_parquet(out_parquet, index=False)

    sample = out.head(int(args.sample_n)).copy()
    sample["text_preview"] = sample["text"].astype(str).str.slice(0, 120)
    sample = sample[
        [
            "ann_id",
            "SecuCode",
            "publish_dt_utc",
            "t_event_bar",
            "text_len_chars",
            "was_truncated",
            "text_len",
            "parse_status",
            "split",
            "text_preview",
        ]
    ]
    miss = out.isna().sum().to_dict()

    print("[build_sentiment_dataset] summary")
    print(f"ann_text_parquet={args.ann_text_parquet}")
    print(f"price_csv={args.price_csv}")
    print(f"universe_csv={args.universe_csv}")
    print(f"out_parquet={out_parquet}")
    print(f"rows={len(out)}")
    print(f"unique_SecuCode={int(out['SecuCode'].nunique())}")
    print(f"rows_t_event_bar_null={int(out['t_event_bar'].isna().sum())}")
    print(f"parse_status_counts={out['parse_status'].value_counts(dropna=False).to_dict()}")
    print(f"split_counts={out['split'].value_counts(dropna=False).to_dict()}")
    print(f"missingness={miss}")
    print("sample_head=")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
