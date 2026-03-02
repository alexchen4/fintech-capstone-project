#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "processed" / "sentiment" / "sentiment_dataset_2025-12.parquet"
DEFAULT_PRICE = ROOT / "data" / "qfq_15min_all.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit t_event_bar alignment quality and price join coverage.")
    p.add_argument("--dataset_parquet", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--price_csv", type=Path, default=DEFAULT_PRICE)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--top_k_worst", type=int, default=10)
    return p.parse_args()


def _required_key_df(ds: pd.DataFrame) -> pd.DataFrame:
    out = ds[["ann_id", "SecuCode", "publish_dt_utc", "t_event_bar"]].copy()
    out["SecuCode"] = out["SecuCode"].astype(str).str.strip().str.zfill(6)
    out["publish_dt_utc"] = pd.to_datetime(out["publish_dt_utc"], errors="coerce", utc=True)
    out["t_event_bar"] = pd.to_datetime(out["t_event_bar"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai")

    out["TradingDay"] = out["t_event_bar"].dt.strftime("%Y%m%d")
    out["TimeStart"] = out["t_event_bar"].dt.strftime("%H%M")
    return out


def _delay_minutes(ds: pd.DataFrame) -> pd.Series:
    pub_local = pd.to_datetime(ds["publish_dt_utc"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai")
    bar_local = pd.to_datetime(ds["t_event_bar"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai")
    delay = (bar_local - pub_local).dt.total_seconds() / 60.0
    return delay


def _build_required_key_set(key_df: pd.DataFrame) -> set[tuple[str, str, str]]:
    sub = key_df.dropna(subset=["TradingDay", "TimeStart"]).copy()
    return set(zip(sub["SecuCode"].tolist(), sub["TradingDay"].tolist(), sub["TimeStart"].tolist()))


def _scan_price_for_keys(price_csv: Path, required_keys: set[tuple[str, str, str]], chunksize: int) -> set[tuple[str, str, str]]:
    matched: set[tuple[str, str, str]] = set()
    if not required_keys:
        return matched

    for ch in pd.read_csv(
        price_csv,
        usecols=["SecuCode", "TradingDay", "TimeStart"],
        dtype={"SecuCode": str, "TradingDay": str, "TimeStart": str},
        chunksize=chunksize,
    ):
        c = ch.copy()
        c["SecuCode"] = c["SecuCode"].fillna("").astype(str).str.strip().str.zfill(6)
        c["TradingDay"] = c["TradingDay"].fillna("").astype(str).str.strip()
        c["TimeStart"] = c["TimeStart"].fillna("").astype(str).str.strip().str.zfill(4)

        for k in zip(c["SecuCode"].tolist(), c["TradingDay"].tolist(), c["TimeStart"].tolist()):
            if k in required_keys:
                matched.add(k)

        if len(matched) == len(required_keys):
            break

    return matched


def main() -> None:
    args = parse_args()
    if not args.dataset_parquet.exists():
        raise FileNotFoundError(f"dataset parquet not found: {args.dataset_parquet}")
    if not args.price_csv.exists():
        raise FileNotFoundError(f"price csv not found: {args.price_csv}")

    ds = pd.read_parquet(args.dataset_parquet)
    req = {"ann_id", "SecuCode", "publish_dt_utc", "t_event_bar"}
    missing = sorted(req - set(ds.columns))
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")

    keys_df = _required_key_df(ds)

    total = len(keys_df)
    non_null_bar = int(keys_df["t_event_bar"].notna().sum())
    non_null_pct = round(100.0 * non_null_bar / total, 2) if total else 0.0

    delay_min = _delay_minutes(keys_df)
    negative_delay = int((delay_min < 0).sum())
    q = delay_min.dropna().quantile([0.5, 0.9, 0.95, 0.99]).to_dict() if non_null_bar else {0.5: None, 0.9: None, 0.95: None, 0.99: None}

    required_keys = _build_required_key_set(keys_df)
    matched_keys = _scan_price_for_keys(args.price_csv, required_keys, args.chunksize)

    matched = len(matched_keys)
    required = len(required_keys)
    join_cov = round(100.0 * matched / required, 2) if required else 0.0

    keys_df["join_key"] = list(zip(keys_df["SecuCode"], keys_df["TradingDay"], keys_df["TimeStart"]))
    keys_df["join_hit"] = keys_df["join_key"].isin(matched_keys)

    by_code = (
        keys_df.groupby("SecuCode", dropna=False)
        .agg(rows=("ann_id", "count"), hit_rows=("join_hit", "sum"))
        .reset_index()
    )
    by_code["join_cov_pct"] = (100.0 * by_code["hit_rows"] / by_code["rows"]).round(2)
    worst = by_code.sort_values(["join_cov_pct", "rows"], ascending=[True, False]).head(int(args.top_k_worst))

    print("[audit_alignment_and_price_join] summary")
    print(f"dataset_parquet={args.dataset_parquet}")
    print(f"price_csv={args.price_csv}")
    print(f"rows={total}")
    print(f"t_event_bar_non_null={non_null_bar}/{total} ({non_null_pct}%)")
    print(f"delay_min_p50={None if q.get(0.5) is None else round(float(q[0.5]), 2)}")
    print(f"delay_min_p90={None if q.get(0.9) is None else round(float(q[0.9]), 2)}")
    print(f"delay_min_p95={None if q.get(0.95) is None else round(float(q[0.95]), 2)}")
    print(f"delay_min_p99={None if q.get(0.99) is None else round(float(q[0.99]), 2)}")
    print(f"negative_delay_count={negative_delay}")
    print(f"join_coverage={matched}/{required} ({join_cov}%)")
    print("worst_by_secuCode=")
    print(worst[["SecuCode", "rows", "hit_rows", "join_cov_pct"]].to_string(index=False))

    if negative_delay > 0 or join_cov < 100.0:
        print("diagnostic_hints=")
        if negative_delay > 0:
            print("- timezone mismatch or wrong event-bar mapping (negative delays should be 0)")
        if join_cov < 100.0:
            print("- SecuCode formatting mismatch (must be 6-digit zero-padded)")
            print("- TradingDay conversion mismatch (expect YYYYMMDD from t_event_bar local date)")
            print("- TimeStart conversion mismatch (expect HHMM from t_event_bar local time)")


if __name__ == "__main__":
    main()
