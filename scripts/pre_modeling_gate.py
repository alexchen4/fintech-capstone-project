#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-modeling quality gate for sentiment data artifacts.")
    p.add_argument("--month", required=True, help="Month tag YYYY-MM")
    p.add_argument("--universe_csv", type=Path, default=ROOT / "data" / "universe" / "universe.csv")
    p.add_argument("--price_csv", type=Path, default=ROOT / "data" / "qfq_15min_all.csv")
    p.add_argument("--meta_path", type=Path, default=None)
    p.add_argument("--text_path", type=Path, default=None)
    p.add_argument("--dataset_path", type=Path, default=None)

    # Thresholds (healthy defaults from docs)
    p.add_argument("--min_parse_ok_rate", type=float, default=95.0)
    p.add_argument("--max_empty_text_rate", type=float, default=5.0)
    p.add_argument("--min_alignment_non_null", type=float, default=99.0)
    p.add_argument("--min_join_coverage", type=float, default=99.0)
    p.add_argument("--max_negative_delay_count", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=500_000)
    return p.parse_args()


def _default_paths(month: str) -> tuple[Path, Path, Path]:
    meta = ROOT / "data" / "raw" / "cninfo_meta" / f"announcements_meta_{month}.parquet"
    text = ROOT / "data" / "processed" / "cninfo_text" / f"ann_text_{month}.parquet"
    ds = ROOT / "data" / "processed" / "sentiment" / f"sentiment_dataset_{month}.parquet"
    return meta, text, ds


def _run_cmd(cmd: list[str]) -> None:
    print(f"[pre_modeling_gate] run: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _compute_metrics(dataset_path: Path, price_csv: Path, chunksize: int) -> dict[str, float | int]:
    ds = pd.read_parquet(dataset_path)
    total = len(ds)
    if total == 0:
        return {
            "parse_ok_rate": 0.0,
            "empty_text_rate": 100.0,
            "alignment_non_null": 0.0,
            "negative_delay_count": 0,
            "join_coverage": 0.0,
        }

    parse_ok_rate = 100.0 * int((ds["parse_status"].astype(str) == "ok").sum()) / total

    len_col = "text_len_chars" if "text_len_chars" in ds.columns else "text_len"
    text_len = pd.to_numeric(ds[len_col], errors="coerce").fillna(0)
    empty_text_rate = 100.0 * int((text_len == 0).sum()) / total

    pub_utc = pd.to_datetime(ds["publish_dt_utc"], errors="coerce", utc=True)
    bar_local = pd.to_datetime(ds["t_event_bar"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai")
    pub_local = pub_utc.dt.tz_convert("Asia/Shanghai")

    alignment_non_null = 100.0 * int(bar_local.notna().sum()) / total
    negative_delay_count = int(((bar_local - pub_local).dt.total_seconds() / 60.0 < 0).sum())

    keys_df = pd.DataFrame(
        {
            "SecuCode": ds["SecuCode"].astype(str).str.strip().str.zfill(6),
            "TradingDay": bar_local.dt.strftime("%Y%m%d"),
            "TimeStart": bar_local.dt.strftime("%H%M"),
        }
    ).dropna()
    required_keys = set(zip(keys_df["SecuCode"].tolist(), keys_df["TradingDay"].tolist(), keys_df["TimeStart"].tolist()))

    matched: set[tuple[str, str, str]] = set()
    if required_keys:
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

    join_coverage = 100.0 * len(matched) / len(required_keys) if required_keys else 0.0

    return {
        "parse_ok_rate": round(parse_ok_rate, 4),
        "empty_text_rate": round(empty_text_rate, 4),
        "alignment_non_null": round(alignment_non_null, 4),
        "negative_delay_count": negative_delay_count,
        "join_coverage": round(join_coverage, 4),
    }


def _assert_thresholds(metrics: dict[str, float | int], args: argparse.Namespace) -> None:
    failures: list[str] = []
    if float(metrics["parse_ok_rate"]) < args.min_parse_ok_rate:
        failures.append(f"parse_ok_rate {metrics['parse_ok_rate']} < {args.min_parse_ok_rate}")
    if float(metrics["empty_text_rate"]) > args.max_empty_text_rate:
        failures.append(f"empty_text_rate {metrics['empty_text_rate']} > {args.max_empty_text_rate}")
    if float(metrics["alignment_non_null"]) < args.min_alignment_non_null:
        failures.append(f"alignment_non_null {metrics['alignment_non_null']} < {args.min_alignment_non_null}")
    if float(metrics["join_coverage"]) < args.min_join_coverage:
        failures.append(f"join_coverage {metrics['join_coverage']} < {args.min_join_coverage}")
    if int(metrics["negative_delay_count"]) > args.max_negative_delay_count:
        failures.append(
            f"negative_delay_count {metrics['negative_delay_count']} > {args.max_negative_delay_count}"
        )

    if failures:
        print("[pre_modeling_gate] FAIL")
        for x in failures:
            print(f"- {x}")
        raise SystemExit(3)


def main() -> None:
    args = parse_args()
    meta_d, text_d, ds_d = _default_paths(args.month)
    meta_path = args.meta_path or meta_d
    text_path = args.text_path or text_d
    dataset_path = args.dataset_path or ds_d

    # 1) Data contracts
    _run_cmd(
        [
            sys.executable,
            "scripts/validate_outputs.py",
            "--month",
            args.month,
            "--universe_csv",
            str(args.universe_csv),
            "--meta_path",
            str(meta_path),
            "--text_path",
            str(text_path),
            "--dataset_path",
            str(dataset_path),
        ]
    )

    # 2) Text quality audit
    _run_cmd(
        [
            sys.executable,
            "scripts/audit_text_quality.py",
            "--input_path",
            str(dataset_path),
            "--universe_csv",
            str(args.universe_csv),
            "--sample_n",
            "60",
            "--seed",
            "42",
        ]
    )

    # 3) Alignment + join audit
    _run_cmd(
        [
            sys.executable,
            "scripts/audit_alignment_and_price_join.py",
            "--dataset_parquet",
            str(dataset_path),
            "--price_csv",
            str(args.price_csv),
            "--chunksize",
            str(args.chunksize),
        ]
    )

    # 4) Fingerprinting
    _run_cmd(
        [
            sys.executable,
            "scripts/fingerprint_artifacts.py",
            "--paths",
            str(dataset_path),
            str(text_path),
            "--month",
            args.month,
        ]
    )

    # Final threshold gate
    metrics = _compute_metrics(dataset_path=dataset_path, price_csv=args.price_csv, chunksize=args.chunksize)
    print("[pre_modeling_gate] threshold_metrics")
    for k, v in metrics.items():
        print(f"{k}={v}")

    _assert_thresholds(metrics, args)
    print("[pre_modeling_gate] PASS")


if __name__ == "__main__":
    main()
