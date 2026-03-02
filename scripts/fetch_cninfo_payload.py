#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentiment.fetch_payload import fetch_payloads, infer_month_tag

DEFAULT_META = ROOT / "data" / "raw" / "cninfo_meta" / "announcements_meta_2025-12.parquet"
DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "cninfo_payload" / "raw_pdfs"
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "universe.csv"
DEFAULT_FAIL_DIR = ROOT / "data" / "raw" / "cninfo_payload"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CNINFO payload files from filtered meta parquet.")
    parser.add_argument("--meta_parquet", type=Path, default=DEFAULT_META)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--universe_csv", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep_sec", type=float, default=0.25)
    parser.add_argument("--jitter_sec", type=float, default=0.15)
    parser.add_argument("--timeout_sec", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.meta_parquet.exists():
        raise FileNotFoundError(f"meta parquet not found: {args.meta_parquet}")
    if not args.universe_csv.exists():
        raise FileNotFoundError(f"universe file not found: {args.universe_csv}")

    month_tag = infer_month_tag(args.meta_parquet)
    failures_csv = DEFAULT_FAIL_DIR / f"failures_{month_tag}.csv"

    stats = fetch_payloads(
        meta_parquet=args.meta_parquet,
        out_dir=args.out_dir,
        universe_csv=args.universe_csv,
        failures_csv=failures_csv,
        limit=args.limit,
        sleep_sec=args.sleep_sec,
        jitter_sec=args.jitter_sec,
        timeout_sec=args.timeout_sec,
    )

    print("[fetch_cninfo_payload] summary")
    print(f"meta_parquet={args.meta_parquet}")
    print(f"out_dir={args.out_dir}")
    print(f"failures_csv={failures_csv}")
    print(f"limit={args.limit}")
    print(f"total={stats['total']}")
    print(f"downloaded={stats['downloaded']}")
    print(f"skipped_existing={stats['skipped_existing']}")
    print(f"failed={stats['failed']}")


if __name__ == "__main__":
    main()
