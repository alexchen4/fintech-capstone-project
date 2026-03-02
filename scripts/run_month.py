#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PHASES = {
    "universe": [
        "scripts/build_universe_from_price.py",
    ],
    "meta": [
        "scripts/filter_cninfo_meta.py",
    ],
    "payload": [
        "scripts/fetch_cninfo_payload.py",
    ],
    "text": [
        "scripts/parse_cninfo_text.py",
    ],
    "dataset": [
        "scripts/build_sentiment_dataset.py",
    ],
    "labels": [
        "scripts/label_from_returns.py",
    ],
    "features": [
        "scripts/build_sentiment_features_15m.py",
    ],
    "gate": [
        "scripts/pre_modeling_gate.py",
    ],
}

ORDER = ["universe", "meta", "payload", "text", "dataset", "labels", "features", "gate"]


def _run(cmd: list[str]) -> None:
    print(f"[run_month] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single entrypoint to run monthly sentiment pipeline phases.")
    p.add_argument("--month", required=True, help="Month tag YYYY-MM")
    p.add_argument("--phase", action="append", choices=ORDER + ["all"], default=["all"])
    p.add_argument("--price_csv", type=Path, default=ROOT / "data" / "qfq_15min_all.csv")
    p.add_argument("--universe_csv", type=Path, default=ROOT / "data" / "universe" / "universe.csv")
    p.add_argument("--meta_csv", type=Path, default=ROOT / "data" / "raw" / "meta" / "announcements_meta_2025-12.csv")
    p.add_argument("--limit", type=int, default=None, help="Optional payload fetch limit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    month = args.month
    meta_parquet = ROOT / "data" / "raw" / "cninfo_meta" / f"announcements_meta_{month}.parquet"
    text_parquet = ROOT / "data" / "processed" / "cninfo_text" / f"ann_text_{month}.parquet"
    dataset_parquet = ROOT / "data" / "processed" / "sentiment" / f"sentiment_dataset_{month}.parquet"

    phases = ORDER if "all" in args.phase else [p for p in ORDER if p in set(args.phase)]

    for ph in phases:
        if ph == "universe":
            _run(
                [
                    sys.executable,
                    "scripts/build_universe_from_price.py",
                    "--price_csv",
                    str(args.price_csv),
                ]
            )
        elif ph == "meta":
            _run(
                [
                    sys.executable,
                    "scripts/filter_cninfo_meta.py",
                    "--meta_csv",
                    str(args.meta_csv),
                    "--universe_csv",
                    str(args.universe_csv),
                    "--out_parquet",
                    str(meta_parquet),
                ]
            )
        elif ph == "payload":
            cmd = [
                sys.executable,
                "scripts/fetch_cninfo_payload.py",
                "--meta_parquet",
                str(meta_parquet),
                "--universe_csv",
                str(args.universe_csv),
            ]
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            _run(cmd)
        elif ph == "text":
            _run(
                [
                    sys.executable,
                    "scripts/parse_cninfo_text.py",
                    "--meta_parquet",
                    str(meta_parquet),
                    "--pdf_dir",
                    str(ROOT / "data" / "raw" / "cninfo_payload" / "raw_pdfs"),
                    "--out_parquet",
                    str(text_parquet),
                ]
            )
        elif ph == "dataset":
            _run(
                [
                    sys.executable,
                    "scripts/build_sentiment_dataset.py",
                    "--ann_text_parquet",
                    str(text_parquet),
                    "--price_csv",
                    str(args.price_csv),
                    "--universe_csv",
                    str(args.universe_csv),
                    "--out_parquet",
                    str(dataset_parquet),
                ]
            )
        elif ph == "labels":
            _run(
                [
                    sys.executable,
                    "scripts/label_from_returns.py",
                    "--dataset_parquet",
                    str(dataset_parquet),
                    "--price_csv",
                    str(args.price_csv),
                    "--universe_csv",
                    str(args.universe_csv),
                ]
            )
        elif ph == "features":
            _run(
                [
                    sys.executable,
                    "scripts/build_sentiment_features_15m.py",
                    "--dataset_parquet",
                    str(dataset_parquet),
                    "--price_csv",
                    str(args.price_csv),
                    "--universe_csv",
                    str(args.universe_csv),
                ]
            )
        elif ph == "gate":
            _run(
                [
                    sys.executable,
                    "scripts/pre_modeling_gate.py",
                    "--month",
                    month,
                    "--universe_csv",
                    str(args.universe_csv),
                    "--price_csv",
                    str(args.price_csv),
                ]
            )

    print(f"[run_month] DONE phases={phases} month={month}")


if __name__ == "__main__":
    main()
