#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import subprocess
import sys
from typing import Any
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


def _maybe_delete_month_pdfs_after_parse(month: str, text_parquet: Path, pdf_dir: Path) -> None:
    # Keep default behavior unchanged unless explicit flag is set.
    if not text_parquet.exists():
        print(f"[run_month] skip pdf deletion: text parquet not found: {text_parquet}")
        return
    if not pdf_dir.exists():
        print(f"[run_month] skip pdf deletion: pdf_dir not found: {pdf_dir}")
        return

    try:
        import pandas as pd  # Lazy import so non-flag runs keep current behavior.
    except Exception as exc:  # pragma: no cover
        print(f"[run_month] skip pdf deletion: pandas import failed: {exc}")
        return

    df = pd.read_parquet(text_parquet, columns=["ann_id", "parse_status"])
    rows = int(len(df))
    ok_rows = int((df["parse_status"] == "ok").sum()) if rows else 0
    parse_ok_rate = (100.0 * ok_rows / rows) if rows else 0.0

    if parse_ok_rate < 85.0:
        print(
            f"[run_month] keep raw PDFs: month={month} parse_ok_rate={parse_ok_rate:.2f}% < 85.00%"
        )
        return

    ann_ids = sorted({str(x).strip() for x in df["ann_id"].tolist() if str(x).strip()})
    deleted_ann_ids: list[str] = []
    freed_bytes = 0

    for ann_id in ann_ids:
        pdf_path = pdf_dir / f"{ann_id}.pdf"
        if not pdf_path.exists():
            continue
        freed_bytes += int(pdf_path.stat().st_size)
        pdf_path.unlink()
        deleted_ann_ids.append(ann_id)

    freed_mb = round(freed_bytes / (1024 * 1024), 2)
    manifest = ROOT / "data" / "raw" / "cninfo_payload" / f"parsed_and_cleared_{month}.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        f"month={month}\n"
        f"parse_ok_rate={parse_ok_rate:.2f}\n"
        f"ann_ids_cleared={','.join(deleted_ann_ids)}\n"
        f"total_files_deleted={len(deleted_ann_ids)}\n"
        f"mb_freed={freed_mb:.2f}\n"
        f"generated_utc={datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    print(f"[run_month] deleted_pdfs month={month} files={len(deleted_ann_ids)} mb_freed={freed_mb:.2f} manifest={manifest}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single entrypoint to run monthly sentiment pipeline phases.")
    p.add_argument("--month", required=True, help="Month tag YYYY-MM")
    p.add_argument("--phase", action="append", choices=ORDER + ["all"], default=["all"])
    p.add_argument("--price_csv", type=Path, default=ROOT / "data" / "qfq_15min_all.csv")
    p.add_argument("--universe_csv", type=Path, default=ROOT / "data" / "universe" / "universe.csv")
    p.add_argument("--meta_csv", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Optional payload fetch limit")
    p.add_argument("--delete-pdfs-after-parse", action="store_true", help="If parse_ok_rate >= 85%%, delete month PDFs and write parsed_and_cleared manifest.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    month = args.month
    if args.meta_csv is None:
        args.meta_csv = (
            ROOT
            / "data"
            / "raw"
            / "meta"
            / f"announcements_meta_{month}.csv"
        )
    meta_parquet = ROOT / "data" / "raw" / "cninfo_meta" / f"announcements_meta_{month}.parquet"
    pdf_dir = ROOT / "data" / "raw" / "cninfo_payload" / "raw_pdfs"
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
                    str(pdf_dir),
                    "--out_parquet",
                    str(text_parquet),
                ]
            )
            if args.delete_pdfs_after_parse:
                _maybe_delete_month_pdfs_after_parse(month=month, text_parquet=text_parquet, pdf_dir=pdf_dir)
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
