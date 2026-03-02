#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.contracts import ContractError, run_all_contracts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate sentiment pipeline artifacts against strict data contracts.")
    p.add_argument("--month", required=True, help="Month tag, format YYYY-MM")
    p.add_argument("--universe_csv", type=Path, default=ROOT / "data" / "universe" / "universe.csv")
    p.add_argument("--meta_path", type=Path, default=None)
    p.add_argument("--text_path", type=Path, default=None)
    p.add_argument("--dataset_path", type=Path, default=None)
    return p.parse_args()


def _default_paths(month: str) -> tuple[Path, Path, Path]:
    meta = ROOT / "data" / "raw" / "cninfo_meta" / f"announcements_meta_{month}.parquet"
    text = ROOT / "data" / "processed" / "cninfo_text" / f"ann_text_{month}.parquet"
    ds = ROOT / "data" / "processed" / "sentiment" / f"sentiment_dataset_{month}.parquet"
    return meta, text, ds


def main() -> None:
    args = parse_args()
    meta_d, text_d, ds_d = _default_paths(args.month)

    meta_path = args.meta_path or meta_d
    text_path = args.text_path or text_d
    dataset_path = args.dataset_path or ds_d

    print("[validate_outputs] inputs")
    print(f"month={args.month}")
    print(f"universe_csv={args.universe_csv}")
    print(f"meta_path={meta_path}")
    print(f"text_path={text_path}")
    print(f"dataset_path={dataset_path}")

    try:
        reports = run_all_contracts(
            universe_csv=args.universe_csv,
            meta_path=meta_path,
            text_path=text_path,
            dataset_path=dataset_path,
        )
    except ContractError as exc:
        print("[validate_outputs] FAIL")
        print(str(exc))
        raise SystemExit(2)

    print("[validate_outputs] PASS")
    for r in reports:
        print(
            f"- {r.name}: rows={r.rows}, unique_tickers={r.unique_tickers}, "
            f"unique_ann_id={r.unique_ann_id}"
        )


if __name__ == "__main__":
    main()
