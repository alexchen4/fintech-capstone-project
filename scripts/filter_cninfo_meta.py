#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentiment.meta import filter_meta_to_universe, summary_stats

DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "universe.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "cninfo_meta"


def _infer_out_parquet(meta_path: Path) -> Path:
    m = re.search(r"(\d{4}-\d{2})", meta_path.name)
    if m:
        tag = m.group(1)
        return DEFAULT_OUT_DIR / f"announcements_meta_{tag}.parquet"
    return DEFAULT_OUT_DIR / "announcements_meta.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter CNINFO meta to strict 50-code universe and save parquet.")
    parser.add_argument("--meta_csv", type=Path, required=True, help="Input meta file (.csv or .parquet)")
    parser.add_argument("--universe_csv", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--out_parquet", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.meta_csv.exists():
        raise FileNotFoundError(f"Meta input not found: {args.meta_csv}")
    if not args.universe_csv.exists():
        raise FileNotFoundError(f"Universe file not found: {args.universe_csv}")

    out_parquet = args.out_parquet or _infer_out_parquet(args.meta_csv)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    filtered = filter_meta_to_universe(args.meta_csv, args.universe_csv)
    filtered.to_parquet(out_parquet, index=False)

    stats = summary_stats(filtered)
    print("[filter_cninfo_meta] summary")
    print(f"meta_input={args.meta_csv}")
    print(f"universe_csv={args.universe_csv}")
    print(f"out_parquet={out_parquet}")
    print(f"rows={stats['rows']}")
    print(f"unique_SecuCode={stats['unique_SecuCode']}")
    print(f"publish_dt_min={stats['publish_dt_min']}")
    print(f"publish_dt_max={stats['publish_dt_max']}")


if __name__ == "__main__":
    main()
