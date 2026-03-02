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

from sentiment.parse_text import parse_cninfo_text, parse_summary

DEFAULT_META = ROOT / "data" / "raw" / "cninfo_meta" / "announcements_meta_2025-12.parquet"
DEFAULT_PDF_DIR = ROOT / "data" / "raw" / "cninfo_payload" / "raw_pdfs"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "cninfo_text"


def _infer_out_parquet(meta_path: Path) -> Path:
    m = re.search(r"(\d{4}-\d{2})", meta_path.name)
    tag = m.group(1) if m else "unknown"
    return DEFAULT_OUT_DIR / f"ann_text_{tag}.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse CNINFO payload PDFs into clean announcement text table.")
    parser.add_argument("--meta_parquet", type=Path, default=DEFAULT_META)
    parser.add_argument("--pdf_dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--out_parquet", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.meta_parquet.exists():
        raise FileNotFoundError(f"meta parquet not found: {args.meta_parquet}")
    if not args.pdf_dir.exists():
        raise FileNotFoundError(f"pdf_dir not found: {args.pdf_dir}")

    out_parquet = args.out_parquet or _infer_out_parquet(args.meta_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    out = parse_cninfo_text(args.meta_parquet, args.pdf_dir)
    out.to_parquet(out_parquet, index=False)

    s = parse_summary(out)
    print("[parse_cninfo_text] summary")
    print(f"meta_parquet={args.meta_parquet}")
    print(f"pdf_dir={args.pdf_dir}")
    print(f"out_parquet={out_parquet}")
    print(f"rows={s['rows']}")
    print(f"pct_parse_ok={s['pct_ok']}")
    print(f"pct_empty={s['pct_empty']}")
    print(f"median_text_len={s['median_text_len']}")


if __name__ == "__main__":
    main()
