#!/usr/bin/env python3
"""
Smoke test for the CNINFO sentiment MVP pipeline.

This script validates a minimal local run path on a small sample (N=5):
metadata load -> PDF download/cache -> text extraction -> lexicon scoring.
All outputs are written to data/interim/ (gitignored).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cninfo.io import ensure_dir, write_parquet
from cninfo.pdf_text import batch_extract_text, download_pdfs
from cninfo.sentiment import batch_score, load_lexicons

META_CANDIDATES = [
    ROOT / "data" / "raw" / "meta.parquet",
    ROOT / "data" / "raw" / "meta.csv",
    ROOT / "data" / "raw" / "cninfo_meta.parquet",
    ROOT / "data" / "raw" / "cninfo_meta.csv",
]

REQUIRED_COLUMNS = [
    "SecuCode",
    "publish_dt",
    "title",
    "category",
    "pdf_url",
    "announcement_id",
]


def _load_meta() -> pd.DataFrame | None:
    meta_path = next((p for p in META_CANDIDATES if p.exists()), None)
    if meta_path is None:
        print("No local metadata file found.")
        print("Provide one of:")
        for p in META_CANDIDATES:
            print(f"  - {p.relative_to(ROOT)}")
        return None

    if meta_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(meta_path)
    else:
        df = pd.read_csv(meta_path)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    print(f"Loaded metadata: {meta_path.relative_to(ROOT)} rows={len(df)}")
    return df


def _load_lexicons(raw_dir: Path) -> tuple[set[str], set[str]]:
    pos_path = raw_dir / "lexicon_pos.txt"
    neg_path = raw_dir / "lexicon_neg.txt"
    if pos_path.exists() and neg_path.exists():
        print("Using lexicons from data/raw.")
        return load_lexicons(str(pos_path), str(neg_path))

    print("Lexicon files not found; using tiny placeholder lexicons.")
    return {"growth", "profit", "improve"}, {"loss", "risk", "decline"}


def main() -> int:
    raw_dir = ensure_dir(ROOT / "data" / "raw")
    interim_dir = ensure_dir(ROOT / "data" / "interim")
    pdf_dir = ensure_dir(raw_dir / "pdfs")

    meta_df = _load_meta()
    if meta_df is None:
        return 0

    sample_df = meta_df.head(5).copy()
    if sample_df.empty:
        print("Metadata file is empty; nothing to test.")
        return 0

    print(f"Running smoke test on first {len(sample_df)} announcements...")

    pdf_manifest = download_pdfs(sample_df, str(pdf_dir), url_col="pdf_url", id_col="announcement_id")
    text_df = batch_extract_text(pdf_manifest)
    pos_set, neg_set = _load_lexicons(raw_dir)
    scored_df = batch_score(text_df, pos_set, neg_set, text_col="text")

    manifest_out = interim_dir / "smoke_pdf_manifest.parquet"
    texts_out = interim_dir / "smoke_texts.parquet"
    scored_out = interim_dir / "smoke_scored.parquet"

    write_parquet(pdf_manifest, manifest_out)
    write_parquet(text_df, texts_out)
    write_parquet(scored_df, scored_out)

    success_rate = float((text_df["error"].isna() | (text_df["error"] == "")).mean()) if len(text_df) else 0.0
    print(f"Extraction success rate: {success_rate:.2%}")

    if "char_count" in text_df.columns and len(text_df):
        print("char_count stats:")
        print(text_df["char_count"].describe(percentiles=[0.5, 0.9, 0.99]))

    sample_row = scored_df.iloc[0]
    sample_title = str(sample_row.get("title", ""))
    sample_text = str(sample_row.get("text", ""))[:200]
    print("Sample title:", sample_title)
    print("Sample text[0:200]:", sample_text)

    print(f"Wrote: {manifest_out.relative_to(ROOT)}")
    print(f"Wrote: {texts_out.relative_to(ROOT)}")
    print(f"Wrote: {scored_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
