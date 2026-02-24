#!/usr/bin/env python3
"""Deterministic local smoke test for CNINFO sentiment MVP using fixture metadata."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

FIXTURE_META = ROOT / "data" / "raw" / "fixtures" / "meta_fixture.csv"
FIXTURE_PDF_DIR = ROOT / "data" / "raw" / "fixtures" / "pdfs"
OUT_DIR = ROOT / "data" / "interim" / "smoke_test_outputs"

REQUIRED_COLUMNS = [
    "SecuCode",
    "publish_dt",
    "title",
    "pdf_url",
    "announcement_id",
    "local_pdf_path",
]


def _require_package(module_name: str, install_hint: str) -> bool:
    if importlib.util.find_spec(module_name) is not None:
        return True
    print(f"Missing dependency: {module_name}")
    print(f"Install hint: {install_hint}")
    return False


def _print_missing_inputs() -> None:
    print("Missing required fixture inputs.")
    print(f"Expected metadata CSV: {FIXTURE_META}")
    print(f"Expected local PDFs under: {FIXTURE_PDF_DIR}")
    print("Setup:")
    print("  1) Ensure data/raw/fixtures/meta_fixture.csv exists.")
    print("  2) Drop 1-3 PDFs into data/raw/fixtures/pdfs/.")
    print("  3) Update local_pdf_path in meta_fixture.csv if needed.")


def _load_fixture_meta() -> pd.DataFrame | None:
    import pandas as pd

    if not FIXTURE_META.exists():
        _print_missing_inputs()
        return None

    df = pd.read_csv(FIXTURE_META)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"Fixture metadata missing columns: {missing_cols}")
        return None

    if df.empty:
        print("Fixture metadata is empty.")
        return None

    return df.copy()


def _load_lexicons() -> tuple[set[str], set[str]]:
    from cninfo.sentiment import load_lexicons

    pos_path = ROOT / "data" / "raw" / "lexicon_pos.txt"
    neg_path = ROOT / "data" / "raw" / "lexicon_neg.txt"
    if pos_path.exists() and neg_path.exists():
        return load_lexicons(str(pos_path), str(neg_path))

    # Tiny deterministic fallback to keep smoke test local and runnable.
    return {"growth", "profit", "improve", "stable"}, {"loss", "risk", "decline", "penalty"}


def _resolve_local_paths(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def resolve_one(path_str: object) -> str | None:
        if not isinstance(path_str, str) or not path_str.strip():
            return None
        p = Path(path_str)
        if not p.is_absolute():
            p = ROOT / p
        return str(p)

    out["local_path"] = out["local_pdf_path"].map(resolve_one)
    return out


def _validate_local_pdfs(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    out = df.copy()
    out["has_local_pdf"] = out["local_path"].map(lambda p: isinstance(p, str) and Path(p).exists())
    if not out["has_local_pdf"].all():
        missing = out.loc[~out["has_local_pdf"], ["announcement_id", "local_pdf_path"]]
        print("Some fixture PDFs are missing. Add files and rerun.")
        print(missing.to_string(index=False))
        return out, False
    return out, True


def main() -> int:
    deps_ok = True
    deps_ok &= _require_package("pandas", "python3 -m pip install pandas")
    deps_ok &= _require_package("fitz", "python3 -m pip install pymupdf")
    deps_ok &= _require_package("pdfplumber", "python3 -m pip install pdfplumber")
    if not deps_ok:
        print("Smoke test did not run because required packages are missing.")
        return 1

    from cninfo.io import ensure_dir
    from cninfo.pdf_text import batch_extract_text, download_pdfs
    from cninfo.sentiment import batch_score

    ensure_dir(OUT_DIR)

    meta_df = _load_fixture_meta()
    if meta_df is None:
        return 1

    fixture_df = _resolve_local_paths(meta_df)
    fixture_df, ok = _validate_local_pdfs(fixture_df)
    if not ok:
        _print_missing_inputs()
        return 1

    # Skip network download path when local PDFs are provided and present.
    if fixture_df["has_local_pdf"].all():
        manifest_df = fixture_df.drop(columns=["has_local_pdf"]).copy()
    else:
        manifest_df = download_pdfs(fixture_df, str(FIXTURE_PDF_DIR), url_col="pdf_url", id_col="announcement_id")

    text_df = batch_extract_text(manifest_df)
    pos_set, neg_set = _load_lexicons()
    scored_df = batch_score(text_df, pos_set, neg_set, text_col="text")

    manifest_out = OUT_DIR / "smoke_manifest.csv"
    text_out = OUT_DIR / "smoke_texts.csv"
    scored_out = OUT_DIR / "smoke_events_scored.csv"

    manifest_df.to_csv(manifest_out, index=False)
    text_df.to_csv(text_out, index=False)
    scored_df.to_csv(scored_out, index=False)

    print(f"docs_processed: {len(scored_df)}")

    method_counts = scored_df["method"].fillna("none").value_counts().to_dict() if "method" in scored_df else {}
    print(f"extraction_method_counts: {method_counts}")

    if "char_count" in scored_df and len(scored_df):
        cc = scored_df["char_count"].fillna(0)
        print(
            "char_count_stats: "
            f"min={int(cc.min())}, median={float(cc.median()):.1f}, max={int(cc.max())}"
        )
    else:
        print("char_count_stats: unavailable")

    sample_cols = ["SecuCode", "publish_dt", "sent_score", "text"]
    for col in sample_cols:
        if col not in scored_df.columns:
            scored_df[col] = None

    print("sample_output:")
    for _, row in scored_df.head(3).iterrows():
        snippet = str(row.get("text") or "")[:200].replace("\n", " ")
        print(
            f"  ({row.get('SecuCode')}, {row.get('publish_dt')}, "
            f"{row.get('sent_score')}, {snippet})"
        )

    print(f"wrote: {manifest_out}")
    print(f"wrote: {text_out}")
    print(f"wrote: {scored_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
