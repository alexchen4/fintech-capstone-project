"""CNINFO meta filtering utilities (strict universe-first)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from common.secu import normalize_secu_series
from common.universe import validate_universe

REQUIRED_OUTPUT_COLUMNS = ["ann_id", "SecuCode", "publish_dt_utc", "title", "detail_url"]

# Documents that are structurally uninformative for sentiment:
# typically scanned PDFs with no extractable text, or pure
# administrative/boilerplate filings with zero news signal.
TITLE_BLOCKLIST: list[str] = [
    "法律意见书",        # Legal opinions - almost always scanned
    "受托管理事务报告",  # Bond trustee periodic reports - boilerplate
    "临时受托管理",      # Bond trustee interim reports - boilerplate
    "债券受托",          # Bond trustee misc
    "证券变动月报表",    # H-share monthly securities change forms
    "验资报告",          # Capital verification reports - scanned
    "审计报告",          # Audit reports - scanned
    "司法",              # Court/judicial notices - scanned
    "仲裁",              # Arbitration notices - scanned
]


def _pick_first_existing(cols: list[str], candidates: list[str], label: str) -> str:
    lower_map = {c.lower(): c for c in cols}
    for k in candidates:
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    raise ValueError(f"Missing required {label} column. Expected one of: {candidates}")


def _extract_ann_id_from_url(url: object) -> str:
    s = "" if pd.isna(url) else str(url)
    m = re.search(r"/(\d+)\.(pdf|PDF|html|HTML)$", s)
    return m.group(1) if m else ""


def _ensure_publish_dt_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _load_meta(meta_path: Path) -> pd.DataFrame:
    suffix = meta_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(meta_path)
    if suffix == ".parquet":
        return pd.read_parquet(meta_path)
    raise ValueError(f"Unsupported meta format: {meta_path}. Use .csv or .parquet")


def _load_universe(universe_csv: Path) -> list[str]:
    uni = pd.read_csv(universe_csv)
    code_col = _pick_first_existing(list(uni.columns), ["SecuCode", "ticker", "secu_code"], "universe code")
    codes = normalize_secu_series(uni[code_col]).dropna().astype(str)
    out = sorted(set(codes.tolist()))
    if len(out) != 50:
        raise ValueError(f"Universe must contain exactly 50 SecuCodes, got {len(out)} from {universe_csv}")
    return out


def filter_meta_to_universe(meta_path: Path, universe_csv: Path) -> pd.DataFrame:
    raw = _load_meta(meta_path)
    cols = list(raw.columns)

    code_col = _pick_first_existing(cols, ["SecuCode", "ticker", "secu_code", "symbol"], "SecuCode")
    publish_col = _pick_first_existing(cols, ["publish_dt", "publish_ts", "publish_time", "publish_datetime"], "publish datetime")
    title_col = _pick_first_existing(cols, ["title", "announcement_title"], "title")
    detail_col = _pick_first_existing(cols, ["detail_url", "pdf_url", "url"], "detail URL")

    ann_col = None
    for c in ["ann_id", "announcement_id", "announcementId"]:
        if c in raw.columns:
            ann_col = c
            break

    df = raw.copy()
    df["SecuCode"] = normalize_secu_series(df[code_col])
    df["publish_dt_utc"] = _ensure_publish_dt_utc(df[publish_col])
    df["title"] = df[title_col].fillna("").astype(str).str.strip()
    df["detail_url"] = df[detail_col].fillna("").astype(str).str.strip()

    if ann_col is not None:
        df["ann_id"] = df[ann_col].fillna("").astype(str).str.strip()
    else:
        df["ann_id"] = ""

    missing_ann = ~df["ann_id"].str.fullmatch(r"\d+")
    if missing_ann.any():
        extracted = df.loc[missing_ann, "detail_url"].map(_extract_ann_id_from_url)
        df.loc[missing_ann, "ann_id"] = extracted

    df = df[df["ann_id"].str.fullmatch(r"\d+")].copy()

    universe_codes = set(_load_universe(universe_csv))
    df = df[df["SecuCode"].isin(universe_codes)].copy()

    # Drop structurally uninformative document types
    blocklist_pattern = "|".join(re.escape(kw) for kw in TITLE_BLOCKLIST)
    before_block = len(df)
    df = df[~df["title"].str.contains(blocklist_pattern, na=False)].copy()
    after_block = len(df)
    if before_block != after_block:
        import sys
        print(f"[filter_meta] blocklist dropped {before_block - after_block} rows "
              f"({before_block} -> {after_block})", file=sys.stderr)

    validate_universe(df, universe_codes, col="SecuCode")

    out = df[REQUIRED_OUTPUT_COLUMNS].drop_duplicates(subset=["ann_id"], keep="last").copy()
    out = out.sort_values(["publish_dt_utc", "ann_id"], kind="stable").reset_index(drop=True)
    return out


def summary_stats(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "rows": 0,
            "unique_SecuCode": 0,
            "publish_dt_min": None,
            "publish_dt_max": None,
        }

    dt = pd.to_datetime(df["publish_dt_utc"], errors="coerce", utc=True)
    return {
        "rows": int(len(df)),
        "unique_SecuCode": int(df["SecuCode"].nunique()),
        "publish_dt_min": None if dt.isna().all() else str(dt.min()),
        "publish_dt_max": None if dt.isna().all() else str(dt.max()),
    }
