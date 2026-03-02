"""Data contracts for sentiment pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from common.secu import normalize_secu_series


class ContractError(ValueError):
    """Raised when a contract validation fails."""


@dataclass
class ContractReport:
    name: str
    rows: int
    unique_tickers: int
    unique_ann_id: int | None


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ContractError(f"[{name}] missing required columns: {missing}")


def _normalize_secu_col(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    if col not in df.columns:
        raise ContractError(f"[{name}] missing column: {col}")
    try:
        secu = normalize_secu_series(df[col])
    except Exception as exc:
        raise ContractError(f"[{name}] failed SecuCode normalization: {exc}") from exc

    bad_len = secu.astype(str).str.len() != 6
    if bool(bad_len.any()):
        raise ContractError(f"[{name}] SecuCode len!=6 for {int(bad_len.sum())} rows")
    return secu


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ContractError(f"file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ContractError(f"unsupported file format: {path}")


def _load_universe_set(universe_csv: Path) -> set[str]:
    df = _load_table(universe_csv)
    if df.empty:
        raise ContractError(f"[universe] empty file: {universe_csv}")
    col = "SecuCode" if "SecuCode" in df.columns else df.columns[0]
    secu = _normalize_secu_col(df, col, "universe")
    codes = set(secu.tolist())
    if len(codes) != 50:
        raise ContractError(f"[universe] expected 50 unique SecuCodes, got {len(codes)}")
    return codes


def validate_universe_contract(universe_csv: Path) -> ContractReport:
    df = _load_table(universe_csv)
    col = "SecuCode" if "SecuCode" in df.columns else df.columns[0]
    secu = _normalize_secu_col(df, col, "universe")
    unique = int(pd.Series(secu).nunique())
    if unique != 50:
        raise ContractError(f"[universe] expected 50 unique SecuCodes, got {unique}")
    return ContractReport(name="universe", rows=int(len(df)), unique_tickers=unique, unique_ann_id=None)


def validate_meta_contract(meta_path: Path, universe_set: set[str]) -> ContractReport:
    name = "meta"
    df = _load_table(meta_path)
    _ensure_columns(df, ["ann_id", "SecuCode", "publish_dt_utc", "title", "detail_url"], name)

    secu = _normalize_secu_col(df, "SecuCode", name)
    ann_id = df["ann_id"].astype(str).str.strip()
    if bool((ann_id == "").any()):
        raise ContractError(f"[{name}] ann_id has empty values")
    if bool(ann_id.duplicated().any()):
        raise ContractError(f"[{name}] ann_id must be unique; duplicates={int(ann_id.duplicated().sum())}")

    dt = pd.to_datetime(df["publish_dt_utc"], errors="coerce", utc=True)
    if bool(dt.isna().any()):
        raise ContractError(f"[{name}] publish_dt_utc parse failures={int(dt.isna().sum())}")

    out_of_uni = sorted(set(secu.tolist()) - set(universe_set))
    if out_of_uni:
        raise ContractError(f"[{name}] out-of-universe SecuCodes={out_of_uni[:20]}")

    return ContractReport(name=name, rows=int(len(df)), unique_tickers=int(secu.nunique()), unique_ann_id=int(ann_id.nunique()))


def validate_text_contract(text_path: Path, universe_set: set[str]) -> ContractReport:
    name = "ann_text"
    df = _load_table(text_path)
    _ensure_columns(
        df,
        ["ann_id", "SecuCode", "publish_dt_utc", "clean_title", "clean_body", "text_len", "parse_status"],
        name,
    )

    secu = _normalize_secu_col(df, "SecuCode", name)
    ann_id = df["ann_id"].astype(str).str.strip()
    if bool((ann_id == "").any()):
        raise ContractError(f"[{name}] ann_id has empty values")
    if bool(ann_id.duplicated().any()):
        raise ContractError(f"[{name}] ann_id must be unique; duplicates={int(ann_id.duplicated().sum())}")

    dt = pd.to_datetime(df["publish_dt_utc"], errors="coerce", utc=True)
    if bool(dt.isna().any()):
        raise ContractError(f"[{name}] publish_dt_utc parse failures={int(dt.isna().sum())}")

    text_len = pd.to_numeric(df["text_len"], errors="coerce")
    if bool(text_len.isna().any()):
        raise ContractError(f"[{name}] text_len non-numeric rows={int(text_len.isna().sum())}")
    if bool((text_len < 0).any()):
        raise ContractError(f"[{name}] text_len < 0 rows={int((text_len < 0).sum())}")

    allowed = {"ok", "empty_text", "parse_error", "missing_file"}
    bad_status = sorted(set(df["parse_status"].astype(str).tolist()) - allowed)
    if bad_status:
        raise ContractError(f"[{name}] invalid parse_status values={bad_status}")

    out_of_uni = sorted(set(secu.tolist()) - set(universe_set))
    if out_of_uni:
        raise ContractError(f"[{name}] out-of-universe SecuCodes={out_of_uni[:20]}")

    return ContractReport(name=name, rows=int(len(df)), unique_tickers=int(secu.nunique()), unique_ann_id=int(ann_id.nunique()))


def validate_dataset_contract(dataset_path: Path, universe_set: set[str]) -> ContractReport:
    name = "sentiment_dataset"
    df = _load_table(dataset_path)
    _ensure_columns(
        df,
        [
            "ann_id",
            "SecuCode",
            "publish_dt_utc",
            "t_event_bar",
            "text",
            "text_len_chars",
            "was_truncated",
            "text_len",
            "parse_status",
            "split",
        ],
        name,
    )

    secu = _normalize_secu_col(df, "SecuCode", name)
    ann_id = df["ann_id"].astype(str).str.strip()
    if bool((ann_id == "").any()):
        raise ContractError(f"[{name}] ann_id has empty values")
    if bool(ann_id.duplicated().any()):
        raise ContractError(f"[{name}] ann_id must be unique; duplicates={int(ann_id.duplicated().sum())}")

    dt_pub = pd.to_datetime(df["publish_dt_utc"], errors="coerce", utc=True)
    if bool(dt_pub.isna().any()):
        raise ContractError(f"[{name}] publish_dt_utc parse failures={int(dt_pub.isna().sum())}")

    dt_bar = pd.to_datetime(df["t_event_bar"], errors="coerce", utc=True)
    if bool(dt_bar.isna().any()):
        raise ContractError(f"[{name}] t_event_bar parse failures={int(dt_bar.isna().sum())}")

    text_len = pd.to_numeric(df["text_len"], errors="coerce")
    if bool(text_len.isna().any()):
        raise ContractError(f"[{name}] text_len non-numeric rows={int(text_len.isna().sum())}")
    if bool((text_len < 0).any()):
        raise ContractError(f"[{name}] text_len < 0 rows={int((text_len < 0).sum())}")
    text_len_chars = pd.to_numeric(df["text_len_chars"], errors="coerce")
    if bool(text_len_chars.isna().any()):
        raise ContractError(f"[{name}] text_len_chars non-numeric rows={int(text_len_chars.isna().sum())}")
    if bool((text_len_chars < 0).any()):
        raise ContractError(f"[{name}] text_len_chars < 0 rows={int((text_len_chars < 0).sum())}")

    bad_trunc = ~df["was_truncated"].map(lambda x: isinstance(x, (bool, int)))
    if bool(bad_trunc.any()):
        raise ContractError(f"[{name}] was_truncated must be boolean-like; bad_rows={int(bad_trunc.sum())}")

    allowed_status = {"ok", "empty_text", "parse_error", "missing_file"}
    bad_status = sorted(set(df["parse_status"].astype(str).tolist()) - allowed_status)
    if bad_status:
        raise ContractError(f"[{name}] invalid parse_status values={bad_status}")

    allowed_split = {"train", "val", "test"}
    bad_split = sorted(set(df["split"].astype(str).tolist()) - allowed_split)
    if bad_split:
        raise ContractError(f"[{name}] invalid split values={bad_split}")

    out_of_uni = sorted(set(secu.tolist()) - set(universe_set))
    if out_of_uni:
        raise ContractError(f"[{name}] out-of-universe SecuCodes={out_of_uni[:20]}")

    return ContractReport(name=name, rows=int(len(df)), unique_tickers=int(secu.nunique()), unique_ann_id=int(ann_id.nunique()))


def run_all_contracts(
    universe_csv: Path,
    meta_path: Path,
    text_path: Path,
    dataset_path: Path,
) -> list[ContractReport]:
    validate_universe_contract(universe_csv)
    universe_set = _load_universe_set(universe_csv)
    reports = [
        validate_universe_contract(universe_csv),
        validate_meta_contract(meta_path, universe_set),
        validate_text_contract(text_path, universe_set),
        validate_dataset_contract(dataset_path, universe_set),
    ]
    return reports
