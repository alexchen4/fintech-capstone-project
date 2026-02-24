"""
This module handles ticker universe loading and CNINFO metadata ingestion scaffolding.
It sits at the data-ingestion entry point of the sentiment pipeline.
Current scraping is a placeholder with explicit TODOs for endpoint integration.
Status: experimental MVP scaffold, not production-ready for full historical coverage.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pandas as pd

EXPECTED_METADATA_COLUMNS = [
    "SecuCode",
    "publish_dt",
    "title",
    "category",
    "pdf_url",
    "announcement_id",
]


def load_ticker_universe(path: str) -> List[str]:
    """Load ticker universe from a text/csv/parquet file.

    Supported inputs:
    - .txt: one ticker per line
    - .csv: first column or a column named one of: ticker, secucode, SecuCode
    - .parquet: same column rules as CSV
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Ticker universe file not found: {in_path}")

    suffix = in_path.suffix.lower()
    if suffix == ".txt":
        tickers = [line.strip() for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return tickers

    if suffix == ".csv":
        df = pd.read_csv(in_path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(in_path)
    else:
        raise ValueError(f"Unsupported ticker universe file extension: {suffix}")

    candidate_cols = ["ticker", "secucode", "SecuCode"]
    col = next((c for c in candidate_cols if c in df.columns), df.columns[0] if len(df.columns) else None)
    if col is None:
        return []

    return [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]


def scrape_cninfo_metadata(
    tickers: List[str],
    start: str,
    end: str,
    out_path: str,
    rate_limit_s: float = 1.0,
) -> pd.DataFrame:
    """Placeholder CNINFO metadata scraper.

    TODO:
    - Replace placeholder logic with verified CNINFO endpoint integration.
    - Implement request retries/backoff, pagination, and robust schema validation.

    Expected output schema columns:
    ['SecuCode','publish_dt','title','category','pdf_url','announcement_id']
    """
    rows = []
    for ticker in tickers:
        # Placeholder row keeps schema stable for downstream testing.
        rows.append(
            {
                "SecuCode": str(ticker),
                "publish_dt": pd.NaT,
                "title": None,
                "category": None,
                "pdf_url": None,
                "announcement_id": None,
            }
        )
        if rate_limit_s > 0:
            time.sleep(rate_limit_s)

    df = pd.DataFrame(rows, columns=EXPECTED_METADATA_COLUMNS)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

    return df
