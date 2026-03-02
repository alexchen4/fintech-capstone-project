"""CNINFO payload downloader (resumable, strict-universe)."""

from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
import requests

from common.secu import normalize_secu_series
from common.universe import validate_universe


def load_universe(universe_csv: Path) -> set[str]:
    uni = pd.read_csv(universe_csv)
    col = "SecuCode" if "SecuCode" in uni.columns else uni.columns[0]
    vals = normalize_secu_series(uni[col]).dropna().astype(str).str.strip()
    return set(vals.tolist())


def infer_month_tag(meta_parquet: Path) -> str:
    name = meta_parquet.name
    for i in range(len(name) - 6):
        s = name[i : i + 7]
        if len(s) == 7 and s[4] == "-" and s[:4].isdigit() and s[5:].isdigit():
            return s
    return "unknown"


def fetch_payloads(
    meta_parquet: Path,
    out_dir: Path,
    universe_csv: Path,
    failures_csv: Path,
    limit: int | None = None,
    sleep_sec: float = 0.25,
    jitter_sec: float = 0.15,
    timeout_sec: float = 30.0,
) -> dict[str, int]:
    meta = pd.read_parquet(meta_parquet)
    required = {"ann_id", "SecuCode", "detail_url"}
    miss = sorted(required - set(meta.columns))
    if miss:
        raise ValueError(f"meta parquet missing columns: {miss}")

    universe_set = load_universe(universe_csv)
    validate_universe(meta, universe_set, col="SecuCode")

    work = meta[["ann_id", "SecuCode", "detail_url"]].copy()
    work["ann_id"] = work["ann_id"].astype(str).str.strip()
    work["detail_url"] = work["detail_url"].fillna("").astype(str).str.strip()
    work = work[work["ann_id"].str.fullmatch(r"\d+")].drop_duplicates("ann_id", keep="last")
    if limit is not None:
        work = work.head(int(limit)).copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    failures_csv.parent.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, str]] = []
    total = int(len(work))
    downloaded = 0
    skipped_existing = 0

    session = requests.Session()
    session.headers.update({"User-Agent": "fintech-capstone-pipeline/phase3"})

    for row in work.itertuples(index=False):
        ann_id = str(row.ann_id)
        url = str(row.detail_url)
        out_path = out_dir / f"{ann_id}.pdf"

        if out_path.exists() and out_path.stat().st_size > 0:
            skipped_existing += 1
            continue

        if not url:
            failures.append({"ann_id": ann_id, "reason": "missing_detail_url"})
            continue

        try:
            resp = session.get(url, timeout=timeout_sec)
            if resp.status_code != 200:
                failures.append({"ann_id": ann_id, "reason": f"http_{resp.status_code}"})
            elif not resp.content:
                failures.append({"ann_id": ann_id, "reason": "empty_content"})
            else:
                out_path.write_bytes(resp.content)
                downloaded += 1
        except requests.RequestException as exc:
            failures.append({"ann_id": ann_id, "reason": f"request_error:{type(exc).__name__}"})

        # Polite rate limit with positive jitter.
        sleep_for = max(0.0, float(sleep_sec) + random.uniform(0.0, float(jitter_sec)))
        time.sleep(sleep_for)

    if failures:
        pd.DataFrame(failures).to_csv(failures_csv, index=False)
    elif failures_csv.exists():
        failures_csv.unlink()

    return {
        "total": total,
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "failed": len(failures),
    }
