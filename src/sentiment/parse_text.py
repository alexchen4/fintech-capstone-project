"""CNINFO payload parsing to clean model-ready text rows."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from cninfo.pdf_text import extract_text_from_pdf

OUTPUT_COLUMNS = [
    "ann_id",
    "SecuCode",
    "publish_dt_utc",
    "clean_title",
    "clean_body",
    "text_len",
    "parse_status",
]


HEADER_FOOTER_PATTERNS = [
    re.compile(r"^证券代码[：:].*$"),
    re.compile(r"^证券简称[：:].*$"),
    re.compile(r"^公告编号[：:].*$"),
    re.compile(r"^公司代码[：:].*$"),
    re.compile(r"^公司简称[：:].*$"),
    re.compile(r"^第\s*\d+\s*页(?:\s*/\s*共\s*\d+\s*页)?$"),
    re.compile(r"^\d{4}年\d{1,2}月\d{1,2}日$"),
]


def _normalize_whitespace(text: object) -> str:
    s = "" if pd.isna(text) else str(text)
    s = s.replace("\u3000", " ").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\t\f\v]+", " ", s)
    s = re.sub(r" +", " ", s)
    return s.strip()


def _drop_boilerplate_lines(lines: list[str]) -> list[str]:
    # Remove obvious CNINFO/statement headers and page markers.
    cleaned: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if "巨潮资讯网" in s or "www.cninfo.com.cn" in s:
            continue
        if "本公司及董事会" in s and "信息披露" in s:
            continue
        if any(p.match(s) for p in HEADER_FOOTER_PATTERNS):
            continue
        cleaned.append(s)

    if not cleaned:
        return cleaned

    # Remove lines repeated many times in a single document (common page header/footer noise).
    freq = Counter(cleaned)
    cleaned = [x for x in cleaned if freq[x] <= 3]

    # Drop consecutive duplicates.
    dedup: list[str] = []
    prev = None
    for x in cleaned:
        if x == prev:
            continue
        dedup.append(x)
        prev = x
    return dedup


def clean_body_text(raw_text: str) -> str:
    base = _normalize_whitespace(raw_text)
    if not base:
        return ""
    lines = [ln.strip() for ln in base.split("\n")]
    lines = _drop_boilerplate_lines(lines)
    return "\n".join(lines).strip()


def parse_cninfo_text(meta_parquet: Path, pdf_dir: Path) -> pd.DataFrame:
    meta = pd.read_parquet(meta_parquet)
    need = {"ann_id", "SecuCode", "publish_dt_utc", "title"}
    missing = sorted(need - set(meta.columns))
    if missing:
        raise ValueError(f"meta parquet missing required columns: {missing}")

    base = meta[["ann_id", "SecuCode", "publish_dt_utc", "title"]].copy()
    base["ann_id"] = base["ann_id"].astype(str).str.strip()
    base["SecuCode"] = base["SecuCode"].astype(str).str.strip()
    base["clean_title"] = base["title"].map(_normalize_whitespace)

    rows: list[dict[str, object]] = []
    for r in base.itertuples(index=False):
        ann_id = str(r.ann_id)
        path = pdf_dir / f"{ann_id}.pdf"

        body = ""
        status = "ok"

        if not path.exists():
            status = "missing_file"
        else:
            extracted = extract_text_from_pdf(str(path))
            raw_text = (extracted.get("text") or "").strip()
            if raw_text:
                body = clean_body_text(raw_text)
                status = "ok" if body else "empty_text"
            else:
                err = str(extracted.get("error") or "")
                status = "parse_error" if err else "empty_text"

        rows.append(
            {
                "ann_id": ann_id,
                "SecuCode": str(r.SecuCode),
                "publish_dt_utc": r.publish_dt_utc,
                "clean_title": str(r.clean_title),
                "clean_body": body,
                "text_len": int(len(body)),
                "parse_status": status,
            }
        )

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out = out.sort_values(["publish_dt_utc", "ann_id"], kind="stable").reset_index(drop=True)
    return out


def parse_summary(df: pd.DataFrame) -> dict[str, object]:
    total = int(len(df))
    if total == 0:
        return {"rows": 0, "pct_ok": 0.0, "pct_empty": 0.0, "median_text_len": 0.0}

    ok = int((df["parse_status"] == "ok").sum())
    empty = int((df["text_len"] == 0).sum())
    median_len = float(df["text_len"].median()) if not df.empty else 0.0

    return {
        "rows": total,
        "pct_ok": round(100.0 * ok / total, 2),
        "pct_empty": round(100.0 * empty / total, 2),
        "median_text_len": round(median_len, 2),
    }
