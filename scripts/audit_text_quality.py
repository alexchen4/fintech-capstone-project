#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.secu import normalize_secu_series
from common.universe import validate_universe

DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "universe.csv"
DEFAULT_AUDIT_DIR = ROOT / "data" / "processed" / "sentiment" / "audits"


def _infer_month_tag(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2})", path.name)
    return m.group(1) if m else "unknown"


def _load_universe(universe_csv: Path) -> set[str]:
    uni = pd.read_csv(universe_csv)
    col = "SecuCode" if "SecuCode" in uni.columns else uni.columns[0]
    vals = normalize_secu_series(uni[col]).dropna().astype(str)
    return set(vals.tolist())


def _split_model_text(text: str) -> tuple[str, str]:
    s = "" if text is None else str(text)
    m = re.match(r"^\[CLS\]\s*(.*?)\s*\[SEP\]\s*(.*?)\s*\[SEP\]\s*$", s, flags=re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", s.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit text quality/readiness for ann_text or sentiment_dataset artifacts.")
    p.add_argument("--input_path", type=Path, required=True, help="Input parquet path (ann_text or sentiment_dataset)")
    p.add_argument("--universe_csv", type=Path, default=DEFAULT_UNIVERSE)
    p.add_argument("--sample_n", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview_chars", type=int, default=300)
    p.add_argument("--out_csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"input_path not found: {args.input_path}")
    if not args.universe_csv.exists():
        raise FileNotFoundError(f"universe_csv not found: {args.universe_csv}")

    month = _infer_month_tag(args.input_path)
    out_csv = args.out_csv or (DEFAULT_AUDIT_DIR / f"text_sample_{month}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_path)
    req = {"ann_id", "SecuCode", "publish_dt_utc", "parse_status"}
    missing = sorted(req - set(df.columns))
    if missing:
        raise ValueError(f"input missing required columns: {missing}")

    df = df.copy()
    df["SecuCode"] = normalize_secu_series(df["SecuCode"])
    universe_set = _load_universe(args.universe_csv)
    validate_universe(df, universe_set, col="SecuCode")

    if "text" in df.columns:
        text = df["text"].fillna("").astype(str)
        if "text_len_chars" in df.columns:
            text_len = pd.to_numeric(df["text_len_chars"], errors="coerce")
        elif "text_len" in df.columns:
            text_len = pd.to_numeric(df["text_len"], errors="coerce")
        else:
            text_len = text.str.len()

        split = text.map(_split_model_text)
        df["title_preview"] = split.map(lambda x: x[0][: args.preview_chars])
        df["body_preview"] = split.map(lambda x: x[1][: args.preview_chars])
        effective_text = text
    else:
        _need = {"clean_title", "clean_body"}
        miss2 = sorted(_need - set(df.columns))
        if miss2:
            raise ValueError(f"ann_text input missing columns: {miss2}")
        title = df["clean_title"].fillna("").astype(str)
        body = df["clean_body"].fillna("").astype(str)
        df["title_preview"] = title.str.slice(0, args.preview_chars)
        df["body_preview"] = body.str.slice(0, args.preview_chars)
        effective_text = "[CLS] " + title + " [SEP] " + body + " [SEP]"
        text_len = pd.to_numeric(df["text_len"], errors="coerce") if "text_len" in df.columns else effective_text.str.len()

    df["text_len"] = text_len.fillna(0).astype(int)
    df["publish_dt_utc"] = pd.to_datetime(df["publish_dt_utc"], errors="coerce", utc=True)

    total = len(df)
    parse_ok = int((df["parse_status"].astype(str) == "ok").sum())
    empty = int((df["text_len"] == 0).sum())

    q = df["text_len"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict() if total else {0.5: 0, 0.9: 0, 0.95: 0, 0.99: 0}

    dup_mask = effective_text.duplicated(keep=False)
    dup_rate = 100.0 * int(dup_mask.sum()) / total if total else 0.0

    sample_n = min(int(args.sample_n), total)
    sample = df.sample(n=sample_n, random_state=int(args.seed), replace=False) if sample_n else df.head(0)
    sample = sample.sort_values(["publish_dt_utc", "ann_id"], kind="stable")
    out_cols = [
        "ann_id",
        "SecuCode",
        "publish_dt_utc",
        "parse_status",
        "text_len",
        "title_preview",
        "body_preview",
    ]
    sample[out_cols].to_csv(out_csv, index=False)

    print("[audit_text_quality] summary")
    print(f"input_path={args.input_path}")
    print(f"universe_csv={args.universe_csv}")
    print(f"out_csv={out_csv}")
    print(f"rows={total}")
    print(f"unique_SecuCode={int(df['SecuCode'].nunique()) if total else 0}")
    print(f"parse_ok_rate={round(100.0 * parse_ok / total, 2) if total else 0.0}")
    print(f"empty_text_rate={round(100.0 * empty / total, 2) if total else 0.0}")
    print(f"text_len_p50={round(float(q.get(0.5, 0)), 2)}")
    print(f"text_len_p90={round(float(q.get(0.9, 0)), 2)}")
    print(f"text_len_p95={round(float(q.get(0.95, 0)), 2)}")
    print(f"text_len_p99={round(float(q.get(0.99, 0)), 2)}")
    print(f"duplicate_text_rate={round(dup_rate, 2)}")
    print(f"sample_rows_written={sample_n}")


if __name__ == "__main__":
    main()
