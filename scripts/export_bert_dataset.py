#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "bert"
REQUIRED_OUT_COLS = [
    "ann_id",
    "SecuCode",
    "publish_dt_utc",
    "t_event_bar",
    "y",
    "ret_fwd_8",
    "split",
    "text_clean",
    "text_len_chars",
    "text_hash",
    "source_month",
    "was_truncated_export",
    "text_len_chars_raw",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export deterministic BERT-ready datasets from labeled sentiment parquets.")
    p.add_argument("--months", required=True, help='Comma-separated months, e.g. "2024-06,2020-03"')
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--min_chars", type=int, default=200)
    p.add_argument("--max_chars", type=int, default=6000)
    p.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop_parse_errors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--self_check", action="store_true", default=False)
    return p.parse_args()


def _normalize_months(raw: str) -> list[str]:
    months = [m.strip() for m in raw.split(",") if m.strip()]
    if not months:
        raise ValueError("--months must contain at least one month")
    return months


def _normalize_secu(x: object) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return digits[-6:].zfill(6)
    return s.zfill(6)


def _clean_text(s: str) -> str:
    t = s
    t = re.sub(r"^\s*\[CLS\]\s*", "", t)
    t = t.replace("[SEP]", "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[\x00-\x09\x0B-\x1F\x7F]", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    lines = [line.strip() for line in t.split("\n")]
    t = "\n".join(lines)
    return t.strip()


def _make_time_split(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    if n == 0:
        return pd.Series([], dtype="object")
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    out = pd.Series(["test"] * n, index=df.index, dtype="object")
    out.iloc[:train_end] = "train"
    out.iloc[train_end:val_end] = "val"
    return out


def _sha256_utf8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_month(month: str) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "sentiment" / f"sentiment_labeled_{month}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing labeled parquet for month={month}: {path}")
    return pd.read_parquet(path)


def _export_month(
    month: str,
    min_chars: int,
    max_chars: int,
    dedupe: bool,
    drop_parse_errors: bool,
    out_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    raw = _load_month(month)
    before_rows = len(raw)
    parse_counts = raw["parse_status"].value_counts(dropna=False).to_dict() if "parse_status" in raw.columns else {}

    df = raw.copy()
    if drop_parse_errors and "parse_status" in df.columns:
        df = df[df["parse_status"].astype(str) == "ok"].copy()

    text = df.get("text", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    non_empty = text.str.strip() != ""
    df = df[non_empty].copy()
    text = df["text"].fillna("").astype(str)

    text_len_raw = text.str.len()
    if "text_len_chars" not in df.columns:
        df["text_len_chars"] = text_len_raw
    else:
        missing_len = df["text_len_chars"].isna()
        if missing_len.any():
            df.loc[missing_len, "text_len_chars"] = text_len_raw[missing_len]

    df = df[text_len_raw >= min_chars].copy()
    text = df["text"].fillna("").astype(str)
    text_len_raw = text.str.len()

    df["text_len_chars_raw"] = text_len_raw.astype(int)
    df["was_truncated_export"] = text_len_raw > int(max_chars)
    text_trunc = text.str.slice(0, int(max_chars))

    df["text_clean"] = text_trunc.map(_clean_text)
    df = df[df["text_clean"].str.strip() != ""].copy()
    df["text_len_chars"] = df["text_clean"].str.len().astype(int)
    df = df[df["text_len_chars"] >= int(min_chars)].copy()

    dt = pd.to_datetime(df["publish_dt_utc"], errors="coerce", utc=True)
    df["_publish_dt"] = dt
    df = df.sort_values(["_publish_dt", "ann_id"], kind="stable", na_position="last").copy()

    df["text_hash"] = df["text_clean"].map(_sha256_utf8)
    dup_dropped = 0
    if dedupe:
        before_dedupe = len(df)
        df = df.drop_duplicates(subset=["text_hash"], keep="first").copy()
        dup_dropped = before_dedupe - len(df)

    if "split" in df.columns:
        df["split"] = df["split"].astype(str)
    else:
        df["split"] = _make_time_split(df)

    df["SecuCode"] = df["SecuCode"].map(_normalize_secu)
    df["source_month"] = month

    out = df[REQUIRED_OUT_COLS].copy()
    out = out.sort_values(["publish_dt_utc", "ann_id"], kind="stable", na_position="last").reset_index(drop=True)

    y_balance = out["y"].value_counts(dropna=False).sort_index().to_dict()
    split_counts = out["split"].value_counts(dropna=False).to_dict()
    if len(out) > 0:
        p50 = float(out["text_len_chars"].quantile(0.50))
        p95 = float(out["text_len_chars"].quantile(0.95))
    else:
        p50 = 0.0
        p95 = 0.0

    print(f"[export_bert] month={month} rows_before={before_rows} rows_after={len(out)}")
    print(f"[export_bert] month={month} parse_status_before={parse_counts}")
    print(f"[export_bert] month={month} y_balance_after={y_balance}")
    print(f"[export_bert] month={month} split_counts={split_counts}")
    print(f"[export_bert] month={month} duplicates_dropped={dup_dropped}")
    print(f"[export_bert] month={month} text_len_chars_p50={p50:.1f} p95={p95:.1f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bert_announcements_{month}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[export_bert] wrote {out_path} rows={len(out)}")
    return out, out_path


def _self_check_one(path: Path, min_chars: int) -> None:
    df = pd.read_parquet(path)

    if df["y"].isna().any() or (~df["y"].isin([-1, 0, 1])).any():
        raise AssertionError(f"{path}: y must be non-null and in {{-1,0,1}}")
    if df["ret_fwd_8"].isna().any():
        raise AssertionError(f"{path}: ret_fwd_8 contains nulls")
    if df["split"].isna().any() or (~df["split"].isin(["train", "val", "test"])).any():
        raise AssertionError(f"{path}: split must be in {{train,val,test}}")

    text = df["text_clean"].fillna("").astype(str)
    text_len = text.str.len()
    if (text.str.strip() == "").any() or (text_len < int(min_chars)).any():
        raise AssertionError(f"{path}: text_clean must be non-empty and >= min_chars")

    sec = df["SecuCode"].fillna("").astype(str)
    if (~sec.str.fullmatch(r"\d{6}")).any():
        raise AssertionError(f"{path}: SecuCode must be 6-digit strings")

    dt = pd.to_datetime(df["publish_dt_utc"], errors="coerce", utc=True)
    if dt.isna().any():
        raise AssertionError(f"{path}: publish_dt_utc contains unparsable values")

    print(f"[self_check] {path}")
    for split_name in ["train", "val", "test"]:
        mask = df["split"] == split_name
        if not mask.any():
            print(f"[self_check] split={split_name} rows=0 earliest=None latest=None")
            continue
        split_dt = dt[mask]
        print(
            f"[self_check] split={split_name} rows={int(mask.sum())} "
            f"earliest={split_dt.min()} latest={split_dt.max()}"
        )


def main() -> int:
    args = parse_args()
    months = _normalize_months(args.months)

    month_dfs: list[pd.DataFrame] = []
    month_paths: list[Path] = []
    for month in months:
        out_df, out_path = _export_month(
            month=month,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            dedupe=args.dedupe,
            drop_parse_errors=args.drop_parse_errors,
            out_dir=args.out_dir,
        )
        month_dfs.append(out_df)
        month_paths.append(out_path)

    all_df = pd.concat(month_dfs, ignore_index=True)
    all_df["_publish_dt"] = pd.to_datetime(all_df["publish_dt_utc"], errors="coerce", utc=True)
    all_df = all_df.sort_values(["_publish_dt", "ann_id"], kind="stable", na_position="last").copy()

    before_collision = len(all_df)
    all_df = all_df.drop_duplicates(subset=["ann_id"], keep="first").copy()
    collision_dropped = before_collision - len(all_df)

    out_all = all_df[REQUIRED_OUT_COLS].reset_index(drop=True)
    out_all_path = args.out_dir / "bert_announcements_all.parquet"
    out_all.to_parquet(out_all_path, index=False)
    print(f"[export_bert] combined_rows={len(out_all)} ann_id_collisions_dropped={collision_dropped}")
    print(f"[export_bert] wrote {out_all_path} rows={len(out_all)}")

    if args.self_check:
        for p in month_paths + [out_all_path]:
            _self_check_one(p, min_chars=args.min_chars)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
