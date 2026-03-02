#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "interim" / "mapping_candidates.csv"
OUT_JSON = ROOT / "data" / "interim" / "mapping_summary.json"


def normalize_secu_code(x: object) -> str:
    s = str(x or "").strip()
    num = pd.to_numeric(pd.Series([s]), errors="coerce").iloc[0]
    if pd.notna(num):
        return str(int(num)).zfill(6)
    m = re.search(r"(\d+)", s)
    return m.group(1).zfill(6) if m else ""


def detect_ann_code_col(df: pd.DataFrame) -> str:
    cols = {c.lower(): c for c in df.columns}
    for k in ["secucode", "secu_code", "ticker", "symbol"]:
        if k in cols:
            return cols[k]
    raise ValueError("ann_meta_csv missing a code column (expected one of SecuCode/ticker/secu_code/symbol)")


def detect_bars_code_col(cols: list[str]) -> str:
    low = {c.lower(): c for c in cols}
    for k in ["secucode", "secu_code", "ticker", "symbol"]:
        if k in low:
            return low[k]
    raise ValueError("bars_path missing a code column (expected one of SecuCode/secu_code/ticker/symbol)")


def load_ann_codes(path: Path) -> list[str]:
    ann = pd.read_csv(path)
    col = detect_ann_code_col(ann)
    codes = ann[col].map(normalize_secu_code)
    codes = sorted({c for c in codes.tolist() if c})
    return codes


def load_bars_raw_codes(path: Path, chunksize: int = 300000) -> list[str]:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        col = detect_bars_code_col(list(df.columns))
        raw = df[col].astype(str).str.strip()
        return sorted({x for x in raw.tolist() if x and x.lower() != "nan"})

    header = pd.read_csv(path, nrows=0)
    col = detect_bars_code_col(list(header.columns))
    seen: set[str] = set()
    for ch in pd.read_csv(path, usecols=[col], chunksize=chunksize):
        vals = ch[col].astype(str).str.strip()
        for v in vals.tolist():
            if v and v.lower() != "nan":
                seen.add(v)
    return sorted(seen)


def build_candidates(ann_codes: list[str], bars_raw_codes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    norm_to_raw: dict[str, list[str]] = {}
    for raw in bars_raw_codes:
        n = normalize_secu_code(raw)
        if n:
            norm_to_raw.setdefault(n, []).append(raw)

    unmatched: list[str] = []
    for ann in ann_codes:
        raws = norm_to_raw.get(ann, [])
        if raws:
            for raw in sorted(set(raws)):
                rows.append(
                    {
                        "cninfo_SecuCode": ann,
                        "bars_SecuCode_raw_candidate": raw,
                        "match_type": "exact_norm",
                        "confidence_score": 1.0,
                        "evidence": f"normalize({raw})={ann}",
                    }
                )
        else:
            unmatched.append(ann)

    # Heuristic candidates for unmatched codes (review only; confidence <= 0.6).
    for ann in unmatched:
        scored: list[tuple[float, str, str]] = []
        ann_nz = ann.lstrip("0") or ann
        for raw in bars_raw_codes:
            raw_norm = normalize_secu_code(raw)
            raw_digits = re.search(r"(\d+)", raw)
            raw_digits_s = raw_digits.group(1) if raw_digits else ""
            raw_nz = (raw_digits_s.lstrip("0") if raw_digits_s else "") or raw_digits_s

            match_type = ""
            score = 0.0
            evidence = ""
            if ann in raw or (ann_nz and ann_nz in raw):
                match_type = "substring"
                score = 0.55
                evidence = f"ann fragment found in raw: {raw}"
            else:
                sim = SequenceMatcher(None, ann, raw_norm or raw).ratio()
                if sim >= 0.67:
                    match_type = "fuzzy"
                    score = min(0.6, round(sim * 0.6, 3))
                    evidence = f"similarity={sim:.3f} raw_norm={raw_norm or 'NA'}"
                elif ann_nz and raw_nz and ann_nz == raw_nz:
                    match_type = "strip0"
                    score = 0.6
                    evidence = f"strip0 match ann={ann_nz} raw={raw_nz}"

            if match_type:
                scored.append((score, raw, f"{match_type}|{evidence}"))

        scored = sorted(scored, key=lambda x: (-x[0], x[1]))[:5]
        for sc, raw, meta in scored:
            mtype, ev = meta.split("|", 1)
            rows.append(
                {
                    "cninfo_SecuCode": ann,
                    "bars_SecuCode_raw_candidate": raw,
                    "match_type": mtype,
                    "confidence_score": float(sc),
                    "evidence": f"{ev}; needs_human_review",
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "cninfo_SecuCode",
            "bars_SecuCode_raw_candidate",
            "match_type",
            "confidence_score",
            "evidence",
        ],
    )
    return out.sort_values(["cninfo_SecuCode", "confidence_score"], ascending=[True, False]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build candidate SecuCode mapping between CNINFO and bars universe.")
    parser.add_argument("--ann_meta_csv", type=Path, required=True, help="Path to CNINFO announcements meta CSV.")
    parser.add_argument("--bars_path", type=Path, required=True, help="Path to bars file (csv/parquet).")
    args = parser.parse_args()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    ann_codes = load_ann_codes(args.ann_meta_csv)
    bars_raw = load_bars_raw_codes(args.bars_path)
    cand = build_candidates(ann_codes, bars_raw)
    cand.to_csv(OUT_CSV, index=False)

    exact_codes = set(cand.loc[cand["match_type"] == "exact_norm", "cninfo_SecuCode"].tolist())
    unmatched = sorted(set(ann_codes) - exact_codes)
    summary = {
        "ann_asset_count": len(ann_codes),
        "bars_raw_id_count": len(bars_raw),
        "candidate_rows": int(len(cand)),
        "match_type_counts": cand["match_type"].value_counts(dropna=False).to_dict() if len(cand) else {},
        "exact_norm_count": len(exact_codes),
        "unmatched_after_exact_count": len(unmatched),
        "unmatched_after_exact_top20": unmatched[:20],
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"wrote: {OUT_CSV}")
    print(f"wrote: {OUT_JSON}")
    print(
        "Next step: manually curate a verified code map CSV with columns "
        "[cninfo_SecuCode,bars_SecuCode_raw]. Do NOT auto-apply heuristic candidates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
