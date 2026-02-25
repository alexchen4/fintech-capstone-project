#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_META = ROOT / "data" / "raw" / "meta" / "announcements_meta_2025-12.csv"
DEFAULT_FIXTURES = ROOT / "data" / "raw" / "fixtures" / "pdfs"
DEFAULT_BARS_CALENDAR = ROOT / "data" / "raw" / "bars" / "calendar_15m.csv"

META_OUT = ROOT / "data" / "interim" / "meta_mvp.csv"
TEXTS_OUT_CSV = ROOT / "data" / "interim" / "texts_mvp.csv"
TEXTS_OUT_PARQUET = ROOT / "data" / "interim" / "texts_mvp.parquet"
FAIL_REPORT_OUT = ROOT / "data" / "interim" / "extraction_failures_report.csv"
SENT_OUT_CSV = ROOT / "data" / "processed" / "sentiment_mvp.csv"
SENT_OUT_PARQUET = ROOT / "data" / "processed" / "sentiment_mvp.parquet"

REQUIRED_META_COLUMNS = {"ticker", "publish_ts", "title", "pdf_url"}
TEXTS_COLUMNS = [
    "ann_id",
    "SecuCode",
    "publish_dt_utc",
    "publish_dt_local",
    "title",
    "pdf_path",
    "text",
    "text_len",
    "extraction_ok",
    "extraction_method",
    "extraction_error",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_ann_id(url: str) -> str:
    if not isinstance(url, str):
        return ""
    match = re.search(r"/(\d+)\.pdf$", url, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fail(msg: str) -> int:
    print(msg)
    return 1


def _append_error(base_error: str, extra_error: str) -> str:
    base = str(base_error or "").strip()
    extra = str(extra_error or "").strip()
    if base and extra:
        return f"{base} | {extra}"
    return base or extra


def _bool_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def _safe_parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _utc_iso_strings(ts: pd.Series) -> pd.Series:
    return ts.map(lambda x: x.isoformat() if pd.notna(x) else "")


def _local_iso_strings(ts: pd.Series) -> tuple[pd.Series, str]:
    try:
        from zoneinfo import ZoneInfo  # py>=3.9 stdlib

        sh_tz = ZoneInfo("Asia/Shanghai")
        local = ts.dt.tz_convert(sh_tz)
        return local.map(lambda x: x.isoformat() if pd.notna(x) else ""), ""
    except Exception as exc:
        return pd.Series([""] * len(ts), index=ts.index, dtype="string"), str(exc)


def _list_fixture_ann_ids(fixtures_dir: Path) -> set[str]:
    if not fixtures_dir.exists():
        return set()
    return {p.stem for p in fixtures_dir.glob("*.pdf") if re.fullmatch(r"\d+", p.stem)}


def build_meta_subset(
    meta_csv: Path,
    fixtures_dir: Path,
    secu_code: str,
    n: int,
    subset_mode: str,
    apply_secu_filter_in_existing: bool,
) -> pd.DataFrame:
    if not meta_csv.exists():
        raise FileNotFoundError(f"missing meta CSV: {meta_csv}")
    raw = pd.read_csv(meta_csv)
    missing = REQUIRED_META_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"meta CSV is missing required columns: {sorted(missing)}")

    df = raw.copy()
    df["ann_id"] = df["pdf_url"].map(_extract_ann_id)
    if "announcement_id" in df.columns:
        df["ann_id"] = df["announcement_id"].fillna(df["ann_id"]).astype(str).str.strip()
    df["SecuCode"] = df["ticker"].map(lambda x: str(x).split(".")[0]).str.zfill(6)
    ts_utc = _safe_parse_utc(df["publish_ts"])
    df["publish_dt_utc"] = _utc_iso_strings(ts_utc)
    local_vals, local_warn = _local_iso_strings(ts_utc)
    df["publish_dt_local"] = local_vals
    if local_warn:
        print(f"warning: failed to convert publish_dt_local to Asia/Shanghai; leaving blank ({local_warn})")

    df["title"] = df["title"].fillna("").astype(str)
    df = df[df["ann_id"].astype(str).str.fullmatch(r"\d+")].copy()

    if subset_mode == "first_n":
        df = df[df["SecuCode"] == secu_code].copy()
        df = df.head(n).copy()
    elif subset_mode == "existing_pdfs":
        fixture_ids = _list_fixture_ann_ids(fixtures_dir)
        if not fixture_ids:
            raise ValueError(f"no fixture PDFs found in {fixtures_dir}")
        df = df[df["ann_id"].isin(fixture_ids)].copy()
        if apply_secu_filter_in_existing:
            df = df[df["SecuCode"] == secu_code].copy()
        df = df.sort_values(["publish_dt_utc", "ann_id"], kind="stable").copy()
    else:
        raise ValueError(f"invalid subset_mode={subset_mode}")

    if df.empty:
        if subset_mode == "existing_pdfs":
            raise ValueError("no rows after filtering metadata to fixture ann_ids; check fixture names and meta CSV")
        raise ValueError(f"no rows after filtering for SecuCode={secu_code}; check meta CSV and selection args")
    df["ann_id"] = df["ann_id"].astype(str)
    return df[["ann_id", "SecuCode", "publish_dt_utc", "publish_dt_local", "title"]].copy()


def verify_fixture_pdfs(meta_df: pd.DataFrame, fixtures_dir: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    if not fixtures_dir.exists():
        raise FileNotFoundError(
            f"fixtures directory not found: {fixtures_dir}\n"
            "Create it and place files as <ann_id>.pdf (for example 1224882117.pdf)."
        )

    verified = meta_df.copy()
    candidate_paths = verified["ann_id"].map(lambda aid: fixtures_dir / f"{aid}.pdf")
    verified["has_pdf"] = candidate_paths.map(Path.exists).astype(bool)
    verified["pdf_path"] = candidate_paths.map(lambda p: str(p) if p.exists() else "")

    missing = verified.loc[~verified["has_pdf"], ["ann_id", "title", "publish_dt_utc"]]
    missing_rows = missing.to_dict(orient="records")
    return verified[
        ["ann_id", "SecuCode", "publish_dt_utc", "publish_dt_local", "title", "pdf_path", "has_pdf"]
    ].copy(), missing_rows


def extract_texts(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    printed_preview = 0
    for row in rows.itertuples(index=False):
        text = ""
        extraction_ok = False
        method = "pdfplumber"
        extraction_error = ""

        if not bool(row.has_pdf) or not str(row.pdf_path):
            extraction_error = "missing_pdf"
        else:
            pdf_path = Path(row.pdf_path)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    pages = [(p.extract_text() or "") for p in pdf.pages]
                text = _normalize_text(" ".join(pages))
                extraction_ok = len(text) > 0
                if not extraction_ok:
                    extraction_error = "image_only_pdf"
            except Exception as exc:
                extraction_ok = False
                text = ""
                extraction_error = f"{type(exc).__name__}: {exc}"

        if extraction_ok and printed_preview < 3:
            preview = _normalize_text(text)[:120]
            print(f"extraction preview {printed_preview + 1}: ann_id={row.ann_id} text[:120]={preview}")
            printed_preview += 1

        out_rows.append(
            {
                "ann_id": row.ann_id,
                "SecuCode": row.SecuCode,
                "publish_dt_utc": row.publish_dt_utc,
                "publish_dt_local": row.publish_dt_local,
                "title": row.title,
                "pdf_path": row.pdf_path,
                "text": text,
                "text_len": len(text),
                "extraction_ok": bool(extraction_ok),
                "extraction_method": method,
                "extraction_error": extraction_error,
            }
        )

    return pd.DataFrame(out_rows, columns=TEXTS_COLUMNS)


def _normalize_existing_texts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TEXTS_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out["ann_id"] = out["ann_id"].astype(str)
    out["SecuCode"] = out["SecuCode"].astype(str)
    out["publish_dt_utc"] = out["publish_dt_utc"].fillna("").astype(str)
    out["publish_dt_local"] = out["publish_dt_local"].fillna("").astype(str)
    out["title"] = out["title"].fillna("").astype(str)
    out["pdf_path"] = out["pdf_path"].fillna("").astype(str)
    out["text"] = out["text"].fillna("").astype(str)
    out["text_len"] = pd.to_numeric(out["text_len"], errors="coerce").fillna(0).astype("int64")
    out["extraction_ok"] = _bool_series(out["extraction_ok"]).astype(bool)
    out["extraction_method"] = out["extraction_method"].fillna("pdfplumber").astype(str)
    out["extraction_error"] = out["extraction_error"].fillna("").astype(str)
    return out[TEXTS_COLUMNS].copy()


def build_triage_report(verified_df: pd.DataFrame, existing_texts_df: pd.DataFrame | None = None) -> pd.DataFrame:
    base = verified_df.copy()
    if existing_texts_df is not None and not existing_texts_df.empty:
        enrich = _normalize_existing_texts(existing_texts_df)
        enrich = enrich[["ann_id", "text_len", "extraction_ok", "extraction_error"]].drop_duplicates("ann_id", keep="last")
        base = base.merge(enrich, on="ann_id", how="left")
    else:
        base["text_len"] = pd.NA
        base["extraction_ok"] = pd.NA
        base["extraction_error"] = ""

    report_rows = []
    for row in base.itertuples(index=False):
        file_bytes: int | None = None
        page_count: int | None = None
        has_any_text = False
        open_error = ""
        prior_error = "" if pd.isna(row.extraction_error) else str(row.extraction_error)
        current_ok = row.extraction_ok

        if bool(row.has_pdf) and str(row.pdf_path):
            pdf_path = Path(str(row.pdf_path))
            try:
                file_bytes = pdf_path.stat().st_size
                with pdfplumber.open(pdf_path) as pdf:
                    page_count = len(pdf.pages)
                    for p in pdf.pages:
                        if _normalize_text(p.extract_text() or ""):
                            has_any_text = True
                            break
            except Exception as exc:
                open_error = f"{type(exc).__name__}: {exc}"
        else:
            prior_error = _append_error(prior_error, "missing_pdf")

        merged_error = _append_error(prior_error, open_error)
        if pd.isna(current_ok):
            current_ok = bool(has_any_text and not merged_error)

        report_rows.append(
            {
                "ann_id": str(row.ann_id),
                "SecuCode": str(row.SecuCode),
                "title": str(row.title),
                "publish_dt_utc": str(row.publish_dt_utc),
                "pdf_path": str(row.pdf_path),
                "has_pdf": bool(row.has_pdf),
                "file_bytes": file_bytes,
                "page_count": page_count,
                "has_any_text": bool(has_any_text),
                "extraction_ok": bool(current_ok),
                "extraction_error": merged_error,
                "text_len": row.text_len,
            }
        )

    cols = [
        "ann_id",
        "SecuCode",
        "title",
        "publish_dt_utc",
        "pdf_path",
        "has_pdf",
        "file_bytes",
        "page_count",
        "has_any_text",
        "extraction_ok",
        "extraction_error",
        "text_len",
    ]
    return pd.DataFrame(report_rows, columns=cols)


def reextract_failed_rows(fixtures_dir: Path) -> pd.DataFrame:
    if not TEXTS_OUT_CSV.exists():
        raise FileNotFoundError(f"cannot re-extract: missing existing texts file {TEXTS_OUT_CSV}")
    existing = _normalize_existing_texts(pd.read_csv(TEXTS_OUT_CSV, keep_default_na=False))

    failed_mask = (~existing["extraction_ok"]) | (existing["extraction_error"].fillna("").astype(str).str.strip() != "")
    failed = existing.loc[failed_mask].copy()
    if failed.empty:
        print("re-extract: no failed rows found in existing texts_mvp.csv")
        return existing

    meta_failed = failed[["ann_id", "SecuCode", "publish_dt_utc", "publish_dt_local", "title"]].drop_duplicates("ann_id", keep="last")
    verified_failed, _ = verify_fixture_pdfs(meta_failed, fixtures_dir)
    refreshed_failed = extract_texts(verified_failed)

    kept = existing.loc[~failed_mask].copy()
    combined = pd.concat([kept, refreshed_failed], ignore_index=True)
    combined = combined.drop_duplicates("ann_id", keep="last")
    combined = combined.sort_values(["publish_dt_utc", "ann_id"], kind="stable").reset_index(drop=True)
    return _normalize_existing_texts(combined)


def _lexicon_score(text: str, pos_words: set[str], neg_words: set[str]) -> float:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return 0.0
    pos = sum(1 for tok in tokens if tok in pos_words)
    neg = sum(1 for tok in tokens if tok in neg_words)
    return (pos - neg) / max(len(tokens), 1)


def score_sentiment(text_df: pd.DataFrame) -> pd.DataFrame:
    ts = datetime.now(timezone.utc).isoformat()
    model_backend = "lexicon"
    model_name = "tiny_en_lexicon_v1"
    scored = text_df.copy().reset_index(drop=True)

    use_transformers = False
    transformer_pipeline = None
    transformer_model_name = ""

    try:
        from transformers import pipeline  # type: ignore

        transformer_pipeline = pipeline("sentiment-analysis")
        use_transformers = True
        model_backend = "transformers"
        transformer_model_name = getattr(transformer_pipeline.model.config, "_name_or_path", "unknown")
        model_name = str(transformer_model_name)
    except Exception:
        use_transformers = False

    if use_transformers and transformer_pipeline is not None:
        scores = []
        labels = []
        for text in scored["text"].fillna("").astype(str).tolist():
            if not text:
                scores.append(0.0)
                labels.append("neu")
                continue
            try:
                pred = transformer_pipeline(text[:512])[0]
                raw_label = str(pred.get("label", "")).upper()
                raw_score = float(pred.get("score", 0.0))
                if "NEG" in raw_label:
                    final_score = -raw_score
                    final_label = "neg"
                elif "POS" in raw_label:
                    final_score = raw_score
                    final_label = "pos"
                else:
                    final_score = 0.0
                    final_label = "neu"
            except Exception:
                final_score = 0.0
                final_label = "neu"
            scores.append(float(final_score))
            labels.append(final_label)
    else:
        pos_words = {"gain", "growth", "improve", "positive", "profit", "up", "upgrade", "strong", "beat"}
        neg_words = {"loss", "decline", "drop", "negative", "risk", "down", "weak", "miss", "default"}
        scores = []
        labels = []
        for text in scored["text"].fillna("").astype(str).tolist():
            s = float(_lexicon_score(text, pos_words, neg_words))
            if s > 0.003:
                l = "pos"
            elif s < -0.003:
                l = "neg"
            else:
                l = "neu"
            scores.append(s)
            labels.append(l)

    sentiment = pd.DataFrame(
        {
            "ann_id": scored["ann_id"].astype(str),
            "SecuCode": scored["SecuCode"].astype(str),
            "publish_dt": scored["publish_dt_utc"].astype(str),
            "title": scored["title"].astype(str),
            "sentiment_score": pd.Series(scores, dtype="float64"),
            "sentiment_label": pd.Series(labels, dtype="string").fillna("neu"),
            "model_backend": model_backend,
            "model_name": model_name,
            "inference_ts": ts,
            "text_len": scored["text_len"].astype("int64"),
            "extraction_ok": scored["extraction_ok"].astype(bool),
        }
    )
    return sentiment[
        [
            "ann_id",
            "SecuCode",
            "publish_dt",
            "title",
            "sentiment_score",
            "sentiment_label",
            "model_backend",
            "model_name",
            "inference_ts",
            "text_len",
            "extraction_ok",
        ]
    ].copy()


def maybe_align_15m(align_15m: bool, bars_calendar_file: Path, publish_dts: Iterable[str]) -> None:
    if not align_15m:
        return
    if not bars_calendar_file.exists():
        print(f"alignment skipped: missing bars calendar file {bars_calendar_file}")
        return

    try:
        cal = pd.read_csv(bars_calendar_file)
        if cal.empty:
            print(f"alignment skipped: empty bars calendar file {bars_calendar_file}")
            return
        first_col = cal.columns[0]
        cal_ts = pd.to_datetime(cal[first_col], errors="coerce").dropna().sort_values()
        if cal_ts.empty:
            print(f"alignment skipped: no parseable timestamps in {bars_calendar_file}")
            return
        pub_ts = pd.to_datetime(pd.Series(list(publish_dts), dtype="string"), errors="coerce", utc=True)
        mapped = 0
        values = cal_ts.to_numpy()
        for ts in pub_ts.dropna():
            idx = values.searchsorted(ts.to_datetime64(), side="left")
            if idx < len(values):
                mapped += 1
        print(
            f"alignment completed (preview only): mapped {mapped}/{len(pub_ts.dropna())} "
            f"publish timestamps using {bars_calendar_file}"
        )
    except Exception as exc:
        print(f"alignment skipped: failed to load bars calendar file {bars_calendar_file} ({exc})")


def print_missing_table(missing_rows: list[dict[str, str]]) -> None:
    preview = pd.DataFrame(missing_rows).head(20)
    if preview.empty:
        return
    print("\nMissing fixture PDFs (showing up to first 20):")
    print(preview.to_string(index=False))


def print_manual_checklist(meta_df: pd.DataFrame, verified_df: pd.DataFrame, text_df: pd.DataFrame, sent_df: pd.DataFrame) -> None:
    pdf_found = int(verified_df["has_pdf"].sum())
    pdf_missing = int((~verified_df["has_pdf"]).sum())
    extracted_ok = int(text_df["extraction_ok"].sum())
    label_dist = sent_df["sentiment_label"].value_counts(dropna=False).to_dict()

    print("\nManual test checklist")
    print(f"- rows in meta subset: {len(meta_df)}")
    print(f"- PDFs found: {pdf_found}")
    print(f"- PDFs missing: {pdf_missing}")
    print(f"- extraction_ok rows: {extracted_ok}")
    print(f"- sentiment_label distribution: {label_dist}")
    print("- sample rows (3):")

    sample = text_df.merge(
        sent_df[["ann_id", "sentiment_score", "sentiment_label"]],
        on="ann_id",
        how="left",
    ).head(3)

    for row in sample.itertuples(index=False):
        snippet = _normalize_text(str(row.text))[:120]
        print(f"  title={row.title} | score={row.sentiment_score:.6f} | label={row.sentiment_label} | text[:120]={snippet}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lean sentiment MVP runner on Dec-2025 CNINFO metadata.")
    parser.add_argument("--meta_csv", type=Path, default=DEFAULT_META, help=f"Path to metadata CSV (default: {DEFAULT_META})")
    parser.add_argument(
        "--fixtures_dir",
        type=Path,
        default=DEFAULT_FIXTURES,
        help=f"Directory containing deterministically named PDFs <ann_id>.pdf (default: {DEFAULT_FIXTURES})",
    )
    parser.add_argument("--secu_code", type=str, default="000001", help="SecuCode filter (default: 000001)")
    parser.add_argument("--n", type=int, default=20, help="Subset size after SecuCode filter (default: 20)")
    parser.add_argument(
        "--subset_mode",
        type=str,
        default="first_n",
        choices=["first_n", "existing_pdfs"],
        help="Subset strategy: first_n (default) or existing_pdfs",
    )
    parser.add_argument(
        "--triage_only",
        action="store_true",
        help="Run subset+fixture diagnostics and write extraction_failures_report.csv, then exit.",
    )
    parser.add_argument(
        "--reextract_failed",
        action="store_true",
        help="Re-run extraction only for failed ann_ids from existing data/interim/texts_mvp.csv.",
    )
    parser.add_argument(
        "--include_failed_in_sentiment",
        action="store_true",
        help="Include extraction failures in sentiment scoring. Default excludes extraction_ok=False rows.",
    )
    parser.add_argument("--align_15m", action="store_true", help="Optional preview alignment against bars calendar file.")
    parser.add_argument(
        "--bars_calendar_file",
        type=Path,
        default=DEFAULT_BARS_CALENDAR,
        help=f"Bars calendar file path (default: {DEFAULT_BARS_CALENDAR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.triage_only and args.reextract_failed:
        return _fail("use only one of --triage_only or --reextract_failed per run")
    if args.n <= 0:
        return _fail("--n must be positive")
    if not re.fullmatch(r"\d{6}", str(args.secu_code)):
        return _fail("--secu_code must be exactly 6 digits (example: 000001)")

    _ensure_dir(META_OUT.parent)
    _ensure_dir(TEXTS_OUT_CSV.parent)
    _ensure_dir(SENT_OUT_CSV.parent)

    if args.reextract_failed:
        try:
            texts = reextract_failed_rows(args.fixtures_dir)
        except Exception as exc:
            return _fail(f"re-extract failed: {exc}")
        texts.to_parquet(TEXTS_OUT_PARQUET, index=False)
        texts.to_csv(TEXTS_OUT_CSV, index=False)
        print(f"re-extract done: wrote {TEXTS_OUT_PARQUET}")
        print(f"re-extract done: wrote {TEXTS_OUT_CSV}")
        meta_from_texts = texts[["ann_id", "SecuCode", "publish_dt_utc", "publish_dt_local", "title"]].drop_duplicates("ann_id")
        verified_for_report, _ = verify_fixture_pdfs(meta_from_texts, args.fixtures_dir)
        meta_out = verified_for_report[
            ["ann_id", "SecuCode", "publish_dt_utc", "publish_dt_local", "title", "pdf_path", "has_pdf"]
        ].copy()
        meta_out.to_csv(META_OUT, index=False)
        report = build_triage_report(verified_for_report, texts)
        report.to_csv(FAIL_REPORT_OUT, index=False)
        print(f"re-extract done: wrote {META_OUT}")
        print(f"triage done: wrote {FAIL_REPORT_OUT}")
        return 0

    apply_secu_filter_in_existing = "--secu_code" in sys.argv

    try:
        meta_mvp = build_meta_subset(
            args.meta_csv,
            args.fixtures_dir,
            args.secu_code,
            args.n,
            args.subset_mode,
            apply_secu_filter_in_existing,
        )
    except Exception as exc:
        return _fail(f"step 1 failed: {exc}")

    try:
        verified, missing_rows = verify_fixture_pdfs(meta_mvp, args.fixtures_dir)
    except Exception as exc:
        return _fail(f"step 2 failed: {exc}")
    verified.to_csv(META_OUT, index=False)
    print(f"step 1 done: wrote {META_OUT}")

    existing_texts_for_report = None
    if TEXTS_OUT_CSV.exists():
        try:
            existing_texts_for_report = pd.read_csv(TEXTS_OUT_CSV, keep_default_na=False)
        except Exception as exc:
            print(f"warning: failed to read existing texts for triage enrichment ({exc})")

    if args.triage_only:
        report = build_triage_report(verified, existing_texts_for_report)
        report.to_csv(FAIL_REPORT_OUT, index=False)
        print(f"triage done: wrote {FAIL_REPORT_OUT}")
        return 0

    if missing_rows:
        print_missing_table(missing_rows)
        if args.subset_mode == "first_n":
            print(
                "\nstep 2 failed: missing fixture PDFs.\n"
                "Required naming format is <ann_id>.pdf in the fixtures directory.\n"
                f"How to proceed:\n- rerun with --subset_mode existing_pdfs\n"
                f"- or add PDFs named <ann_id>.pdf into {args.fixtures_dir}"
            )
            return 1
    pdf_found = int(verified["has_pdf"].sum())
    pdf_missing = int((~verified["has_pdf"]).sum())
    print(f"step 2 done: PDFs found={pdf_found}, missing={pdf_missing} in {args.fixtures_dir}")

    texts = extract_texts(verified)
    texts.to_parquet(TEXTS_OUT_PARQUET, index=False)
    texts.to_csv(TEXTS_OUT_CSV, index=False)
    print(f"step 3 done: wrote {TEXTS_OUT_PARQUET}")
    print(f"step 3 done: wrote {TEXTS_OUT_CSV}")
    report = build_triage_report(verified, texts)
    report.to_csv(FAIL_REPORT_OUT, index=False)
    print(f"step 3 done: wrote {FAIL_REPORT_OUT}")

    sentiment_input = texts if args.include_failed_in_sentiment else texts.loc[texts["extraction_ok"]].copy()
    sentiment_input = sentiment_input.reset_index(drop=True)
    if not args.include_failed_in_sentiment:
        removed = len(texts) - len(sentiment_input)
        print(f"step 4 info: excluding {removed} extraction-failed rows from sentiment (default behavior)")
    sentiments = score_sentiment(sentiment_input)
    sentiments.to_parquet(SENT_OUT_PARQUET, index=False)
    sentiments.to_csv(SENT_OUT_CSV, index=False)
    print(f"step 4 done: wrote {SENT_OUT_PARQUET}")
    print(f"step 4 done: wrote {SENT_OUT_CSV}")

    maybe_align_15m(args.align_15m, args.bars_calendar_file, sentiments["publish_dt"].tolist())
    print_manual_checklist(meta_mvp, verified, texts, sentiments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
