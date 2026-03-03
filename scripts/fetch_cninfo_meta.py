#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from calendar import monthrange
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "universe.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "meta"

API_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://www.cninfo.com.cn/new/index",
}

OUTPUT_COLUMNS = ["ticker", "publish_ts", "title", "pdf_url", "source", "column", "orgId"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch CNINFO announcement metadata for one month.")
    p.add_argument("--month", required=True, help="Month tag YYYY-MM")
    p.add_argument("--universe_csv", type=Path, default=DEFAULT_UNIVERSE)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--sleep_sec", type=float, default=1.0)
    p.add_argument("--jitter_sec", type=float, default=0.5)
    p.add_argument("--page_size", type=int, default=30)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing month CSV")
    return p.parse_args()


def normalize_code(x: object) -> str:
    s = str(x).strip()
    if not s:
        return ""
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return ""
    return digits[-6:].zfill(6)


def load_universe_codes(universe_csv: Path) -> list[str]:
    if not universe_csv.exists():
        raise FileNotFoundError(f"universe_csv not found: {universe_csv}")

    df = pd.read_csv(universe_csv)
    if df.empty:
        raise ValueError(f"universe_csv is empty: {universe_csv}")

    code_col = None
    for c in ["SecuCode", "ticker", "secu_code", "symbol"]:
        if c in df.columns:
            code_col = c
            break
    if code_col is None:
        code_col = df.columns[0]

    codes = sorted({normalize_code(v) for v in df[code_col].tolist() if normalize_code(v)})
    return codes


def derive_exchange_orgid(code: str) -> tuple[str, str]:
    if code.startswith(("000", "001", "002", "003", "300")):
        return "szse", f"gssz{code}"
    if code.startswith(("600", "601", "603", "605")):
        return "sse", f"gssh{code}"
    # Default fallback: treat as SSE pattern if prefix is unexpected.
    return "sse", f"gssh{code}"


def month_date_range(month: str) -> str:
    dt = datetime.strptime(month, "%Y-%m")
    last_day = monthrange(dt.year, dt.month)[1]
    return f"{dt.year:04d}-{dt.month:02d}-01~{dt.year:04d}-{dt.month:02d}-{last_day:02d}"


def to_utc_string(ms_epoch: Any) -> str:
    try:
        ts = pd.to_datetime(ms_epoch, unit="ms", utc=True, errors="coerce")
    except Exception:
        return ""
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M:%S+00:00")


def request_page(
    session: requests.Session,
    payload: dict[str, Any],
    sleep_sec: float,
    jitter_sec: float,
) -> tuple[dict[str, Any] | None, str | None]:
    time.sleep(max(0.0, sleep_sec) + random.uniform(0.0, max(0.0, jitter_sec)))

    try:
        resp = session.post(API_URL, data=payload, headers=API_HEADERS, timeout=20)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    if resp.status_code == 429:
        time.sleep(60.0)
        try:
            resp = session.post(API_URL, data=payload, headers=API_HEADERS, timeout=20)
        except Exception as e:
            return None, f"429_retry_{type(e).__name__}: {e}"

    if resp.status_code >= 400:
        return None, f"HTTP {resp.status_code}"

    try:
        data = resp.json()
    except Exception as e:
        return None, f"JSONDecodeError: {e}"

    if not isinstance(data, dict):
        return None, "Invalid response type (not dict)"
    return data, None


def fetch_for_stock_param(
    session: requests.Session,
    code: str,
    exchange: str,
    orgid: str,
    stock_param: str,
    month: str,
    page_size: int,
    sleep_sec: float,
    jitter_sec: float,
    max_pages: int = 20,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    se_date = month_date_range(month)
    rows: list[dict[str, Any]] = []
    page = 1
    failed = False
    fail_msg = None

    while page <= max_pages:
        payload = {
            "stock": stock_param,
            "tabName": "fulltext",
            "pageSize": page_size,
            "pageNum": page,
            "column": exchange,
            "category": "",
            "plate": "",
            "seDate": se_date,
            "searchkey": "",
            "secid": "",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }

        data, err = request_page(session, payload, sleep_sec=sleep_sec, jitter_sec=jitter_sec)
        if err is not None:
            failed = True
            fail_msg = err
            break

        announcements = data.get("announcements") or []
        if not isinstance(announcements, list):
            announcements = []

        total_record_num_raw = data.get("totalRecordNum", 0)
        try:
            total_record_num = int(total_record_num_raw)
        except Exception:
            total_record_num = 0

        valid_on_page = 0
        for ann in announcements:
            if not isinstance(ann, dict):
                continue

            adjunct_url = (ann.get("adjunctUrl") or "").strip()
            if not adjunct_url:
                # Skip rows missing adjunctUrl per requirement.
                continue

            title = (ann.get("announcementTitle") or "").strip()
            publish_ts = to_utc_string(ann.get("announcementTime"))

            rows.append(
                {
                    "ticker": code,
                    "publish_ts": publish_ts,
                    "title": title,
                    "pdf_url": f"https://static.cninfo.com.cn/finalpage/{adjunct_url}",
                    "source": "CNINFO",
                    "column": exchange,
                    "orgId": orgid,
                }
            )
            valid_on_page += 1

        print(
            f"[fetch_cninfo_meta] stock={code} exchange={exchange} page={page} "
            f"announcements={valid_on_page} total_so_far={len(rows)}"
        )

        # Stop conditions per requirement:
        # - announcements empty
        # - or pageNum*pageSize >= totalRecordNum
        # - or max page cap reached by loop condition
        if not announcements:
            break
        if page * page_size >= total_record_num:
            break

        page += 1

    return rows, failed, fail_msg


def main() -> int:
    args = parse_args()

    month = args.month
    try:
        datetime.strptime(month, "%Y-%m")
    except ValueError:
        raise ValueError(f"--month must be YYYY-MM, got: {month}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"announcements_meta_{month}.csv"
    out_summary = args.out_dir / f"fetch_summary_{month}.json"

    if out_csv.exists() and not args.overwrite:
        print(f"[fetch_cninfo_meta] WARNING output exists, skip: {out_csv}")
        return 0

    codes = load_universe_codes(args.universe_csv)
    session = requests.Session()

    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    stocks_with_results = 0
    stocks_empty = 0
    stocks_failed = 0

    for code in codes:
        exchange, orgid = derive_exchange_orgid(code)
        stock_param = f"{orgid},{code}"

        rows, failed, fail_msg = fetch_for_stock_param(
            session=session,
            code=code,
            exchange=exchange,
            orgid=orgid,
            stock_param=stock_param,
            month=month,
            page_size=args.page_size,
            sleep_sec=args.sleep_sec,
            jitter_sec=args.jitter_sec,
        )

        # Fallback required: if standard orgId gets zero results, try ticker only.
        if (not failed) and len(rows) == 0:
            fallback_rows, fallback_failed, fallback_msg = fetch_for_stock_param(
                session=session,
                code=code,
                exchange=exchange,
                orgid=orgid,
                stock_param=code,  # ticker-only fallback
                month=month,
                page_size=args.page_size,
                sleep_sec=args.sleep_sec,
                jitter_sec=args.jitter_sec,
            )
            if fallback_failed:
                failed = True
                fail_msg = f"fallback_failed: {fallback_msg}"
            elif len(fallback_rows) > 0:
                rows = fallback_rows

        if failed:
            stocks_failed += 1
            failures.append({"ticker": code, "error": fail_msg or "unknown_error"})
            continue

        if len(rows) == 0:
            stocks_empty += 1
        else:
            stocks_with_results += 1
            all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    if not out_df.empty:
        out_df = out_df.drop_duplicates(
            subset=["ticker", "publish_ts", "title", "pdf_url"], keep="first"
        ).sort_values(["ticker", "publish_ts", "title"], kind="stable").reset_index(drop=True)

    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "month": month,
        "total_stocks_queried": len(codes),
        "stocks_with_results": stocks_with_results,
        "stocks_empty": stocks_empty,
        "stocks_failed": stocks_failed,
        "total_announcements": int(len(out_df)),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "failures": failures,
    }

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[fetch_cninfo_meta] summary")
    print(f"month={summary['month']}")
    print(f"total_stocks_queried={summary['total_stocks_queried']}")
    print(f"stocks_with_results={summary['stocks_with_results']}")
    print(f"stocks_empty={summary['stocks_empty']}")
    print(f"stocks_failed={summary['stocks_failed']}")
    print(f"total_announcements={summary['total_announcements']}")
    print(f"out_csv={out_csv}")
    print(f"out_summary={out_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
