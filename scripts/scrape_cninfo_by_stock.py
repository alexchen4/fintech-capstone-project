#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "universe.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "meta" / "by_stock"

API_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "https://www.cninfo.com.cn/new/disclosure/stock?orgId=gssz0000858&stockCode=000858",
}

OUT_COLUMNS = ["SecuCode", "publish_ts", "title", "pdf_url", "source", "column", "orgId"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-time full-history CNINFO scrape by stock.")
    p.add_argument("--universe_csv", type=Path, default=DEFAULT_UNIVERSE)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--start_date", default="2018-01-02")
    p.add_argument("--end_date", default="2025-12-31")
    p.add_argument("--page_size", type=int, default=30)
    p.add_argument("--sleep_sec", type=float, default=1.0)
    p.add_argument("--jitter_sec", type=float, default=0.3)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_code(x: object) -> str:
    s = str(x).strip()
    if not s:
        return ""
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return ""
    return digits[-6:].zfill(6)


def derive_exchange_plate_orgid(code: str) -> tuple[str, str, str]:
    if code.startswith(("000", "001", "002", "003", "300")):
        return "szse", "sz", f"gssz0{code}"
    if code.startswith(("600", "601", "603", "605")):
        return "sse", "sh", f"gssh0{code}"
    return "sse", "sh", f"gssh0{code}"


def to_utc_str_from_ms(ms: Any) -> str:
    try:
        ts = pd.to_datetime(ms, unit="ms", utc=True, errors="coerce")
    except Exception:
        return ""
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M:%S+00:00")


def load_universe_codes(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"universe_csv not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return []
    code_col = None
    for c in ["SecuCode", "ticker", "secu_code", "symbol"]:
        if c in df.columns:
            code_col = c
            break
    if code_col is None:
        code_col = df.columns[0]
    codes = sorted({normalize_code(v) for v in df[code_col].tolist() if normalize_code(v)})
    return codes


def post_with_retry(session: requests.Session, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
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
        return None, "Invalid response type"
    return data, None


def scrape_one_stock(
    session: requests.Session,
    code: str,
    exchange: str,
    plate: str,
    derived_orgid: str,
    start_date: str,
    end_date: str,
    page_size: int,
    sleep_sec: float,
    jitter_sec: float,
    page_cap: int = 500,
) -> tuple[pd.DataFrame, str | None]:
    rows: list[dict[str, Any]] = []
    page = 1

    while page <= page_cap:
        payload = {
            "stock": f"{code},{derived_orgid}",
            "tabName": "fulltext",
            "pageSize": page_size,
            "pageNum": page,
            "column": exchange,
            "category": "",
            "plate": plate,
            "seDate": f"{start_date}~{end_date}",
            "searchkey": "",
            "secid": "",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }

        data, err = post_with_retry(session, payload)
        if err is not None:
            return pd.DataFrame(columns=OUT_COLUMNS), err

        announcements = data.get("announcements")
        if announcements is None:
            announcements = []
        if not isinstance(announcements, list):
            announcements = []

        try:
            api_total = int(data.get("totalRecordNum", 0))
        except Exception:
            api_total = 0

        page_rows = 0
        for ann in announcements:
            if not isinstance(ann, dict):
                continue

            adjunct_url = (ann.get("adjunctUrl") or "").strip()
            if not adjunct_url:
                continue

            publish_ts = to_utc_str_from_ms(ann.get("announcementTime"))
            if not publish_ts:
                continue

            sec_code = normalize_code(ann.get("secCode") or code)
            if not sec_code:
                sec_code = code

            title = (ann.get("announcementTitle") or "").strip()
            actual_orgid = str(ann.get("orgId") or "").strip()

            rows.append(
                {
                    "SecuCode": sec_code,
                    "publish_ts": publish_ts,
                    "title": title,
                    "pdf_url": f"https://static.cninfo.com.cn/{adjunct_url}",
                    "source": "CNINFO",
                    "column": exchange,
                    "orgId": actual_orgid,
                }
            )
            page_rows += 1

        print(
            f"[scrape] code={code} page={page} fetched={page_rows} "
            f"total_so_far={len(rows)} api_total={api_total}"
        )

        if not announcements:
            break
        if page * page_size >= api_total:
            break

        page += 1
        time.sleep(max(0.0, sleep_sec) + random.uniform(0.0, max(0.0, jitter_sec)))

    out = pd.DataFrame(rows, columns=OUT_COLUMNS)
    if out.empty:
        return out, None

    out = out.drop_duplicates(subset=["SecuCode", "publish_ts", "title"], keep="first")
    out = out.sort_values(["publish_ts", "SecuCode", "title"], kind="stable").reset_index(drop=True)
    return out, None


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    codes = load_universe_codes(args.universe_csv)
    session = requests.Session()
    # Warm up session with a page visit before scraping
    try:
        session.get(
            "https://www.cninfo.com.cn/new/index",
            headers=API_HEADERS,
            timeout=15
        )
        time.sleep(1.0)
    except Exception:
        pass  # proceed anyway

    completed = 0
    skipped = 0
    failed = 0
    total_announcements = 0
    failures: list[dict[str, str]] = []

    for i, code in enumerate(codes):
        if i > 0 and i % 10 == 0:
            session = requests.Session()
            try:
                session.get(
                    "https://www.cninfo.com.cn/new/index",
                    headers=API_HEADERS,
                    timeout=15
                )
                time.sleep(2.0)
                print(f"[scrape] session refreshed at stock {i}")
            except Exception:
                pass
        out_file = args.out_dir / f"announcements_{code}.csv"
        if out_file.exists() and not args.overwrite:
            print(f"[scrape] code={code} SKIP existing file={out_file}")
            skipped += 1
            continue

        exchange, plate, derived_orgid = derive_exchange_plate_orgid(code)

        try:
            df, err = scrape_one_stock(
                session=session,
                code=code,
                exchange=exchange,
                plate=plate,
                derived_orgid=derived_orgid,
                start_date=args.start_date,
                end_date=args.end_date,
                page_size=args.page_size,
                sleep_sec=args.sleep_sec,
                jitter_sec=args.jitter_sec,
            )
            if err is not None:
                failed += 1
                failures.append({"SecuCode": code, "error": err})
                print(f"[scrape] code={code} FAIL error={err}")
                continue

            df.to_csv(out_file, index=False, encoding="utf-8")
            completed += 1
            total_announcements += int(len(df))
            print(f"[scrape] code={code} DONE rows={len(df)} file={out_file}")
        except Exception as e:
            failed += 1
            failures.append({"SecuCode": code, "error": f"{type(e).__name__}: {e}"})
            print(f"[scrape] code={code} FAIL error={type(e).__name__}: {e}")

    summary = {
        "total_stocks": len(codes),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "total_announcements": total_announcements,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "failures": failures,
    }

    summary_path = args.out_dir / "scrape_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[scrape] summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[scrape] summary_file={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
