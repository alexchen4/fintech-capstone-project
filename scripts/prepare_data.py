"""
Data Preparation Script (pre-inference validation)
===================================================
Data source hierarchy:
  2018-2019 : CNINFO announcements (per-stock) + CCTV financial news (market-level)
  2020-2022 : CNINFO announcements (per-stock) + Baidu economic news (market-level, extended)
  2023-2025 : CNINFO announcements (per-stock) + Baidu economic news + EastMoney news

Steps:
  1. Verify announcements completeness (50 stocks x 2018-2025)
  2. Normalize all timestamp formats to YYYY-MM-DD HH:MM:SS
  3. Download CCTV financial news (2018-01-01 ~ 2019-12-31)  -> market_news
  4. Download Baidu economic news  (2020-01-01 ~ 2025-12-31) -> market_news
  5. Print final readiness report before starting NolBERT inference

New database table:
  market_news  -- market-level daily financial news
                  source field: 'cctv' / 'baidu_economic'

Python 3.9 compatible
"""

from __future__ import annotations

import io
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import akshare as ak
import pandas as pd
from tqdm import tqdm

DB_PATH = Path(__file__).parent.parent / "sentiment_data.db"

STOCKS = [
    "000400", "000415", "000423", "000425", "000503", "000559",
    "000581", "000629", "000738", "000778", "000826", "000831",
    "000858", "001979", "002129", "002142", "002153", "002236",
    "002353", "002456", "002594", "300024", "300058", "300146",
    "600009", "600030", "600038", "600085", "600111", "600309",
    "600332", "600362", "600398", "600570", "600585", "600642",
    "600663", "600688", "600999", "601018", "601169", "601231",
    "601328", "601555", "601718", "601818", "601857", "601933",
    "601988", "603000",
]
YEARS = [str(y) for y in range(2018, 2026)]

# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────
MARKET_NEWS_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_news (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    news_date   TEXT    NOT NULL,
    news_time   TEXT,
    title       TEXT    NOT NULL,
    event_tag   TEXT,
    news_dt     TEXT    NOT NULL,
    source      TEXT    DEFAULT 'baidu_economic',
    UNIQUE(news_date, title)
);
CREATE INDEX IF NOT EXISTS idx_mkt_date   ON market_news(news_date);
CREATE INDEX IF NOT EXISTS idx_mkt_dt     ON market_news(news_dt);
CREATE INDEX IF NOT EXISTS idx_mkt_source ON market_news(source);
"""

INSERT_SQL = """
INSERT OR IGNORE INTO market_news (news_date, news_time, title, event_tag, news_dt, source)
VALUES (?, ?, ?, ?, ?, ?)
"""


def init_market_news(conn: sqlite3.Connection):
    conn.executescript(MARKET_NEWS_SCHEMA)
    conn.commit()
    print("[DB] market_news table ready")


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────
def _date_range(start: date, end: date) -> List[date]:
    d, out = start, []
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _existing_dates(conn: sqlite3.Connection, source: str) -> set:
    rows = conn.execute(
        "SELECT DISTINCT news_date FROM market_news WHERE source=?", (source,)
    ).fetchall()
    return {r[0] for r in rows}


# ──────────────────────────────────────────────
# Step 1: Announcement completeness check
# ──────────────────────────────────────────────
def check_announcements(conn: sqlite3.Connection) -> bool:
    print("\n" + "=" * 65)
    print("[STEP 1/5] Announcement Completeness Check")

    rows = conn.execute("""
        SELECT SecuCode, SUBSTR(announce_dt, 1, 4) AS yr, COUNT(*) AS cnt
        FROM announcements
        GROUP BY SecuCode, yr
    """).fetchall()

    from collections import defaultdict
    stock_year: dict = defaultdict(dict)
    for code, yr, cnt in rows:
        stock_year[code][yr] = cnt

    gaps = [
        (code, yr) for code in STOCKS for yr in YEARS
        if stock_year[code].get(yr, 0) == 0
    ]

    total    = conn.execute("SELECT COUNT(*) FROM announcements").fetchone()[0]
    n_stocks = conn.execute(
        "SELECT COUNT(DISTINCT SecuCode) FROM announcements"
    ).fetchone()[0]

    print(f"  Total announcements : {total:,}")
    print(f"  Stocks covered      : {n_stocks} / {len(STOCKS)}")
    print(f"  Gaps (stock, year)  : {len(gaps)}", end="")
    if gaps:
        print("  <- missing data!")
        for g in gaps[:10]:
            print(f"    {g[0]} {g[1]}")
        return False
    print("  <- complete")
    return True


# ──────────────────────────────────────────────
# Step 2: Timestamp normalization
# ──────────────────────────────────────────────
def normalize_timestamps(conn: sqlite3.Connection):
    print("\n[STEP 2/5] Timestamp Normalization")
    for tbl, col in [("announcements", "announce_dt"), ("news_em", "news_dt")]:
        bad = conn.execute(f"""
            SELECT COUNT(*) FROM {tbl}
            WHERE {col} NOT LIKE '____-__-__ __:__:__'
        """).fetchone()[0]
        if bad > 0:
            conn.execute(f"""
                UPDATE {tbl} SET {col} = {col} || ' 00:00:00'
                WHERE LENGTH({col}) = 10
            """)
            conn.commit()
            print(f"  Fixed {tbl}.{col}: {bad} rows")

    for tbl, col in [("announcements", "announce_dt"), ("news_em", "news_dt")]:
        sample = [r[0] for r in conn.execute(
            f"SELECT {col} FROM {tbl} ORDER BY RANDOM() LIMIT 2"
        ).fetchall()]
        print(f"  {tbl:<18} sample: {sample}")
    print("  Format: YYYY-MM-DD HH:MM:SS — normalized")


# ──────────────────────────────────────────────
# Step 3: CCTV financial news (2018-2019)
# ──────────────────────────────────────────────
def fetch_cctv_day(day: date, retries: int = 2) -> Optional[pd.DataFrame]:
    date_str = day.strftime("%Y%m%d")
    for attempt in range(retries):
        try:
            return ak.news_cctv(date=date_str)
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


def download_cctv_news(conn: sqlite3.Connection):
    """CCTV financial news 2018-01-01 ~ 2019-12-31 to fill early market-level news gap."""
    print("\n[STEP 3/5] CCTV Financial News (2018-01-01 ~ 2019-12-31)")

    existing  = _existing_dates(conn, "cctv")
    all_dates = _date_range(date(2018, 1, 1), date(2019, 12, 31))
    pending   = [d for d in all_dates if d.strftime("%Y-%m-%d") not in existing]
    print(f"  Existing: {len(existing)} days  |  Pending: {len(pending)} days")

    if not pending:
        n = conn.execute(
            "SELECT COUNT(*) FROM market_news WHERE source='cctv'"
        ).fetchone()[0]
        print(f"  CCTV news complete: {n:,} rows")
        return

    # Test API availability
    test = fetch_cctv_day(pending[0])
    if test is None:
        print("  [WARN] ak.news_cctv() unavailable — skipping CCTV step")
        print("  [HINT] 2018-2019 will use CNINFO announcements only")
        return

    total_inserted = 0
    empty_days = 0

    for day in tqdm(pending, desc="CCTV news", unit="day", ascii=True):
        df       = fetch_cctv_day(day)
        date_str = day.strftime("%Y-%m-%d")

        if df is None or df.empty:
            empty_days += 1
            time.sleep(0.3)
            continue

        cols  = list(df.columns)
        rows  = []
        for _, r in df.iterrows():
            # Auto-detect title column (AKShare column names may vary by version)
            t_title = ""
            t_time  = ""
            for c in cols:
                cl = c.lower()
                if "title" in cl or "标题" in c:  # "标题" = "title" in Chinese (AKShare may return Chinese column names)
                    t_title = str(r[c]).strip()
                elif "time" in cl or "时间" in c:  # "时间" = "time" in Chinese (AKShare may return Chinese column names)
                    t_time = str(r[c]).strip()
            if not t_title:
                t_title = str(r[cols[-1]]).strip()

            if not t_title or t_title.lower() in ("nan", "none", ""):
                continue

            t_dt = (f"{date_str} {t_time[:5]}:00"
                    if t_time and len(t_time) >= 5 and ":" in t_time
                    else f"{date_str} 00:00:00")
            rows.append((date_str, t_time, t_title, "", t_dt, "cctv"))

        if rows:
            conn.executemany(INSERT_SQL, rows)
            conn.commit()
            total_inserted += len(rows)

        time.sleep(0.5)

    n_db = conn.execute(
        "SELECT COUNT(*) FROM market_news WHERE source='cctv'"
    ).fetchone()[0]
    print(f"  Inserted: {total_inserted:,}  |  Empty days: {empty_days}  |  Total: {n_db:,}")


# ──────────────────────────────────────────────
# Step 4: Baidu economic news (2020-2025)
# ──────────────────────────────────────────────
def fetch_baidu_day(day: date, retries: int = 2) -> Optional[pd.DataFrame]:
    date_str = day.strftime("%Y%m%d")
    for attempt in range(retries):
        try:
            return ak.news_economic_baidu(date=date_str)
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


def download_baidu_news(conn: sqlite3.Connection):
    """
    Baidu economic news 2020-01-01 ~ 2025-12-31.
    2020-2022 is an extended range test (data may be sparse);
    2023-2025 coverage is stable.
    """
    print("\n[STEP 4/5] Baidu Economic News (2020-01-01 ~ 2025-12-31)")
    print("  Note: 2020-2022 extended range test — data may be sparse")

    existing  = _existing_dates(conn, "baidu_economic")
    all_dates = _date_range(date(2020, 1, 1), date(2025, 12, 31))
    pending   = [d for d in all_dates if d.strftime("%Y-%m-%d") not in existing]
    print(f"  Existing: {len(existing)} days  |  Pending: {len(pending)} days  "
          f"(est. ~{len(pending) * 0.5 / 60:.0f} min)")

    if not pending:
        total = conn.execute(
            "SELECT COUNT(*) FROM market_news WHERE source='baidu_economic'"
        ).fetchone()[0]
        print(f"  Baidu news complete: {total:,} rows")
        return

    total_inserted = 0
    empty_days = 0

    for day in tqdm(pending, desc="Baidu news", unit="day", ascii=True):
        df       = fetch_baidu_day(day)
        date_str = day.strftime("%Y-%m-%d")

        if df is None or df.empty:
            empty_days += 1
            time.sleep(0.3)
            continue

        cols = list(df.columns)
        rows = []
        for _, r in df.iterrows():
            # Baidu economic news columns: [time, title, summary, tag]
            t_time  = str(r[cols[0]]).strip() if len(cols) > 0 else ""
            t_title = str(r[cols[1]]).strip() if len(cols) > 1 else str(r[cols[0]]).strip()
            t_tag   = str(r[cols[3]]).strip() if len(cols) > 3 else ""

            if not t_title or t_title.lower() in ("nan", "none", ""):
                continue

            t_dt = (f"{date_str} {t_time[:5]}:00"
                    if t_time and len(t_time) >= 5 and ":" in t_time
                    else f"{date_str} 00:00:00")
            rows.append((date_str, t_time, t_title, t_tag, t_dt, "baidu_economic"))

        if rows:
            conn.executemany(INSERT_SQL, rows)
            conn.commit()
            total_inserted += len(rows)

        time.sleep(0.4)

    total_db = conn.execute(
        "SELECT COUNT(*) FROM market_news WHERE source='baidu_economic'"
    ).fetchone()[0]
    dr = conn.execute(
        "SELECT MIN(news_date), MAX(news_date) "
        "FROM market_news WHERE source='baidu_economic'"
    ).fetchone()
    print(f"  Inserted: {total_inserted:,}  |  Empty days: {empty_days}")
    print(f"  Baidu total: {total_db:,} rows  |  {dr[0]} ~ {dr[1]}")


# ──────────────────────────────────────────────
# Step 5: Final readiness report
# ──────────────────────────────────────────────
def final_report(conn: sqlite3.Connection):
    print("\n" + "=" * 65)
    print("[STEP 5/5] Final Readiness Report")
    print("=" * 65)

    tables = [
        ("announcements", "announce_dt", "SecuCode"),
        ("news_em",       "news_dt",     "SecuCode"),
        ("market_news",   "news_date",    None),
    ]
    descs = {
        "announcements": "CNINFO official disclosures (per-stock, 2018-2025)",
        "news_em":       "EastMoney news (per-stock, recent)",
        "market_news":   "Market-level news (CCTV 2018-19 + Baidu 2020-25)",
    }

    total_texts = 0
    for tbl, dt_col, code_col in tables:
        n  = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        dr = conn.execute(f"SELECT MIN({dt_col}), MAX({dt_col}) FROM {tbl}").fetchone()
        d0 = dr[0][:10] if dr[0] else "N/A"
        d1 = dr[1][:10] if dr[1] else "N/A"
        if code_col:
            nc = conn.execute(
                f"SELECT COUNT(DISTINCT {code_col}) FROM {tbl}"
            ).fetchone()[0]
            print(f"  {tbl:<18}: {n:>8,} rows | {nc} stocks | {d0} ~ {d1}")
        else:
            n_cctv  = conn.execute(
                "SELECT COUNT(*) FROM market_news WHERE source='cctv'"
            ).fetchone()[0]
            n_baidu = conn.execute(
                "SELECT COUNT(*) FROM market_news WHERE source='baidu_economic'"
            ).fetchone()[0]
            print(f"  {tbl:<18}: {n:>8,} rows | market   | {d0} ~ {d1}")
            print(f"    |- cctv           : {n_cctv:>6,} rows (2018-2019)")
            print(f"    `- baidu_economic : {n_baidu:>6,} rows (2020-2025)")
        print(f"    source: {descs.get(tbl, tbl)}")
        total_texts += n

    n_done    = conn.execute("SELECT COUNT(*) FROM sentiment_raw").fetchone()[0]
    n_pending = total_texts - n_done

    print(f"\n  Inference progress:")
    print(f"    Total texts  : {total_texts:>8,}")
    print(f"    Inferred     : {n_done:>8,}")
    print(f"    Pending      : {n_pending:>8,}")

    print(f"\n  Coverage:")
    print(f"    2018-2019 : CNINFO announcements + CCTV news (market-level)")
    print(f"    2020-2022 : CNINFO announcements + Baidu economic news")
    print(f"    2023-2025 : CNINFO announcements + Baidu news + EastMoney news")

    print("\n" + "=" * 65)
    if n_pending > 0:
        import torch
        device = "GPU" if torch.cuda.is_available() else "CPU"
        secs = 0.02 if torch.cuda.is_available() else 0.20   # translate + infer combined
        eta  = n_pending * secs / 60
        print(f"  Data preparation complete — inference ready (includes zh->en translation)")
        print(f"  Command: python scripts/run_sentiment_nlp.py")
        print(f"  Estimated time ({device}): {eta:.0f} min")
    else:
        print("  All texts have been inferred")
    print("=" * 65)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Data Preparation Script")
    print(f"Database: {DB_PATH}")
    print("=" * 65)

    if not DB_PATH.exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    init_market_news(conn)

    check_announcements(conn)   # Step 1
    normalize_timestamps(conn)  # Step 2
    download_cctv_news(conn)    # Step 3: 2018-2019
    download_baidu_news(conn)   # Step 4: 2020-2025
    final_report(conn)          # Step 5

    conn.close()


if __name__ == "__main__":
    main()
