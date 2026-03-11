"""
A-Share Sentiment / News Raw Text Collection Script
====================================================
Purpose : Collect raw text data for FinBERT / NolBERT processing
Coverage: 2018-01-01 ~ 2025-12-31

Data sources:
  1. CNINFO (AKShare) -- official regulatory disclosures, full historical coverage,
                         highest source credibility
  2. EastMoney news (AKShare) -- media / analyst reports, includes content snippets

Influence ranking fields:
  CNINFO announcements : importance_rank inferred from title keywords
                         (annual report=1, major events=4, general=99)
  EastMoney news       : source_rank by outlet tier
                         (official media=1, professional media=2, others=3)

Database tables:
  announcements  -- CNINFO official disclosures (raw title text + URL)
  news_em        -- EastMoney media news (raw title + content snippet)
  stocks         -- stock basic info
  meta           -- collection run metadata

Python 3.9 compatible (no X|Y union type hints)
"""

import io
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Force UTF-8 stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import akshare as ak
import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
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

# CNINFO fetched in 6-month segments to avoid oversized single requests
DATE_SEGMENTS = [
    ("20180101", "20180630"), ("20180701", "20181231"),
    ("20190101", "20190630"), ("20190701", "20191231"),
    ("20200101", "20200630"), ("20200701", "20201231"),
    ("20210101", "20210630"), ("20210701", "20211231"),
    ("20220101", "20220630"), ("20220701", "20221231"),
    ("20230101", "20230630"), ("20230701", "20231231"),
    ("20240101", "20240630"), ("20240701", "20241231"),
    ("20250101", "20250630"), ("20250701", "20251231"),
]

MARKET_PARAM = "沪深京"  # valid market param for stock_zh_a_disclosure_report_cninfo

DB_PATH = Path(__file__).parent.parent / "sentiment_data.db"

# ──────────────────────────────────────────────
# Influence ranking: CNINFO title keywords -> importance_rank
# Lower value = higher importance (used to sort before FinBERT processing)
# ──────────────────────────────────────────────
TITLE_IMPORTANCE_RULES = [
    (1,  ["年度报告", "年报", "年度财务报告"]),
    (2,  ["半年度报告", "半年报", "中期报告"]),
    (3,  ["季度报告", "一季报", "三季报"]),
    (4,  ["重大资产重组", "重大事项", "重大合同", "重大诉讼"]),
    (5,  ["股权激励", "股票期权", "限制性股票"]),
    (6,  ["分红", "派息", "利润分配", "现金红利"]),
    (7,  ["股东大会", "临时股东大会", "年度股东大会"]),
    (8,  ["收购", "兼并", "合并", "战略合作"]),
    (9,  ["增持", "减持", "股权变动", "持股变动"]),
    (10, ["定向增发", "非公开发行", "配股", "募资"]),
    (11, ["业绩预告", "业绩快报", "盈利预测"]),
    (12, ["诉讼", "仲裁", "行政处罚", "立案"]),
    (13, ["董事长", "总经理", "高管", "人事变动"]),
    (14, ["监管", "问询函", "关注函", "交易所"]),
    (50, ["公告", "通知", "披露"]),   # general announcements
    (99, []),                          # catch-all
]

def get_importance_rank(title: str) -> int:
    """Infer importance rank from announcement title keywords (lower = more important)."""
    if not title:
        return 99
    for rank, keywords in TITLE_IMPORTANCE_RULES:
        if any(kw in title for kw in keywords):
            return rank
    return 99

# ──────────────────────────────────────────────
# Source credibility: EastMoney outlet -> source_rank
# ──────────────────────────────────────────────
SOURCE_RANK_MAP = {
    "中国证券报": 1, "上海证券报": 1, "证券时报": 1, "证券日报": 1,
    "人民日报": 1, "新华社": 1, "中央电视台": 1,
    "21世纪经济报道": 2, "第一财经": 2, "财联社": 2, "界面新闻": 2,
    "华尔街见闻": 2, "Wind资讯": 2,
}

def get_source_rank(source: str) -> int:
    if not source:
        return 5
    for k, v in SOURCE_RANK_MAP.items():
        if k in source:
            return v
    return 3


# ──────────────────────────────────────────────
# Database initialization
# ──────────────────────────────────────────────
def init_db(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS announcements (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        SecuCode         TEXT    NOT NULL,
        stock_name       TEXT,
        title            TEXT    NOT NULL,
        announce_dt      TEXT    NOT NULL,
        url              TEXT,
        importance_rank  INTEGER,   -- lower = more important
        segment_start    TEXT,      -- collection batch start date
        segment_end      TEXT,
        UNIQUE (SecuCode, announce_dt, title)
    );

    CREATE TABLE IF NOT EXISTS news_em (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        SecuCode        TEXT    NOT NULL,
        stock_name      TEXT,
        title           TEXT    NOT NULL,
        content         TEXT,       -- content snippet for FinBERT
        news_dt         TEXT,
        source          TEXT,
        url             TEXT,
        source_rank     INTEGER,    -- lower = more credible
        UNIQUE (SecuCode, news_dt, title)
    );

    CREATE TABLE IF NOT EXISTS stocks (
        SecuCode   TEXT PRIMARY KEY,
        name       TEXT,
        market     TEXT
    );

    CREATE TABLE IF NOT EXISTS meta (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        run_time             TEXT,
        total_stocks         INTEGER,
        ann_rows_inserted    INTEGER,
        news_rows_inserted   INTEGER,
        ann_errors           INTEGER,
        notes                TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_ann_code    ON announcements (SecuCode);
    CREATE INDEX IF NOT EXISTS idx_ann_dt      ON announcements (announce_dt);
    CREATE INDEX IF NOT EXISTS idx_ann_rank    ON announcements (importance_rank);
    CREATE INDEX IF NOT EXISTS idx_news_code   ON news_em (SecuCode);
    CREATE INDEX IF NOT EXISTS idx_news_dt     ON news_em (news_dt);
    """)
    conn.commit()
    print(f"[DB] Initialized: {DB_PATH}")


# ──────────────────────────────────────────────
# CNINFO announcement collection (6-month segments)
# ──────────────────────────────────────────────
def fetch_cninfo_segment(code: str, start: str, end: str, retries: int = 3) -> Optional[pd.DataFrame]:
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_disclosure_report_cninfo(
                symbol=code,
                market=MARKET_PARAM,
                keyword="",
                category="",
                start_date=start,
                end_date=end,
            )
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 + attempt * 2)
            else:
                return None
    return None


def process_cninfo_df(df: pd.DataFrame, code: str, seg_start: str, seg_end: str) -> pd.DataFrame:
    """Clean CNINFO DataFrame and compute importance_rank field."""
    if df is None or df.empty:
        return pd.DataFrame()

    cols = list(df.columns)  # ['code', 'name', 'title', 'datetime', 'url']
    code_col  = cols[0]
    name_col  = cols[1]
    title_col = cols[2]
    dt_col    = cols[3]
    url_col   = cols[4] if len(cols) > 4 else None

    out = pd.DataFrame()
    out["SecuCode"]      = df[code_col].astype(str).str.strip()
    out["stock_name"]    = df[name_col].astype(str).str.strip()
    out["title"]         = df[title_col].astype(str).str.strip()
    out["announce_dt"]   = pd.to_datetime(df[dt_col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    out["url"]           = df[url_col].astype(str).str.strip() if url_col else ""
    out["importance_rank"] = out["title"].apply(get_importance_rank)
    out["segment_start"] = seg_start
    out["segment_end"]   = seg_end

    out = out.dropna(subset=["title", "announce_dt"])
    out = out[out["title"].str.len() > 0]
    return out


def upsert_announcements(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    sql = """
    INSERT OR IGNORE INTO announcements
        (SecuCode, stock_name, title, announce_dt, url, importance_rank, segment_start, segment_end)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = df[["SecuCode","stock_name","title","announce_dt","url","importance_rank","segment_start","segment_end"]
              ].itertuples(index=False, name=None)
    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    return cur.rowcount


# ──────────────────────────────────────────────
# EastMoney news collection
# ──────────────────────────────────────────────
def fetch_em_news(code: str, retries: int = 3) -> Optional[pd.DataFrame]:
    for attempt in range(retries):
        try:
            df = ak.stock_news_em(symbol=code)
            return df
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 + attempt)
            else:
                return None
    return None


def process_em_news(df: pd.DataFrame, code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    cols = list(df.columns)  # ['keyword', 'title', 'content', 'datetime', 'source', 'url']
    title_col   = cols[1]
    content_col = cols[2]
    dt_col      = cols[3]
    source_col  = cols[4]
    url_col     = cols[5] if len(cols) > 5 else None
    name_col    = cols[0]  # keyword = stock name

    out = pd.DataFrame()
    out["SecuCode"]    = code
    out["stock_name"]  = df[name_col].astype(str).str.strip()
    out["title"]       = df[title_col].astype(str).str.strip()
    out["content"]     = df[content_col].astype(str).str.strip()
    out["news_dt"]     = pd.to_datetime(df[dt_col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    out["source"]      = df[source_col].astype(str).str.strip()
    out["url"]         = df[url_col].astype(str).str.strip() if url_col else ""
    out["source_rank"] = out["source"].apply(get_source_rank)

    out = out.dropna(subset=["title", "news_dt"])
    out = out[out["title"].str.len() > 0]
    return out


def upsert_news(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    sql = """
    INSERT OR IGNORE INTO news_em
        (SecuCode, stock_name, title, content, news_dt, source, url, source_rank)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = df[["SecuCode","stock_name","title","content","news_dt","source","url","source_rank"]
              ].itertuples(index=False, name=None)
    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    return cur.rowcount


# ──────────────────────────────────────────────
# Stock basic info seed
# ──────────────────────────────────────────────
def seed_stocks(conn: sqlite3.Connection, stocks: list):
    rows = []
    for code in stocks:
        if code.startswith("6"):
            mkt = "SH"
        elif code.startswith("3"):
            mkt = "CYB"
        else:
            mkt = "SZ"
        rows.append((code, "", mkt))
    conn.executemany(
        "INSERT OR IGNORE INTO stocks (SecuCode, name, market) VALUES (?,?,?)",
        rows
    )
    conn.commit()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 70)
    print("A-Share Sentiment / News Raw Text Collection (for FinBERT/NolBERT)")
    print(f"Stocks: {len(STOCKS)}  |  Period: 2018-01-01 ~ 2025-12-31")
    print(f"Output: {DB_PATH}")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    seed_stocks(conn, STOCKS)

    ann_total  = 0
    ann_errors = 0
    news_total = 0

    # 1. CNINFO official announcements (2018-2025, 6-month segments)
    print(f"\n[1/2] CNINFO Announcement Collection")
    print(f"  Strategy: {len(STOCKS)} stocks x {len(DATE_SEGMENTS)} segments = "
          f"{len(STOCKS)*len(DATE_SEGMENTS)} requests")
    print(f"  Ranking field: importance_rank (annual report=1, major events=4, general=99)")

    total_requests = len(STOCKS) * len(DATE_SEGMENTS)
    pbar = tqdm(total=total_requests, desc="CNINFO announcements", ascii=True)

    for code in STOCKS:
        for (seg_start, seg_end) in DATE_SEGMENTS:
            df_raw   = fetch_cninfo_segment(code, seg_start, seg_end)
            df_clean = process_cninfo_df(df_raw, code, seg_start, seg_end)

            if not df_clean.empty:
                upsert_announcements(conn, df_clean)
                ann_total += len(df_clean)
            else:
                ann_errors += 1

            pbar.update(1)
            time.sleep(0.5)   # polite rate limiting

    pbar.close()
    print(f"  CNINFO: {ann_total} rows written (duplicates ignored), {ann_errors} empty responses")

    # 2. EastMoney news (recent, includes content snippets)
    print(f"\n[2/2] EastMoney News Collection (recent raw text + content snippets)")
    print(f"  Note: free API returns ~10 most recent items per stock")

    for code in tqdm(STOCKS, desc="EastMoney news", ascii=True):
        df_raw   = fetch_em_news(code)
        df_clean = process_em_news(df_raw, code)
        if not df_clean.empty:
            upsert_news(conn, df_clean)
            news_total += len(df_clean)
        time.sleep(0.4)

    print(f"  EastMoney: {news_total} rows written")

    # Summary statistics
    ann_db  = conn.execute("SELECT COUNT(*) FROM announcements").fetchone()[0]
    news_db = conn.execute("SELECT COUNT(*) FROM news_em").fetchone()[0]

    ann_date = conn.execute(
        "SELECT MIN(announce_dt), MAX(announce_dt) FROM announcements"
    ).fetchone()
    ann_by_rank = conn.execute(
        "SELECT importance_rank, COUNT(*) FROM announcements "
        "GROUP BY importance_rank ORDER BY importance_rank LIMIT 8"
    ).fetchall()

    print("\n" + "=" * 70)
    print("Collection Complete — Database Statistics")
    print(f"  announcements : {ann_db:>8,} rows  |  {ann_date[0]} ~ {ann_date[1]}")
    print(f"  news_em       : {news_db:>8,} rows")
    print(f"\n  Announcement distribution by importance_rank:")
    rank_labels = {
        1:"Annual Report", 2:"Semi-annual", 3:"Quarterly", 4:"Major Event",
        5:"Equity Incentive", 6:"Dividend", 7:"Shareholder Meeting", 8:"M&A",
        9:"Shareholding Change", 10:"Private Placement", 11:"Earnings Forecast",
        12:"Litigation", 13:"Management Change", 14:"Regulatory", 50:"General", 99:"Other"
    }
    for rank, cnt in ann_by_rank:
        label = rank_labels.get(rank, f"rank={rank}")
        print(f"    rank={rank:>2}  {label:<22}  {cnt:>6,} rows")

    print(f"\n  Database path: {DB_PATH}")
    print("\n  Next steps:")
    print("  1. Sort by importance_rank ASC to process high-impact announcements first")
    print("  2. Run FinBERT/NolBERT on 'title' and 'content' fields for sentiment")
    print("  3. CNINFO URLs can be used to download full PDF filings (src/cninfo/pdf_text.py)")
    print("=" * 70)

    # Write metadata record
    conn.execute(
        "INSERT INTO meta (run_time,total_stocks,ann_rows_inserted,news_rows_inserted,ann_errors,notes) "
        "VALUES (?,?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), len(STOCKS), ann_total, news_total, ann_errors,
         f"akshare {ak.__version__} | CNINFO+EastMoney | 2018-2025 | 6-month segments")
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
