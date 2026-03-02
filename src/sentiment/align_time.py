"""Announcement-to-15m bar alignment utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from common.secu import normalize_secu_series
from common.universe import validate_universe

CN_TZ = "Asia/Shanghai"


def _parse_time_start_hhmm(x: object) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.isdigit():
        s = s.zfill(4)
    if len(s) != 4 or not s.isdigit():
        return None
    hh = int(s[:2])
    mm = int(s[2:])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return f"{hh:02d}:{mm:02d}:00"


def _bar_ts_local(trading_day: object, time_start: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(trading_day) or pd.isna(time_start):
        return pd.NaT
    day = str(trading_day).strip()
    hhmmss = _parse_time_start_hhmm(time_start)
    if not day or hhmmss is None:
        return pd.NaT
    if len(day) != 8 or not day.isdigit():
        return pd.NaT
    dt = datetime.strptime(f"{day} {hhmmss}", "%Y%m%d %H:%M:%S")
    return pd.Timestamp(dt, tz=CN_TZ)


def build_universe_bar_index(price_csv: Path, universe_set: set[str], chunksize: int = 500_000) -> pd.DataFrame:
    """Load tradable 15m bar timestamps from source-of-truth price CSV for universe codes only."""
    parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        price_csv,
        usecols=["SecuCode", "TradingDay", "TimeStart"],
        dtype={"SecuCode": str, "TradingDay": str, "TimeStart": str},
        chunksize=chunksize,
    ):
        chunk = chunk.copy()
        chunk["SecuCode"] = normalize_secu_series(chunk["SecuCode"])
        chunk = chunk[chunk["SecuCode"].isin(universe_set)]
        if chunk.empty:
            continue

        chunk["bar_ts_local"] = [
            _bar_ts_local(d, t) for d, t in zip(chunk["TradingDay"].tolist(), chunk["TimeStart"].tolist())
        ]
        chunk = chunk.dropna(subset=["bar_ts_local"])
        parts.append(chunk[["SecuCode", "bar_ts_local"]])

    if not parts:
        return pd.DataFrame(columns=["SecuCode", "bar_ts_local"])

    bars = pd.concat(parts, ignore_index=True)
    bars = bars.drop_duplicates(subset=["SecuCode", "bar_ts_local"], keep="first")
    bars = bars.sort_values(["SecuCode", "bar_ts_local"], kind="stable").reset_index(drop=True)
    return bars


def align_events_to_next_bar(events_df: pd.DataFrame, bars_df: pd.DataFrame) -> pd.DataFrame:
    """Map each event to first tradable 15m bar strictly after publish time.

    publish_dt_utc -> converted to Asia/Shanghai.
    t_event_bar stays in Asia/Shanghai timezone.
    """
    out = events_df.copy()
    out["publish_dt_utc"] = pd.to_datetime(out["publish_dt_utc"], errors="coerce", utc=True)
    out["publish_dt_local"] = out["publish_dt_utc"].dt.tz_convert(CN_TZ)
    out["t_event_bar"] = pd.Series(
        pd.NaT,
        index=out.index,
        dtype=f"datetime64[ns, {CN_TZ}]",
    )

    for secu, grp in out.groupby("SecuCode", sort=False):
        bar_grp = bars_df[bars_df["SecuCode"] == secu]
        if bar_grp.empty:
            continue

        bar_ts = pd.to_datetime(bar_grp["bar_ts_local"], errors="coerce")
        bar_values = bar_ts.to_numpy()
        idx = grp.index

        event_ts = pd.to_datetime(out.loc[idx, "publish_dt_local"], errors="coerce")
        pos = bar_ts.searchsorted(event_ts, side="right")

        mapped: list[pd.Timestamp | pd.NaT] = []
        for p in pos.tolist():
            if 0 <= int(p) < len(bar_values):
                mapped.append(bar_values[int(p)])
            else:
                mapped.append(pd.NaT)
        mapped_ser = pd.to_datetime(pd.Series(mapped, index=idx), errors="coerce", utc=True).dt.tz_convert(CN_TZ)
        out.loc[idx, "t_event_bar"] = mapped_ser

    return out


def load_universe_set(universe_csv: Path) -> set[str]:
    uni = pd.read_csv(universe_csv)
    col = "SecuCode" if "SecuCode" in uni.columns else uni.columns[0]
    vals = normalize_secu_series(uni[col]).dropna().astype(str)
    out = set(vals.tolist())
    if len(out) != 50:
        raise ValueError(f"Universe must contain exactly 50 codes, got {len(out)} from {universe_csv}")
    return out


def validate_events_universe(events_df: pd.DataFrame, universe_set: set[str]) -> None:
    validate_universe(events_df, universe_set, col="SecuCode")
