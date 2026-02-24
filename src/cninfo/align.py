"""
This module aligns sentiment events to leakage-safe 15-minute market bars.
It bridges event timestamps and panel-based factor and return datasets.
Mapping uses the next available bar per ticker to avoid look-ahead leakage.
Status: MVP-ready core alignment logic for staged validation workflows.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import pandas as pd


def align_events_to_15m(
    events_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    event_time_col: str = "publish_dt",
    keys: Tuple[str, str, str] = ("SecuCode", "TradingDay", "TimeEnd"),
) -> pd.DataFrame:
    """Map each event timestamp to the next available 15m bar for the same ticker.

    Leakage-safe rule:
    - For each ``SecuCode``, event at time T maps to the first bar where ``TimeEnd >= T``.
    - If event is outside trading hours, this naturally maps to the next available open bar
      because no intermediate bars exist in the ticker's bar series.

    Assumptions:
    - ``bars_df`` contains at least ``SecuCode`` and ``TimeEnd``.
    - ``TimeEnd`` is timezone-consistent with ``publish_dt``.
    - ``bars_df`` includes regular-session bars only.
    """
    secu_col, trading_day_col, time_end_col = keys

    ev = events_df.copy()
    br = bars_df.copy()

    ev[event_time_col] = pd.to_datetime(ev[event_time_col], errors="coerce")
    br[time_end_col] = pd.to_datetime(br[time_end_col], errors="coerce")

    if trading_day_col not in br.columns:
        br[trading_day_col] = br[time_end_col].dt.normalize()

    aligned_parts = []
    for secu_code, ev_grp in ev.groupby(secu_col, dropna=False):
        br_grp = br[br[secu_col] == secu_code].sort_values(time_end_col)
        ev_grp_sorted = ev_grp.sort_values(event_time_col)

        if br_grp.empty:
            empty_aligned = ev_grp_sorted.copy()
            empty_aligned["aligned_time_end"] = pd.NaT
            empty_aligned["aligned_trading_day"] = pd.NaT
            empty_aligned["is_aligned"] = False
            aligned_parts.append(empty_aligned)
            continue

        ev_keyed = ev_grp_sorted.reset_index().rename(columns={"index": "_event_index"})
        br_keyed = br_grp.reset_index(drop=True)

        merged = pd.merge_asof(
            ev_keyed,
            br_keyed,
            left_on=event_time_col,
            right_on=time_end_col,
            direction="forward",
            allow_exact_matches=True,
            suffixes=("", "_bar"),
        )

        # Preserve deterministic event identity and expose chosen bar keys.
        merged["aligned_time_end"] = merged[time_end_col]
        merged["aligned_trading_day"] = merged[trading_day_col]
        merged["is_aligned"] = merged["aligned_time_end"].notna()

        # Drop duplicated bar columns that may appear due to suffixing.
        drop_cols = [c for c in merged.columns if c.endswith("_bar")]
        merged = merged.drop(columns=drop_cols)

        aligned_parts.append(merged)

    aligned = pd.concat(aligned_parts, ignore_index=True) if aligned_parts else ev.copy()

    if "_event_index" in aligned.columns:
        aligned = aligned.sort_values("_event_index").drop(columns=["_event_index"]) 

    return aligned.reset_index(drop=True)
