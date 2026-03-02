"""Universe validation helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_universe(df: pd.DataFrame, universe_set: Iterable[str], col: str = "SecuCode") -> None:
    """Raise if ``df[col]`` contains codes outside ``universe_set``.

    Expects normalized 6-digit SecuCode strings.
    """
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

    allowed = set(universe_set)
    present = set(df[col].dropna().astype(str).tolist())
    unexpected = sorted(present - allowed)

    if unexpected:
        preview = ", ".join(unexpected[:20])
        more = "" if len(unexpected) <= 20 else f" ... (+{len(unexpected) - 20} more)"
        raise ValueError(
            f"Found {len(unexpected)} unexpected SecuCode values in '{col}': {preview}{more}"
        )
