"""SecuCode normalization helpers.

Examples:
    >>> normalize_secu_code(400)
    '000400'
    >>> normalize_secu_code(' 1 ')
    '000001'
    >>> normalize_secu_series(pd.Series([400, '000001', '  23']))
    0    000400
    1    000001
    2    000023
    dtype: object

Quick assertions:
    >>> assert normalize_secu_code(400) == '000400'
    >>> assert normalize_secu_code('000400') == '000400'
"""

from __future__ import annotations

import pandas as pd


def normalize_secu_code(x: object) -> str:
    """Normalize a SecuCode to a 6-digit zero-padded string."""
    if pd.isna(x):
        raise ValueError("SecuCode cannot be null")
    s = str(x).strip()
    if not s:
        raise ValueError("SecuCode cannot be empty")
    return s.zfill(6)


def normalize_secu_series(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series of SecuCode values to 6-digit strings."""
    return series.map(normalize_secu_code)
