"""
This module provides filesystem and parquet I/O helpers for the sentiment pipeline.
It is used across raw, interim, and processed staging steps.
The functions keep file handling consistent and easy to test.
Status: MVP-ready utility layer; expected to remain stable with minor extensions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if it does not exist and return its ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_parquet(path: PathLike, **kwargs) -> pd.DataFrame:
    """Read a parquet file into a DataFrame."""
    return pd.read_parquet(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: PathLike, **kwargs) -> None:
    """Write a DataFrame to parquet, creating parent directories as needed."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    df.to_parquet(out_path, index=False, **kwargs)
