#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.secu import normalize_secu_code
from common.universe import validate_universe

DEFAULT_PRICE_CSV = ROOT / "data" / "qfq_15min_all.csv"
UNIVERSE_DIR = ROOT / "data" / "universe"
UNIVERSE_PRICE_CSV = UNIVERSE_DIR / "universe_price.csv"
UNIVERSE_TARGET_CSV = UNIVERSE_DIR / "universe_target.csv"
UNIVERSE_CSV = UNIVERSE_DIR / "universe.csv"
EXPECTED_TARGET_SIZE = 50


def _parse_cli_target_codes(raw: list[str] | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for item in raw:
        for part in str(item).split(","):
            c = part.strip()
            if c:
                out.append(c)
    return out


def _normalize_codes(values: list[object]) -> list[str]:
    normed = [normalize_secu_code(v) for v in values if str(v).strip()]
    return sorted(set(normed))


def _load_target_codes(target_csv: Path, cli_codes: list[str]) -> list[str]:
    if cli_codes:
        return _normalize_codes(cli_codes)

    if not target_csv.exists():
        raise FileNotFoundError(
            "Target universe file not found. Provide --target_codes or create "
            f"{target_csv} with one column named SecuCode containing exactly 50 codes."
        )

    df = pd.read_csv(target_csv)
    if df.empty:
        raise ValueError(f"{target_csv} is empty. It must contain exactly 50 target SecuCodes.")

    if "SecuCode" in df.columns:
        col = "SecuCode"
    else:
        col = df.columns[0]

    return _normalize_codes(df[col].tolist())


def _collect_price_codes(price_csv: Path, chunksize: int) -> set[str]:
    seen: set[str] = set()
    for chunk in pd.read_csv(price_csv, usecols=["SecuCode"], dtype={"SecuCode": str}, chunksize=chunksize):
        vals = chunk["SecuCode"].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        for v in vals.tolist():
            seen.add(normalize_secu_code(v))
    return seen


def _trading_day_range_for_universe(price_csv: Path, universe_set: set[str], chunksize: int) -> tuple[str | None, str | None]:
    day_min: str | None = None
    day_max: str | None = None

    for chunk in pd.read_csv(
        price_csv,
        usecols=["SecuCode", "TradingDay"],
        dtype={"SecuCode": str, "TradingDay": str},
        chunksize=chunksize,
    ):
        secu = chunk["SecuCode"].dropna().astype(str).str.strip().map(normalize_secu_code)
        sub = chunk.loc[secu.isin(universe_set), ["TradingDay"]].copy()
        if sub.empty:
            continue

        days = sub["TradingDay"].dropna().astype(str).str.strip()
        days = days[days != ""]
        if days.empty:
            continue

        local_min = days.min()
        local_max = days.max()
        day_min = local_min if day_min is None else min(day_min, local_min)
        day_max = local_max if day_max is None else max(day_max, local_max)

    return day_min, day_max


def _write_codes_csv(path: Path, codes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SecuCode": codes}).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict universe from price source-of-truth and target code list.")
    parser.add_argument("--price_csv", type=Path, default=DEFAULT_PRICE_CSV)
    parser.add_argument(
        "--target_codes",
        nargs="*",
        default=None,
        help="Optional target SecuCodes. Accepts space-separated and/or comma-separated values.",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    price_csv = args.price_csv
    if not price_csv.exists():
        raise FileNotFoundError(f"Price CSV not found: {price_csv}")

    cli_codes = _parse_cli_target_codes(args.target_codes)
    target_codes = _load_target_codes(UNIVERSE_TARGET_CSV, cli_codes)

    if len(target_codes) != EXPECTED_TARGET_SIZE:
        raise ValueError(
            f"Target code count must be exactly {EXPECTED_TARGET_SIZE}, got {len(target_codes)}. "
            f"Edit {UNIVERSE_TARGET_CSV} or pass --target_codes."
        )

    price_codes = _collect_price_codes(price_csv, args.chunksize)
    price_codes_sorted = sorted(price_codes)
    _write_codes_csv(UNIVERSE_PRICE_CSV, price_codes_sorted)

    target_set = set(target_codes)
    universe = sorted(price_codes & target_set)

    _write_codes_csv(UNIVERSE_TARGET_CSV, target_codes)
    _write_codes_csv(UNIVERSE_CSV, universe)

    missing_in_price = sorted(target_set - price_codes)

    if len(universe) != EXPECTED_TARGET_SIZE:
        details = [
            f"Intersection size mismatch: expected {EXPECTED_TARGET_SIZE}, got {len(universe)}.",
            f"Missing in price ({len(missing_in_price)}): {missing_in_price}",
        ]
        raise ValueError("\n".join(details))

    validate_universe(pd.DataFrame({"SecuCode": universe}), target_set, col="SecuCode")

    day_min, day_max = _trading_day_range_for_universe(price_csv, set(universe), args.chunksize)

    print("[build_universe_from_price] summary")
    print(f"price_csv={price_csv}")
    print(f"universe_price_count={len(price_codes_sorted)} -> {UNIVERSE_PRICE_CSV}")
    print(f"universe_target_count={len(target_codes)} -> {UNIVERSE_TARGET_CSV}")
    print(f"universe_intersection_count={len(universe)} -> {UNIVERSE_CSV}")
    print(f"trading_day_min={day_min}")
    print(f"trading_day_max={day_max}")


if __name__ == "__main__":
    main()
