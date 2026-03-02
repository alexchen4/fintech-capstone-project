#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = ROOT / "data" / "processed" / "sentiment" / "audits"


def _infer_month_tag(paths: list[Path]) -> str:
    for p in paths:
        m = re.search(r"(\d{4}-\d{2})", p.name)
        if m:
            return m.group(1)
    return "unknown"


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported artifact format: {path}")


def _stable_hash(lines: list[str]) -> str:
    joined = "\n".join(sorted(lines))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _hash_ann_id_set(df: pd.DataFrame) -> tuple[int, str]:
    if "ann_id" not in df.columns:
        return 0, ""
    vals = df["ann_id"].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    uniq = sorted(set(vals.tolist()))
    return len(uniq), _stable_hash(uniq)


def _hash_pair(df: pd.DataFrame, c1: str, c2: str) -> tuple[int, str]:
    if c1 not in df.columns or c2 not in df.columns:
        return 0, ""
    a = df[c1].fillna("").astype(str).str.strip()
    b = df[c2].fillna("").astype(str).str.strip()
    pair = [f"{x}|{y}" for x, y in zip(a.tolist(), b.tolist()) if x and y]
    uniq = sorted(set(pair))
    return len(uniq), _stable_hash(uniq)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute stable fingerprints for sentiment artifacts.")
    p.add_argument("--paths", nargs="+", type=Path, required=True, help="One or more parquet/csv artifact paths")
    p.add_argument("--month", default=None, help="Optional month tag YYYY-MM")
    p.add_argument("--out_json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    month = args.month or _infer_month_tag(args.paths)
    out_json = args.out_json or (DEFAULT_AUDIT_DIR / f"fingerprint_{month}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {"month": month, "artifacts": []}

    print("[fingerprint_artifacts] summary")
    print(f"month={month}")

    for p in args.paths:
        df = _load_table(p)
        ann_n, ann_hash = _hash_ann_id_set(df)
        pair_ab_n, pair_ab_hash = _hash_pair(df, "ann_id", "t_event_bar")
        pair_sa_n, pair_sa_hash = _hash_pair(df, "SecuCode", "ann_id")

        item = {
            "path": str(p),
            "rows": int(len(df)),
            "ann_id_set_count": ann_n,
            "ann_id_set_hash": ann_hash,
            "ann_id_t_event_bar_pair_count": pair_ab_n,
            "ann_id_t_event_bar_pair_hash": pair_ab_hash,
            "SecuCode_ann_id_pair_count": pair_sa_n,
            "SecuCode_ann_id_pair_hash": pair_sa_hash,
        }
        payload["artifacts"].append(item)

        print(f"path={p}")
        print(f"rows={item['rows']}")
        print(f"ann_id_set_count={ann_n}")
        print(f"ann_id_set_hash={ann_hash}")
        print(f"ann_id_t_event_bar_pair_count={pair_ab_n}")
        print(f"ann_id_t_event_bar_pair_hash={pair_ab_hash}")
        print(f"SecuCode_ann_id_pair_count={pair_sa_n}")
        print(f"SecuCode_ann_id_pair_hash={pair_sa_hash}")

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_json={out_json}")


if __name__ == "__main__":
    main()
