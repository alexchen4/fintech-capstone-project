"""
This module implements lexicon-based sentiment scoring on extracted announcement text.
It converts text artifacts into structured event-level sentiment features.
The scoring layer is intended as a transparent baseline before advanced NLP models.
Status: MVP baseline, reproducible and extensible but not a final model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import pandas as pd

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _read_lexicon(path: str) -> Set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Lexicon file not found: {p}")
    terms = {
        line.strip().lower()
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return terms


def load_lexicons(pos_path: str, neg_path: str) -> Tuple[Set[str], Set[str]]:
    """Load positive and negative lexicons as lowercase sets."""
    return _read_lexicon(pos_path), _read_lexicon(neg_path)


def score_lexicon(text: str, pos_set: Set[str], neg_set: Set[str]) -> Dict[str, int]:
    """Score text using lexicon hit counts and net sentiment."""
    tokens = [t.lower() for t in TOKEN_RE.findall(text or "")]
    pos_hits = sum(1 for t in tokens if t in pos_set)
    neg_hits = sum(1 for t in tokens if t in neg_set)
    sent_score = pos_hits - neg_hits
    return {"sent_score": sent_score, "pos_hits": pos_hits, "neg_hits": neg_hits}


def batch_score(
    text_df: pd.DataFrame,
    pos_set: Set[str],
    neg_set: Set[str],
    text_col: str = "text",
) -> pd.DataFrame:
    """Apply lexicon scoring across a DataFrame and append score columns."""
    df = text_df.copy()
    scores = df[text_col].fillna("").map(lambda x: score_lexicon(str(x), pos_set, neg_set))
    score_df = pd.DataFrame(list(scores))
    return pd.concat([df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)
