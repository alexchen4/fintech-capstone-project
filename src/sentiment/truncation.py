"""Deterministic text truncation policy for model input construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TruncationResult:
    text: str
    text_len_chars: int
    was_truncated: bool
    title_used: str
    body_used: str


def truncate_text(
    title: object,
    body: object,
    max_chars_total: int = 4000,
    max_chars_body: int = 3500,
) -> TruncationResult:
    """Build model text with title-preserving truncation.

    Final text format:
        "[CLS] {title} [SEP] {body} [SEP]"
    """
    if max_chars_total <= 0:
        raise ValueError("max_chars_total must be > 0")
    if max_chars_body < 0:
        raise ValueError("max_chars_body must be >= 0")

    t = "" if title is None else str(title).strip()
    b = "" if body is None else str(body).strip()

    prefix = "[CLS] "
    mid = " [SEP] "
    suffix = " [SEP]"
    overhead = len(prefix) + len(mid) + len(suffix)

    # First pass: cap body by policy max.
    body_used = b[:max_chars_body]
    title_used = t

    # Second pass: enforce global max while preserving title as much as possible.
    allowed_for_title_and_body = max(0, max_chars_total - overhead)
    if len(title_used) > allowed_for_title_and_body:
        title_used = title_used[:allowed_for_title_and_body]
        body_used = ""
    else:
        allowed_body = max(0, allowed_for_title_and_body - len(title_used))
        if len(body_used) > allowed_body:
            body_used = body_used[:allowed_body]

    text = f"{prefix}{title_used}{mid}{body_used}{suffix}"
    if len(text) > max_chars_total:
        # Final defensive trim on body only.
        extra = len(text) - max_chars_total
        body_used = body_used[: max(0, len(body_used) - extra)]
        text = f"{prefix}{title_used}{mid}{body_used}{suffix}"

    was_truncated = (title_used != t) or (body_used != b)
    return TruncationResult(
        text=text,
        text_len_chars=len(text),
        was_truncated=was_truncated,
        title_used=title_used,
        body_used=body_used,
    )
