"""
This module downloads announcement PDFs and extracts text content for scoring.
It forms the document-processing stage between metadata and sentiment features.
PyMuPDF is used first, with pdfplumber fallback for robustness without OCR.
Status: MVP-ready baseline, suitable for iterative quality and coverage upgrades.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import fitz
import pandas as pd
import pdfplumber
import requests


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def download_pdfs(
    meta_df: pd.DataFrame,
    pdf_dir: str,
    url_col: str = "pdf_url",
    id_col: str = "announcement_id",
) -> pd.DataFrame:
    """Download PDFs and return a manifest with cache-aware metadata.

    Output includes original metadata plus:
    - local_path
    - sha1
    - http_status
    """
    out_dir = Path(pdf_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in meta_df.iterrows():
        rec = row.to_dict()
        ann_id = str(rec.get(id_col) or "")
        url = rec.get(url_col)

        local_path = None
        sha1 = None
        status = None

        if ann_id and ann_id.lower() != "nan":
            local_path = out_dir / f"{ann_id}.pdf"
        elif isinstance(url, str) and url.strip():
            sha1_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
            local_path = out_dir / f"{sha1_id}.pdf"

        if local_path is None:
            rec.update({"local_path": None, "sha1": None, "http_status": None})
            records.append(rec)
            continue

        if local_path.exists():
            data = local_path.read_bytes()
            sha1 = _sha1_bytes(data)
            status = 200
            rec.update({"local_path": str(local_path), "sha1": sha1, "http_status": status})
            records.append(rec)
            continue

        if not isinstance(url, str) or not url.strip():
            rec.update({"local_path": str(local_path), "sha1": None, "http_status": None})
            records.append(rec)
            continue

        try:
            resp = requests.get(url, timeout=30)
            status = resp.status_code
            if resp.ok and resp.content:
                data = resp.content
                local_path.write_bytes(data)
                sha1 = _sha1_bytes(data)
            rec.update({"local_path": str(local_path), "sha1": sha1, "http_status": status})
        except requests.RequestException:
            rec.update({"local_path": str(local_path), "sha1": None, "http_status": None})

        records.append(rec)

    return pd.DataFrame(records)


def extract_text_from_pdf(path: str) -> Dict[str, Any]:
    """Extract PDF text via PyMuPDF first, then pdfplumber fallback.

    Returns keys: text, method, char_count, page_count, error.
    """
    result: Dict[str, Any] = {
        "text": "",
        "method": None,
        "char_count": 0,
        "page_count": 0,
        "error": None,
    }

    pdf_path = Path(path)
    if not pdf_path.exists():
        result["error"] = "file_not_found"
        return result

    try:
        with fitz.open(pdf_path) as doc:
            pages = [page.get_text("text") or "" for page in doc]
            text = "\n".join(pages).strip()
            if text:
                result.update(
                    {
                        "text": text,
                        "method": "pymupdf",
                        "char_count": len(text),
                        "page_count": len(doc),
                        "error": None,
                    }
                )
                return result
    except Exception as exc:
        result["error"] = f"pymupdf_error:{type(exc).__name__}"

    try:
        with pdfplumber.open(pdf_path) as doc:
            pages = [(page.extract_text() or "") for page in doc.pages]
            text = "\n".join(pages).strip()
            result.update(
                {
                    "text": text,
                    "method": "pdfplumber",
                    "char_count": len(text),
                    "page_count": len(doc.pages),
                    "error": None if text else (result["error"] or "empty_text"),
                }
            )
            return result
    except Exception as exc:
        result["error"] = f"{result['error'] or ''};pdfplumber_error:{type(exc).__name__}".strip(";")

    return result


def batch_extract_text(pdf_manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Batch extract text using ``local_path`` from a PDF manifest DataFrame."""
    records = []
    for _, row in pdf_manifest_df.iterrows():
        rec = row.to_dict()
        local_path: Optional[str] = rec.get("local_path")
        if isinstance(local_path, str) and local_path.strip():
            extracted = extract_text_from_pdf(local_path)
        else:
            extracted = {
                "text": "",
                "method": None,
                "char_count": 0,
                "page_count": 0,
                "error": "missing_local_path",
            }
        rec.update(extracted)
        records.append(rec)

    return pd.DataFrame(records)
