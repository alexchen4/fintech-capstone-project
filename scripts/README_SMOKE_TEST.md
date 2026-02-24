# Sentiment MVP Smoke Test

This smoke test validates a minimal local run of the CNINFO sentiment pipeline on a small sample (`N=5`) without requiring full scraping.

## What it checks
- Load local announcement metadata from `data/raw/`.
- Download/cache PDFs (if `pdf_url` values exist).
- Extract text (PyMuPDF first, pdfplumber fallback; no OCR).
- Score sentiment with lexicon baseline.
- Write outputs to `data/interim/` only.

## Required local input
Provide at least one metadata file:
- `data/raw/meta.parquet` or `data/raw/meta.csv`
- or `data/raw/cninfo_meta.parquet` or `data/raw/cninfo_meta.csv`

Expected columns (missing ones are filled as null for test robustness):
`SecuCode`, `publish_dt`, `title`, `category`, `pdf_url`, `announcement_id`

Optional lexicons:
- `data/raw/lexicon_pos.txt`
- `data/raw/lexicon_neg.txt`

If lexicons are absent, the script uses a tiny placeholder lexicon.

## Run
```bash
python scripts/smoke_test_sentiment.py
```

## Outputs (gitignored)
- `data/interim/smoke_pdf_manifest.parquet`
- `data/interim/smoke_texts.parquet`
- `data/interim/smoke_scored.parquet`
