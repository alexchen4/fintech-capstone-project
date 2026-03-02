# Sentiment Pipeline (Safe Phases)

## Conventions
- Price data source of truth: `data/qfq_15min_all.csv`.
- Universe is strict: use only the exact 50 target `SecuCode` values provided by the user.
- `SecuCode` must be normalized as a 6-digit zero-padded string across all datasets (price + CNINFO).

## Folder Structure
- `data/universe/`
- `data/raw/cninfo_meta/`
- `data/raw/cninfo_payload/`
- `data/processed/cninfo_text/`
- `data/processed/sentiment/`

## Outputs by Phase
- Phase 0:
  - `src/common/secu.py` normalization helpers.
  - Directory layout above.
- Phase 1:
  - `data/universe/universe_price.csv`
  - `data/universe/universe_target.csv`
  - `data/universe/universe.csv`
- Phase 2:
  - `data/raw/cninfo_meta/announcements_meta_YYYY-MM.parquet`
- Phase 3:
  - `data/raw/cninfo_payload/raw_pdfs/{ann_id}.pdf`
  - `data/raw/cninfo_payload/failures_YYYY-MM.csv`
- Phase 4:
  - `data/processed/cninfo_text/ann_text_YYYY-MM.parquet`
- Phase 5:
  - Deterministic event-bar alignment fields (`t_event_bar`) added for announcement rows.
- Phase 6:
  - `data/processed/sentiment/sentiment_dataset_YYYY-MM.parquet`
- Phase 7 (optional scaffold):
  - Return-based weak labels joined to announcement rows.
- Phase 8:
  - 15-min sentiment feature panel per `(SecuCode, bar_ts)`.
