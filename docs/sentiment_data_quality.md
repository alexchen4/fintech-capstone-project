# Sentiment Data Quality Gate

## Definition: Complete Text Data
A month is considered model-ready when all of the following hold:
- Data contracts pass for universe/meta/text/dataset artifacts.
- Text parsing quality is stable (high parse success, low empty-text ratio).
- Alignment is leakage-safe (`t_event_bar` after publish time) with no negative delays.
- Price join keys `(SecuCode, TradingDay, TimeStart)` exist for aligned rows.
- Fingerprints are stable across repeated runs from same inputs.

## Scripts to Run
1. `python scripts/validate_outputs.py --month YYYY-MM --universe_csv data/universe/universe.csv`
2. `python scripts/audit_text_quality.py --input_path data/processed/sentiment/sentiment_dataset_YYYY-MM.parquet --universe_csv data/universe/universe.csv --sample_n 60 --seed 42`
3. `python scripts/audit_alignment_and_price_join.py --dataset_parquet data/processed/sentiment/sentiment_dataset_YYYY-MM.parquet --price_csv data/qfq_15min_all.csv`
4. `python scripts/fingerprint_artifacts.py --paths data/processed/sentiment/sentiment_dataset_YYYY-MM.parquet data/processed/cninfo_text/ann_text_YYYY-MM.parquet --month YYYY-MM`

## Recommended Healthy Thresholds
- `validate_outputs`: must PASS.
- `parse_ok_rate`: >= 95%
- `empty_text_rate`: <= 5%
- `negative_delay_count`: 0
- `t_event_bar_non_null`: >= 99%
- `join_coverage`: >= 99%
- Fingerprints: hashes should match across repeated runs with unchanged inputs.

## Notes
- Use `data/qfq_15min_all.csv` as the only price source-of-truth.
- Always normalize `SecuCode` to 6-digit zero-padded strings.
- Keep audit sample generation deterministic using fixed random seed.
