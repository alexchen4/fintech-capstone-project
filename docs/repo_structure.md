# Repository Structure (Organized)

## Active Workflow Entrypoints
- `scripts/run_month.py`: single monthly orchestration entrypoint (phases or all).
- `scripts/pre_modeling_gate.py`: full pre-modeling quality gate.
- `scripts/validate_outputs.py`: contract checks only.

## Active Pipeline Scripts
- `scripts/build_universe_from_price.py`
- `scripts/filter_cninfo_meta.py`
- `scripts/fetch_cninfo_payload.py`
- `scripts/parse_cninfo_text.py`
- `scripts/build_sentiment_dataset.py`
- `scripts/label_from_returns.py` (optional scaffold)
- `scripts/build_sentiment_features_15m.py`
- `scripts/audit_text_quality.py`
- `scripts/audit_alignment_and_price_join.py`
- `scripts/fingerprint_artifacts.py`

## Legacy Scripts
Legacy 1d/factor helper scripts are moved to `scripts/legacy/` and are not part of the active sentiment workflow.

## Source Modules
- `src/common/`: shared normalization and universe helpers.
- `src/sentiment/`: meta/payload/text/alignment/truncation logic.
- `src/validation/`: contract checks for artifact schemas.
- `src/cninfo/pdf_text.py`: PDF extraction utility used by parser.

## Data Policy
- Source-of-truth price data: `data/qfq_15min_all.csv`.
- Generated artifacts should not be committed (`data/interim/`, `data/processed/`, generated raw payload/meta parquet outputs).
