# Adaptive Multi-Factor Aggregation with CNINFO Sentiment Context on CSI 300 (15-Minute Horizon)

## Executive Summary
This capstone project studies how to improve short-horizon equity signal aggregation in the CSI 300 universe by combining price-derived factors with disclosure-based sentiment context. The core problem is not only factor construction, but dynamic weighting: in real markets, the usefulness of a factor can vary across volatility regimes, liquidity conditions, and information cycles. Static averaging of signals can therefore underperform when market structure changes intraday.

Adaptive signal weighting is central because it frames portfolio decision-making as a state-dependent allocation problem: given a set of candidate factors, the system should learn when each factor should receive more or less weight. Reinforcement learning is relevant at this stage because it can model sequential decisions under uncertainty, using realized outcomes to update a policy that maps market state features to aggregation weights.

Within this framework, sentiment analysis is not treated as a standalone trading model. Instead, it is developed as contextual regime information derived from CNINFO public announcements. The sentiment pipeline extracts and scores announcement text, then aligns events to leakage-safe 15-minute bar timestamps so that the aggregation layer can consume sentiment features without look-ahead bias. This repository currently focuses on building that reproducible sentiment data pipeline and integration scaffolding; policy learning and full CAM/RL evaluation are active development components rather than completed deliverables.

## System Architecture Overview
- **15-minute price panel**: intraday bar data for CSI 300 constituents serves as the common temporal backbone.
- **Factor generation (`Factor.py`)**: existing factor logic produces candidate market signals from price/volume data.
- **Sentiment pipeline**: CNINFO metadata collection (placeholder/expandable) -> PDF download -> text extraction (PyMuPDF with pdfplumber fallback) -> lexicon scoring -> alignment to the next valid 15-minute bar.
- **Aggregation layer (CAM / RL policy)**: consumes factor signals and contextual features (including sentiment-derived state inputs) to learn adaptive weighting rules. This layer is conceptually defined and under iterative implementation/evaluation.

## Current Repository Structure
- **`notebooks/`**: interactive workflow notebooks, currently `price_factor_research.ipynb` (factor exploration) and `sentiment_pipeline_mvp.ipynb` (sentiment MVP run path).
- **`src/`**: reusable pipeline modules (`src/cninfo`) for I/O, metadata loading/scraping stubs, PDF processing, sentiment scoring, and event-bar alignment.
- **`data/`**: staged local artifacts (`raw/`, `interim/`, `processed/`), intentionally gitignored to prevent accidental data commits.
- **`Factor.py`**: existing factor-generation script retained as-is.
Place `announcements_meta_*.csv` under `data/raw/meta/` (ignored by git), or pass an absolute path via `--meta_csv`.

Completed currently: repository hygiene, modular sentiment MVP skeleton, and notebook wiring for reproducible local runs.  
In progress: broader data coverage, production-grade CNINFO ingestion, and full CAM/RL integration/testing.

## MVP Scope (Current Milestone)
This MVP is scoped to staged, leakage-aware data engineering and preliminary validation:
- Build a deterministic sentiment pipeline with explicit intermediate outputs.
- Align announcement-derived features to next-available 15-minute bars to avoid look-ahead leakage.
- Evaluate predictive validity using event-study style analysis and/or information coefficient (IC) diagnostics as validation steps.
- Prioritize reproducibility through modular code, cached artifacts, and clearly defined inputs/outputs before model complexity is expanded.

## Ethical and Data Considerations
- CNINFO is a public corporate disclosure platform; this project uses publicly available announcements only.
- Data collection should respect platform terms and operational limits, including request rate limiting.
- The workflow is designed for compliant research use and does not include bypassing access controls or restrictions.
- No private, proprietary, or restricted personal data is required for the current pipeline.

## Roadmap
- Expand sentiment coverage window to 2018-2025 with stable metadata and text quality controls.
- Integrate sentiment-derived regime features into CAM state representations.
- Evaluate whether factor weights shift systematically across sentiment/market regimes.
- Run robustness checks across subperiods, sectors, and alternative sentiment specifications.

## Sentiment MVP (Dec-2025 Meta Only)
- Required inputs:
  - Meta CSV: `data/raw/meta/announcements_meta_2025-12.csv`
  - Fixture PDFs in one folder, named exactly `<ann_id>.pdf` (for example `1224882117.pdf`)
- Run command:
  - `.venv/bin/python scripts/sentiment_mvp.py --meta_csv data/raw/meta/announcements_meta_2025-12.csv --fixtures_dir data/raw/fixtures/pdfs --subset_mode existing_pdfs`
  - `.venv/bin/python scripts/sentiment_mvp.py --meta_csv data/raw/meta/announcements_meta_2025-12.csv --fixtures_dir data/raw/fixtures/pdfs --subset_mode first_n --secu_code 000001 --n 20`
- `--subset_mode`:
  - `first_n` (default): filter by `--secu_code`, take first `--n`, fail-fast if PDFs are missing.
  - `existing_pdfs`: build subset directly from `fixtures_dir/*.pdf` ann_ids for deterministic end-to-end fixture runs.
- Optional alignment preview:
  - Add `--align_15m` and (optionally) `--bars_calendar_file <path>`
  - If bars/calendar file is missing, script prints: `alignment skipped: missing bars calendar file <path>`
- Artifacts written:
  - `data/interim/meta_mvp.csv`
  - `data/interim/texts_mvp.parquet`
  - `data/interim/texts_mvp.csv`
  - `data/processed/sentiment_mvp.parquet`
  - `data/processed/sentiment_mvp.csv`
- Quick manual verification:
  - `.venv/bin/python -c "import pandas as pd; print(pd.read_csv('data/interim/meta_mvp.csv').head(3))"`
  - `.venv/bin/python -c "import pandas as pd; print(pd.read_csv('data/interim/texts_mvp.csv')[['ann_id','text_len','extraction_ok']].head(3))"`
  - `.venv/bin/python -c "import pandas as pd; print(pd.read_csv('data/processed/sentiment_mvp.csv')[['ann_id','sentiment_score','sentiment_label','model_backend']].head(3))"`
- Troubleshooting:
  - Missing PDFs: rename/copy fixture files into the fixtures directory as `<ann_id>.pdf`.
  - Short extracted text: inspect source PDF quality/content; extraction still keeps row with `extraction_ok=False` if needed.
  - `image_only_pdf`: PDF opens but has no selectable text (scanned/image-based); excluded from sentiment by default unless `--include_failed_in_sentiment` is set.
  - `transformers` missing or unavailable: script auto-falls back to built-in lexicon baseline.

## Handling Extraction Failures
- Triage only (writes `data/interim/extraction_failures_report.csv`, no scoring):
  - `.venv/bin/python scripts/sentiment_mvp.py --meta_csv data/raw/meta/announcements_meta_2025-12.csv --fixtures_dir data/raw/fixtures/pdfs --subset_mode existing_pdfs --triage_only`
- Re-extract only failed announcement IDs from existing `data/interim/texts_mvp.csv`:
  - `.venv/bin/python scripts/sentiment_mvp.py --fixtures_dir data/raw/fixtures/pdfs --reextract_failed`
- OCR is intentionally not included in MVP; unresolved failures remain flagged via `extraction_ok` and `extraction_error`.
- Sentiment excludes extraction failures by default; use `--include_failed_in_sentiment` to include them.
