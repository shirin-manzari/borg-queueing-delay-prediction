# Borg Results

This directory contains lightweight, publication-safe result summaries for the
Borg queueing-delay prediction pipeline.

The full local pipeline writes generated outputs under `artifacts/`. Those files
can be large or machine-specific, so they are ignored by git. After a local run,
copy only compact summaries and selected figures here.

## Current Status

No completed local run metrics are committed yet. The files in this directory
define the expected publication format and should be replaced or filled after a
verified pipeline run.

## Files

- `run_metadata.txt`: run configuration and reproducibility notes.
- `metrics_summary.json`: compact model, baseline, and evaluation summary.
- `baseline_metrics.txt`: lightweight baseline summary.
- `figures/`: selected publication figures only.

## Do Not Commit

- raw Google trace CSVs;
- DuckDB databases;
- Parquet feature tables;
- trained model artifacts;
- full artifact directories;
- large logs or temporary files.

## Recommended Figure Set

After a successful run, copy a small subset of diagnostic figures, such as:

- predicted vs actual delays;
- residual diagnostics;
- queue-delay target distribution;
- feature-importance summary.
