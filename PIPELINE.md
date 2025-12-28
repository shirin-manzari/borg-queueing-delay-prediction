# Pipeline

This document describes the end-to-end workflow for predicting task queueing
delay from the Google 2011 Borg cluster trace.

The pipeline is designed for large local CSV files. DuckDB handles raw-data
validation and feature construction, then XGBoost trains from the generated
Parquet feature table in streamed batches.

## Local Data Layout

Place the extracted trace CSVs under `extracted/`:

```text
.
|-- extracted/
|   |-- task_events.csv
|   |-- job_events.csv
|   |-- machine_events.csv
|   |-- machine_attributes.csv
|   |-- task_constraints.csv
|   `-- task_usage.csv
|-- artifacts/
`-- scripts/
```

`extracted/` contains local raw data and is ignored by git. `artifacts/` contains
generated reports, features, models, and plots, and is also ignored by git.

## Full Pipeline

Run all stages:

```bash
python -m scripts.run_pipeline
```

The full pipeline runs:

1. raw-data validation;
2. feature-table construction;
3. pre-training analysis;
4. XGBoost training;
5. post-training evaluation.

## Stage 1: Validate Raw Data

Command:

```bash
python -m scripts.validate_raw
```

Optional output path:

```bash
python -m scripts.validate_raw --output artifacts/validation_report_duckdb.txt
```

Reads:

- `extracted/task_events.csv`;
- `extracted/job_events.csv`;
- `extracted/machine_events.csv`;
- `extracted/machine_attributes.csv`;
- `extracted/task_constraints.csv`;
- `extracted/task_usage.csv`.

Writes:

- `artifacts/validation_report_duckdb.txt`.

This stage checks row counts, timestamp ranges, event-type distributions,
resource request ranges, machine capacities, and other basic sanity checks using
DuckDB queries.

## Stage 2: Build Features

Command:

```bash
python -m scripts.build_features
```

Optional output paths:

```bash
python -m scripts.build_features --output artifacts/features.parquet --summary-output artifacts/feature_summary.txt
```

Reads:

- raw CSV views registered in DuckDB.

Writes:

- `artifacts/features.parquet`;
- `artifacts/feature_summary.txt`;
- `artifacts/google_trace.duckdb`;
- temporary DuckDB spill files under `artifacts/duckdb_tmp/`.

This stage derives the supervised learning table. The target is queueing delay
in seconds:

```text
first SCHEDULE timestamp - first SUBMIT timestamp
```

The feature table includes task attributes, job task counts, temporal submit
features, per-minute load proxies, backlog estimates, machine capacity features,
and deterministic train/validation/test split labels.

## Stage 3: Pre-Training Analysis

Command:

```bash
python -m scripts.pretrain_analysis --features artifacts/features.parquet
```

Writes:

- `artifacts/pretrain_metrics.json`;
- `artifacts/pretrain_queue_delay_dist.png`;
- `artifacts/pretrain_feature_*.png`;
- `artifacts/pretrain_feature_corr.png` when correlation plotting succeeds.

Useful options:

```bash
python -m scripts.pretrain_analysis --features artifacts/features.parquet --clip-target 7200 --sample-cap 500000
```

This stage samples the feature table to inspect the target distribution, split
sizes, feature histograms, and feature correlations before model training.

## Stage 4: Compute Baselines

Command:

```bash
python -m scripts.baseline_metrics
```

Optional paths:

```bash
python -m scripts.baseline_metrics --features artifacts/features.parquet --output artifacts/baseline_metrics.txt
```

Writes:

- `artifacts/baseline_metrics.txt`.

This stage evaluates constant mean and median predictors. These baselines are
used as simple references for judging model performance.

## Stage 5: Train Model

Command:

```bash
python -m scripts.train_model --features artifacts/features.parquet
```

Writes:

- `artifacts/xgb_queue_delay.json`;
- `artifacts/model_metrics.json`.

Useful options:

```bash
python -m scripts.train_model \
  --features artifacts/features.parquet \
  --num-boost-round 800 \
  --learning-rate 0.05 \
  --max-depth 10 \
  --batch-size 250000 \
  --clip-target 7200
```

By default, the model clips large targets and trains on `log1p(queue_delay)` to
reduce the effect of the heavy-tailed queueing-delay distribution. Use
`--no-log1p-target` only if you want to train directly on the clipped target in
seconds.

## Stage 6: Evaluate Model

Command:

```bash
python -m scripts.evaluate_model --features artifacts/features.parquet --model artifacts/xgb_queue_delay.json
```

Writes:

- `artifacts/evaluation_metrics.json`;
- `artifacts/evaluation_plots.png`;
- `artifacts/evaluation_plots_dist.png`;
- `artifacts/evaluation_random_plots.png`;
- `artifacts/evaluation_random_plots_dist.png`.

The evaluation reports test-set RMSE and MAE, baseline comparisons, absolute
error quantiles, sampled predicted-vs-actual diagnostics, residual plots, and
feature-importance summaries.

If the model was trained with `--no-log1p-target`, evaluate with the same flag:

```bash
python -m scripts.evaluate_model --features artifacts/features.parquet --model artifacts/xgb_queue_delay.json --no-log1p-target
```

## Common Run Patterns

Reuse an existing feature table:

```bash
python -m scripts.run_pipeline --skip-validate --skip-features --features-path artifacts/features.parquet
```

Build features and run analysis, but skip training:

```bash
python -m scripts.run_pipeline --skip-train
```

Train without post-training evaluation:

```bash
python -m scripts.run_pipeline --skip-eval
```

Run with raw-target training:

```bash
python -m scripts.run_pipeline --no-log1p-target
```

## Resource Settings

DuckDB defaults are configured in `scripts/common.py`. You can override memory
and thread settings with environment variables.

Bash:

```bash
DUCKDB_MEMORY_LIMIT=16GB DUCKDB_THREADS=8 python -m scripts.run_pipeline
```

Windows PowerShell:

```powershell
$env:DUCKDB_MEMORY_LIMIT = "16GB"
$env:DUCKDB_THREADS = "8"
python -m scripts.run_pipeline
```

## Publication Outputs

Generated artifacts are ignored by git. For a public repository, copy only small
curated summaries into `results/borg/`, such as:

- compact metrics JSON files;
- baseline summaries;
- selected diagnostic figures;
- a short note describing the command, target clipping value, model settings,
  and trace version.

Do not commit raw trace files, generated Parquet feature tables, DuckDB
databases, temporary files, or trained model artifacts.
