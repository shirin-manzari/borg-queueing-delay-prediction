# Borg Queueing Delay Prediction Deep Dive

This document explains the repository from the outside in: what problem it
studies, how the Google Borg trace moves through the pipeline, how features are
created, how the model is trained, and how outputs should be interpreted.

## Executive Summary

This repository is an offline machine-learning workflow for predicting task
queueing delay in the public Google 2011 Borg cluster trace.

For each task, the supervised target is:

```text
first SCHEDULE timestamp - first SUBMIT timestamp
```

The raw trace CSVs are large, so the workflow is built around two constraints:

- keep raw-data aggregation inside DuckDB;
- stream model-training batches from Parquet instead of loading the full feature
  table into memory.

The active workflow validates the raw CSVs, builds a per-task feature table,
performs pre-training analysis, trains an XGBoost regressor, and evaluates model
quality against simple constant baselines.

## What This Repository Is

This is a focused predictability study. It asks whether queueing delay can be
estimated from task, job, machine, and cluster-load features derived from the
public trace.

It is not:

- a production Borg scheduler;
- a reconstruction of Google's private scheduling logic;
- a simulator of all cluster-side constraints;
- a repository for redistributing the raw Google trace data.

The public repository is intended to contain source code, documentation, and
small curated result summaries. Raw data and generated artifacts remain local.

## Repository Structure

```text
.
|-- README.md
|-- PIPELINE.md
|-- PROJECT_DEEP_DIVE.md
|-- requirements.txt
|-- docs/
|   `-- data_schema.md
|-- results/
|   `-- borg/
|-- scripts/
|   |-- common.py
|   |-- validate_raw.py
|   |-- build_features.py
|   |-- baseline_metrics.py
|   |-- pretrain_analysis.py
|   |-- train_model.py
|   |-- evaluate_model.py
|   `-- run_pipeline.py
```

Local runtime directories are intentionally ignored:

```text
extracted/     raw trace CSVs
artifacts/     generated features, metrics, models, plots, DuckDB files
work/          optional scratch space
```

## External Inputs

The pipeline expects the Google cluster trace CSV files under `extracted/`:

```text
extracted/task_events.csv
extracted/job_events.csv
extracted/machine_events.csv
extracted/machine_attributes.csv
extracted/task_constraints.csv
extracted/task_usage.csv
```

The raw CSV schemas are defined in `scripts/common.py`. The code passes explicit
column names and DuckDB types into `read_csv` so DuckDB does not need to infer
types across very large files.

## Shared Infrastructure

`scripts/common.py` owns shared project configuration:

- repository-relative paths;
- `extracted/` raw-data directory;
- `artifacts/` output directory;
- `artifacts/google_trace.duckdb` database path;
- explicit schema dictionaries for each raw CSV;
- DuckDB connection setup;
- model feature column order.

DuckDB uses a local temporary directory under `artifacts/duckdb_tmp/`. Memory and
thread settings can be overridden with:

```text
DUCKDB_MEMORY_LIMIT
DUCKDB_THREADS
```

## Active Workflow

The full pipeline entrypoint is:

```bash
python -m scripts.run_pipeline
```

`scripts/run_pipeline.py` runs:

1. `validate_raw.run_validation`;
2. `build_features.build_feature_table`;
3. `pretrain_analysis.analyze`;
4. `train_model.train`;
5. `evaluate_model.evaluate`.

Each stage can also be run independently. See `PIPELINE.md` for exact commands.

## Stage 1: Raw Data Validation

`scripts/validate_raw.py` performs sanity checks without loading the full trace
into pandas. DuckDB executes the heavy queries and returns compact summaries.

The validation report includes:

- row counts for each raw table;
- task and job counts;
- timestamp ranges;
- task event-type distribution;
- priority histogram;
- machine event distribution;
- machine capacity ranges;
- task usage ranges;
- invalid event or aggregation-type counts.

Output:

```text
artifacts/validation_report_duckdb.txt
```

This stage is mostly defensive. It helps catch missing files, unexpected schema
issues, and obviously invalid ranges before feature construction starts.

## Stage 2: Feature Construction

`scripts/build_features.py` constructs a per-task supervised learning table.
The output is:

```text
artifacts/features.parquet
artifacts/feature_summary.txt
```

The feature build creates several DuckDB temporary tables.

`minute_events`

Counts submit and schedule events per minute:

- `submits`;
- `schedules`;
- `minute_bucket`.

`backlog`

Computes a simple cumulative pending estimate:

```text
cumulative sum of submits - schedules
```

This is a trace-derived load proxy, not a full scheduler queue reconstruction.

`submits`

Finds the first submit timestamp for each `(job_id, task_index)` pair.

`schedules`

Finds the first schedule timestamp and corresponding scheduled machine for each
task.

`task_attributes`

Collects task-level attributes such as:

- priority;
- scheduling class;
- CPU, RAM, and disk requests;
- different-machine constraint flag;
- missing-info flag.

`job_task_counts`

Counts how many submitted tasks appear in each job.

`machine_capacity`

Uses the latest known machine event to derive CPU and memory capacity per
machine.

## Target Definition

The supervised target is:

```text
queue_delay_seconds = (schedule_ts - submit_ts) / 1e6
```

The code keeps only tasks where:

- a submit event exists;
- a schedule event exists;
- `schedule_ts >= submit_ts`.

Timestamps are stored in microseconds in the raw trace and converted to seconds
for modeling and evaluation.

## Feature Columns

The model uses the ordered feature list in `scripts/common.py`:

```text
priority
scheduling_class
cpu_request
ram_request
disk_request
request_sum
different_machine_constraint
has_missing_info
submit_day
submit_hour
submit_minute
backlog_estimate
submits_same_minute
schedules_same_minute
job_task_count
machine_cpu_capacity
machine_memory_capacity
```

Feature values are derived before training and stored in `features.parquet`.
Missing numeric values used by the current feature logic are generally filled
with `0.0` or `0` during feature construction.

## Split Strategy

The feature table includes a deterministic split column:

```text
abs(hash(job_id)) % 100 < 80    train
abs(hash(job_id)) % 100 < 90    valid
otherwise                      test
```

This gives an approximate 80/10/10 split while keeping all tasks from the same
job in the same split. Job-level splitting reduces leakage compared with random
per-task splitting.

## Baselines

`scripts/baseline_metrics.py` computes simple constant predictors from the
training split:

- train mean queueing delay;
- train median queueing delay.

It evaluates those predictors on each split using MAE and RMSE. These baselines
provide a low-complexity reference point for the XGBoost model.

Output:

```text
artifacts/baseline_metrics.txt
```

## Pre-Training Analysis

`scripts/pretrain_analysis.py` samples rows from `features.parquet` and produces
lightweight diagnostics before training.

Outputs include:

```text
artifacts/pretrain_metrics.json
artifacts/pretrain_queue_delay_dist.png
artifacts/pretrain_feature_*.png
artifacts/pretrain_feature_corr.png
```

The analysis summarizes split sizes, target quantiles, sampled feature
distributions, and correlations when plotting dependencies are available.

## Model Training

`scripts/train_model.py` trains an XGBoost regressor.

The training design is built around large feature tables:

- `pyarrow.dataset` reads Parquet batches;
- `ParquetIterator` streams batches into XGBoost;
- `xgb.QuantileDMatrix` avoids loading all training rows at once;
- histogram-based trees are used with `tree_method = hist`;
- early stopping uses the validation split.

Default training settings include:

- squared-error regression objective;
- RMSE and MAE evaluation metrics;
- learning rate `0.05`;
- max depth `10`;
- subsample `0.9`;
- column sample by tree `0.9`;
- 800 boosting rounds;
- 50 early-stopping rounds;
- random seed `42`.

Outputs:

```text
artifacts/xgb_queue_delay.json
artifacts/model_metrics.json
```

## Target Transformation

Queueing delay is heavy-tailed. By default, training applies:

1. upper clipping at `7200` seconds;
2. `log1p` transformation.

During evaluation, predictions are transformed back with `expm1`, clipped to the
same maximum, and clamped to non-negative values where applicable.

The `--no-log1p-target` flag disables log-transform training. When used for
training, the same flag should also be used for evaluation.

## Evaluation

`scripts/evaluate_model.py` evaluates the trained model on the test split and on
a random sampled subset.

The evaluation reports:

- RMSE in seconds;
- MAE in seconds;
- absolute-error quantiles;
- mean and median baseline comparisons;
- sampled predicted-vs-actual summaries;
- feature importance by XGBoost gain, weight, and cover.

Outputs include:

```text
artifacts/evaluation_metrics.json
artifacts/evaluation_plots.png
artifacts/evaluation_plots_dist.png
artifacts/evaluation_random_plots.png
artifacts/evaluation_random_plots_dist.png
```

## Generated Artifact Policy

The full local run can produce large or machine-specific files:

- raw extracted CSVs;
- DuckDB databases and temporary spill files;
- Parquet feature tables;
- trained model files;
- full plot sets;
- logs and scratch files.

These are ignored by git. Public commits should include only source code,
documentation, and small curated summaries under `results/borg/`.

## Interpreting Results

The most useful comparisons are:

- XGBoost test MAE/RMSE vs mean and median baselines;
- error quantiles, especially p90, p95, and p99;
- predicted-vs-actual plots on log-like axes;
- residual patterns across short and long delays;
- feature importance stability across runs.

Because the target is heavy-tailed, average metrics alone can be misleading.
Median and upper-quantile errors are important for understanding practical model
behavior.

## Limitations

The model only uses public trace-derived features. It cannot observe:

- hidden scheduler state;
- private Borg policy details;
- real-time production signals omitted from the trace;
- user intent or unrecorded job constraints;
- cluster state after trace preprocessing choices.

The backlog estimate is a coarse aggregate signal based on submit and schedule
event counts. It should not be interpreted as the exact internal Borg queue.

Machine capacity is attached through the scheduled machine, so those features
should be interpreted carefully. They describe the eventual scheduled machine,
not necessarily every candidate machine considered before scheduling.

Overall, this project should be read as an offline queue-delay predictability
study, not a complete scheduling simulator.

## Reproducibility Checklist

Before publishing a result, record:

- trace version and local CSV preparation steps;
- command used to run the pipeline;
- DuckDB memory and thread settings;
- target clipping value;
- whether `log1p` target training was enabled;
- training hyperparameters;
- row counts by split;
- baseline metrics;
- model evaluation metrics.

Store small summaries in `results/borg/` and keep large generated artifacts
local.
