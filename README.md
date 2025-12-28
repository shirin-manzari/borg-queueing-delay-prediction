# Borg Queueing Delay Prediction

This repository studies task queueing delay prediction on the public Google 2011
Borg cluster trace. It builds an offline machine-learning pipeline that validates
raw trace CSVs, constructs queue-delay features with DuckDB, trains an XGBoost
regressor, and evaluates prediction quality against simple baselines.

## Scope

This study predicts per-task queueing delay, defined as:

```text
first SCHEDULE timestamp - first SUBMIT timestamp
```

The project evaluates whether queueing delay can be predicted from public
trace-derived features, including:

- task priority, scheduling class, and resource requests;
- job-level task counts;
- temporal submit features;
- simple cluster-load proxies such as submits, schedules, and backlog per minute;
- machine capacity information available in the trace.

This is not a reproduction of Google's production Borg scheduler. It is a
focused offline prediction workflow using only information available in the
public trace.

## Repository Contents

- `scripts/run_pipeline.py`: end-to-end pipeline entrypoint.
- `scripts/validate_raw.py`: raw CSV validation and sanity checks.
- `scripts/build_features.py`: DuckDB feature-table construction.
- `scripts/baseline_metrics.py`: constant mean and median baselines.
- `scripts/pretrain_analysis.py`: pre-training feature and target analysis.
- `scripts/train_model.py`: XGBoost training on streamed Parquet batches.
- `scripts/evaluate_model.py`: post-training evaluation and diagnostic plots.
- `scripts/common.py`: shared paths, schemas, DuckDB setup, and model features.
- `PIPELINE.md`: stage-by-stage commands and generated outputs.
- `PROJECT_DEEP_DIVE.md`: detailed explanation of the project workflow.
- `docs/data_schema.md`: expected local data layout.
- `results/borg/`: curated lightweight results for publication.

Raw traces, generated features, DuckDB databases, trained models, and full local
artifact directories are not tracked in git.

## Requirements

- Python 3.10 or newer recommended.
- Local copy of the Google 2011 cluster trace CSV files.
- Python packages listed in `requirements.txt`.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Run

Place the required raw CSV files under:

```text
extracted/
```

Expected files include:

```text
task_events.csv
job_events.csv
machine_events.csv
machine_attributes.csv
task_constraints.csv
task_usage.csv
```

Then run:

```bash
python -m scripts.run_pipeline
```

Optional reuse of an existing feature table:

```bash
python -m scripts.run_pipeline --skip-validate --skip-features --features-path artifacts/features.parquet
```

The pipeline writes local outputs under `artifacts/`.

## Results

Generated metrics and plots include:

- validation report for raw trace tables;
- feature summary;
- baseline mean and median predictor metrics;
- pre-training target and feature analysis;
- XGBoost training metrics;
- test-set evaluation metrics;
- predicted-vs-actual and residual diagnostic plots;
- feature-importance summaries.

Curated publishable summaries should be placed under `results/borg/`. Full
artifacts remain ignored because they can be large and machine-specific.

## Modeling Notes

The target distribution is heavy-tailed, so the default training configuration
clips large queueing delays and trains on a `log1p` transformed target. During
evaluation, predictions are converted back to seconds and compared against
baselines in the original unit.

Train, validation, and test splits are assigned deterministically from `job_id`
so that all tasks from the same job stay in the same split.

## Limitations

This repository uses only public trace data. It does not model hidden scheduler
state, production Borg policies, user intent, or cluster conditions that are not
included in the released trace.

The results should be interpreted as an offline predictability study, not as a
complete scheduler simulation or a claim about production deployment behavior.

## References

- Google cluster-usage traces, version 2.
- Reiss et al., Heterogeneity and Dynamicity of Clouds at Scale: Google Trace
  Analysis, SoCC 2012.
