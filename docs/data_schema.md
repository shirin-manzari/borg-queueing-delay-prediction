# Data Access and Local Layout

This project uses the public Google 2011 Borg cluster trace, also known as
`clusterdata-2011-2` or Google cluster trace version 2.

The raw trace files are not included in this repository. Download them from the
official Google trace source, prepare them locally, and place the CSV files under
`extracted/`.

## Official Data Source

Google maintains trace documentation in the `google/cluster-data` repository:

- Repository: <https://github.com/google/cluster-data>
- 2011 trace notes: <https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md>

The official notes describe the 2011 trace as a Borg cell trace from May 2011,
covering about 29 days from a cluster of roughly 12.5k machines. The compressed
trace is large, so plan for substantial download time and local disk usage.

Follow Google's current download instructions from the official trace
documentation. This project assumes the final local input is plain CSV files with
the names listed below.

## Required Local Files

Create this directory at the repository root:

```text
extracted/
```

The pipeline expects these files:

```text
extracted/task_events.csv
extracted/job_events.csv
extracted/machine_events.csv
extracted/machine_attributes.csv
extracted/task_constraints.csv
extracted/task_usage.csv
```

The code currently expects one CSV file per logical table. If your downloaded
trace is split into many part files, concatenate or otherwise prepare each table
into the expected filename before running the pipeline.

## Expected Repository Layout

After preparing the raw data, the local tree should look like:

```text
.
|-- extracted/
|   |-- task_events.csv
|   |-- job_events.csv
|   |-- machine_events.csv
|   |-- machine_attributes.csv
|   |-- task_constraints.csv
|   `-- task_usage.csv
|-- scripts/
|-- artifacts/          generated locally after running the pipeline
|-- README.md
`-- PIPELINE.md
```

`extracted/` is ignored by git.

## Schema Handling

Column schemas are defined in `scripts/common.py`:

- `TASK_EVENTS_COLUMNS`
- `JOB_EVENTS_COLUMNS`
- `MACHINE_EVENTS_COLUMNS`
- `MACHINE_ATTRIBUTES_COLUMNS`
- `TASK_CONSTRAINTS_COLUMNS`
- `TASK_USAGE_COLUMNS`

DuckDB reads the CSVs with explicit column names and types:

```text
header=False
auto_detect=False
delim=','
nullstr=['']
```

This avoids expensive type inference on very large files and keeps parsing
consistent with the public trace schema.

## Tables Used By This Project

`task_events.csv`

Main source for task lifecycle events. The pipeline uses submit and schedule
events to derive the queueing-delay target.

`job_events.csv`

Used in validation and available for future job-level analysis.

`machine_events.csv`

Used to derive machine capacity features.

`machine_attributes.csv`

Used in validation and available for future machine-attribute features.

`task_constraints.csv`

Used in validation and available for future placement-constraint features.

`task_usage.csv`

Used in validation. The current model does not use post-schedule usage features
for prediction because the target is queueing delay before scheduling.

## Generated Local Outputs

Running the pipeline creates:

```text
artifacts/
```

Typical generated files include:

- `artifacts/google_trace.duckdb`
- `artifacts/duckdb_tmp/`
- `artifacts/features.parquet`
- `artifacts/feature_summary.txt`
- `artifacts/validation_report_duckdb.txt`
- `artifacts/pretrain_metrics.json`
- `artifacts/baseline_metrics.txt`
- `artifacts/xgb_queue_delay.json`
- `artifacts/model_metrics.json`
- `artifacts/evaluation_metrics.json`
- diagnostic PNG plots

`artifacts/` is ignored by git.

## Storage Policy

Do not commit:

- raw Google trace files;
- extracted CSVs;
- downloaded archives;
- DuckDB databases;
- temporary DuckDB spill files;
- Parquet feature tables;
- trained model artifacts;
- full local plot/log directories.

Only commit small curated summaries under:

```text
results/borg/
```

## Quick Verification

After preparing the data, run:

```bash
python -m scripts.validate_raw
```

If the local files are present and parseable, this writes:

```text
artifacts/validation_report_duckdb.txt
```

If validation fails, check:

- each required CSV filename is present under `extracted/`;
- files are plain CSVs, not compressed archives;
- the trace version matches the Google 2011 version 2 schema;
- there is enough disk space for DuckDB temporary files;
- `DUCKDB_MEMORY_LIMIT` is reasonable for the machine.

## Citation and License Notes

The Google trace documentation states that the trace data and documentation are
made available under a Creative Commons Attribution license. Follow the official
Google trace documentation for current license and citation requirements when
publishing results.
