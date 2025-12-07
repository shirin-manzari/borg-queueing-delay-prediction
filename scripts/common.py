from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import duckdb

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "extracted"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DB_PATH = ARTIFACTS_DIR / "google_trace.duckdb"


# Schema definitions taken from
# "Google cluster-usage traces format schema 2014-11-17 external.pdf".
TASK_EVENTS_COLUMNS: Dict[str, str] = {
    "timestamp": "UBIGINT",
    "missing_info": "UBIGINT",
    "job_id": "UBIGINT",
    "task_index": "UBIGINT",
    "machine_id": "UBIGINT",
    "event_type": "INTEGER",
    "user_name": "VARCHAR",
    "scheduling_class": "INTEGER",
    "priority": "INTEGER",
    "cpu_request": "DOUBLE",
    "ram_request": "DOUBLE",
    "disk_request": "DOUBLE",
    "different_machine_constraint": "BOOLEAN",
}

JOB_EVENTS_COLUMNS: Dict[str, str] = {
    "timestamp": "UBIGINT",
    "missing_info": "UBIGINT",
    "job_id": "UBIGINT",
    "event_type": "INTEGER",
    "user_name": "VARCHAR",
    "scheduling_class": "INTEGER",
    "job_name": "VARCHAR",
    "logical_job_name": "VARCHAR",
}

MACHINE_EVENTS_COLUMNS: Dict[str, str] = {
    "timestamp": "UBIGINT",
    "machine_id": "UBIGINT",
    "event_type": "INTEGER",
    "platform_id": "VARCHAR",
    "capacity_cpu": "DOUBLE",
    "capacity_memory": "DOUBLE",
}

MACHINE_ATTRIBUTES_COLUMNS: Dict[str, str] = {
    "timestamp": "UBIGINT",
    "machine_id": "UBIGINT",
    "attribute_name": "VARCHAR",
    "attribute_value": "VARCHAR",
    "attribute_deleted": "BOOLEAN",
}

TASK_CONSTRAINTS_COLUMNS: Dict[str, str] = {
    "timestamp": "UBIGINT",
    "job_id": "UBIGINT",
    "task_index": "UBIGINT",
    "attribute_name": "VARCHAR",
    "attribute_value": "VARCHAR",
    "comparison_operator": "INTEGER",
}

TASK_USAGE_COLUMNS: Dict[str, str] = {
    "start_time": "UBIGINT",
    "end_time": "UBIGINT",
    "job_id": "UBIGINT",
    "task_index": "UBIGINT",
    "machine_id": "UBIGINT",
    "mean_cpu_usage_rate": "DOUBLE",
    "canonical_memory_usage": "DOUBLE",
    "assigned_memory_usage": "DOUBLE",
    "unmapped_page_cache_memory_usage": "DOUBLE",
    "total_page_cache_memory_usage": "DOUBLE",
    "maximum_memory_usage": "DOUBLE",
    "mean_disk_io_time": "DOUBLE",
    "mean_local_disk_space_used": "DOUBLE",
    "maximum_cpu_usage": "DOUBLE",
    "maximum_disk_io_time": "DOUBLE",
    "cycles_per_instruction": "DOUBLE",
    "memory_accesses_per_instruction": "DOUBLE",
    "sample_portion": "DOUBLE",
    "aggregation_type": "INTEGER",
    "sampled_cpu_usage": "DOUBLE",
}


def _duckdb_config() -> dict:
    """
    Conservative defaults with spill-to-disk to avoid RAM blow-ups.
    Override via env:
      DUCKDB_MEMORY_LIMIT (e.g., '8GB')
      DUCKDB_THREADS (e.g., '4')
    """
    temp_dir = ARTIFACTS_DIR / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "temp_directory": str(temp_dir),
        "memory_limit": os.getenv("DUCKDB_MEMORY_LIMIT", "8GB"),
        "threads": int(os.getenv("DUCKDB_THREADS", "8")),
    }
    return config


def connect_duckdb(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Connect to (or create) the on-disk DuckDB database stored in artifacts/.
    Ensures the artifacts directory exists to avoid runtime surprises.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH), read_only=read_only, config=_duckdb_config())


def _read_csv_sql(path: Path, columns: Dict[str, str]) -> str:
    quoted_columns = ", ".join(f"'{k}': '{v}'" for k, v in columns.items())
    columns_map = "{" + quoted_columns + "}"
    return (
        f"SELECT * FROM read_csv('{path}', "
        f"columns={columns_map}, "
        "nullstr=[''], delim=',', header=False, auto_detect=False)"
    )


def register_views(con: duckdb.DuckDBPyConnection) -> None:
    """
    Register the CSV files as DuckDB views with explicit schema to avoid
    type inference on the massive inputs.
    """
    con.execute(
        f"CREATE OR REPLACE VIEW task_events AS {_read_csv_sql(DATA_DIR / 'task_events.csv', TASK_EVENTS_COLUMNS)}"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW job_events AS {_read_csv_sql(DATA_DIR / 'job_events.csv', JOB_EVENTS_COLUMNS)}"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW machine_events AS {_read_csv_sql(DATA_DIR / 'machine_events.csv', MACHINE_EVENTS_COLUMNS)}"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW machine_attributes AS {_read_csv_sql(DATA_DIR / 'machine_attributes.csv', MACHINE_ATTRIBUTES_COLUMNS)}"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW task_constraints AS {_read_csv_sql(DATA_DIR / 'task_constraints.csv', TASK_CONSTRAINTS_COLUMNS)}"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW task_usage AS {_read_csv_sql(DATA_DIR / 'task_usage.csv', TASK_USAGE_COLUMNS)}"
    )


def ensure_database() -> duckdb.DuckDBPyConnection:
    """
    Convenience wrapper to create a DuckDB connection and register all views.
    """
    con = connect_duckdb()
    register_views(con)
    return con


MODEL_FEATURE_COLUMNS = [
    "priority",
    "scheduling_class",
    "cpu_request",
    "ram_request",
    "disk_request",
    "request_sum",
    "different_machine_constraint",
    "has_missing_info",
    "submit_day",
    "submit_hour",
    "submit_minute",
    "backlog_estimate",
    "submits_same_minute",
    "schedules_same_minute",
    "job_task_count",
    "machine_cpu_capacity",
    "machine_memory_capacity",
]
