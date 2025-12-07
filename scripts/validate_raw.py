from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd

from scripts.common import ARTIFACTS_DIR, ensure_database


def _df_section(title: str, df: pd.DataFrame) -> str:
    header = f"{title}\n{'-' * len(title)}"
    return header + "\n" + df.to_string(index=False)


def run_validation() -> Path:
    """
    Run basic sanity checks against the raw CSVs using DuckDB without loading
    everything into memory. Writes a human-readable report to artifacts/.
    """
    con = ensure_database()

    report: list[str] = []

    # Row counts for each table.
    row_counts = con.execute(
        """
        SELECT 'task_events' AS table, COUNT(*) AS rows FROM task_events
        UNION ALL
        SELECT 'job_events', COUNT(*) FROM job_events
        UNION ALL
        SELECT 'task_constraints', COUNT(*) FROM task_constraints
        UNION ALL
        SELECT 'machine_events', COUNT(*) FROM machine_events
        UNION ALL
        SELECT 'machine_attributes', COUNT(*) FROM machine_attributes
        UNION ALL
        SELECT 'task_usage', COUNT(*) FROM task_usage
        """
    ).fetchdf()
    report.append(_df_section("Row counts", row_counts))

    # Task-level uniqueness and ranges.
    task_core_stats = con.execute(
        """
        WITH distinct_tasks AS (
            SELECT job_id, task_index FROM task_events GROUP BY job_id, task_index
        )
        SELECT
            COUNT(*) AS total_rows,
            (SELECT COUNT(*) FROM distinct_tasks) AS unique_tasks,
            (SELECT COUNT(DISTINCT job_id) FROM distinct_tasks) AS unique_jobs,
            MIN(timestamp) AS min_timestamp,
            MAX(timestamp) AS max_timestamp,
            MIN(event_type) AS min_event_type,
            MAX(event_type) AS max_event_type,
            SUM(CASE WHEN event_type NOT BETWEEN 0 AND 8 THEN 1 ELSE 0 END) AS invalid_event_type_rows,
            MIN(priority) AS min_priority,
            MAX(priority) AS max_priority,
            MIN(cpu_request) AS min_cpu_request,
            MAX(cpu_request) AS max_cpu_request,
            MIN(ram_request) AS min_ram_request,
            MAX(ram_request) AS max_ram_request,
            MIN(disk_request) AS min_disk_request,
            MAX(disk_request) AS max_disk_request,
            SUM(CASE WHEN missing_info IS NOT NULL THEN 1 ELSE 0 END) AS missing_info_rows
        FROM task_events
        """
    ).fetchdf()
    report.append(_df_section("Task event stats", task_core_stats))

    event_counts = con.execute(
        """
        SELECT event_type, COUNT(*) AS rows
        FROM task_events
        GROUP BY event_type
        ORDER BY event_type
        """
    ).fetchdf()
    report.append(_df_section("Task event type distribution", event_counts))

    priority_hist = con.execute(
        """
        SELECT priority, COUNT(*) AS rows
        FROM task_events
        GROUP BY priority
        ORDER BY priority
        LIMIT 50
        """
    ).fetchdf()
    report.append(_df_section("Priority histogram (first 50 values)", priority_hist))

    machine_event_counts = con.execute(
        """
        SELECT event_type, COUNT(*) AS rows
        FROM machine_events
        GROUP BY event_type
        ORDER BY event_type
        """
    ).fetchdf()
    report.append(_df_section("Machine event distribution", machine_event_counts))

    machine_capacity_ranges = con.execute(
        """
        SELECT
            MIN(capacity_cpu) AS min_cpu_capacity,
            MAX(capacity_cpu) AS max_cpu_capacity,
            MIN(capacity_memory) AS min_memory_capacity,
            MAX(capacity_memory) AS max_memory_capacity
        FROM machine_events
        """
    ).fetchdf()
    report.append(_df_section("Machine capacity ranges", machine_capacity_ranges))

    # Basic usage sanity checks (these tables are huge, keep it simple).
    usage_ranges = con.execute(
        """
        SELECT
            MIN(start_time) AS min_start_time,
            MAX(end_time) AS max_end_time,
            MIN(mean_cpu_usage_rate) AS min_mean_cpu,
            MAX(mean_cpu_usage_rate) AS max_mean_cpu,
            MIN(canonical_memory_usage) AS min_canonical_mem,
            MAX(canonical_memory_usage) AS max_canonical_mem,
            SUM(CASE WHEN aggregation_type NOT IN (0,1) THEN 1 ELSE 0 END) AS invalid_agg_type_rows
        FROM task_usage
        """
    ).fetchdf()
    report.append(_df_section("Task usage ranges", usage_ranges))

    output_path = ARTIFACTS_DIR / "validation_report_duckdb.txt"
    output_path.write_text(
        "\n\n".join(
            textwrap.dedent(section).strip() for section in report
        )
        + "\n"
    )
    return output_path


def main() -> None:
    output_path = run_validation()
    print(f"Validation report written to {output_path}")


if __name__ == "__main__":
    main()

