from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import (
    ARTIFACTS_DIR,
    MODEL_FEATURE_COLUMNS,
    ensure_database,
)


def build_feature_table(
    output_path: Path | None = None,
    summary_path: Path | None = None,
) -> Path:
    con = ensure_database()
    if output_path is None:
        output_path = ARTIFACTS_DIR / "features.parquet"
    if summary_path is None:
        summary_path = ARTIFACTS_DIR / "feature_summary.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Event-level aggregates to support load features.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE minute_events AS
        SELECT
            CAST(floor(timestamp / 60000000) AS BIGINT) AS minute_bucket,
            SUM(CASE WHEN event_type = 0 THEN 1 ELSE 0 END) AS submits,
            SUM(CASE WHEN event_type = 1 THEN 1 ELSE 0 END) AS schedules
        FROM task_events
        WHERE event_type IN (0, 1)
        GROUP BY 1
        ORDER BY 1
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE backlog AS
        SELECT
            minute_bucket,
            submits,
            schedules,
            SUM(submits - schedules) OVER (
                ORDER BY minute_bucket
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS pending_estimate
        FROM minute_events
        """
    )

    # Per-task submission and scheduling timestamps.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE submits AS
        SELECT job_id, task_index, MIN(timestamp) AS submit_ts
        FROM task_events
        WHERE event_type = 0
        GROUP BY job_id, task_index
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE schedules AS
        SELECT
            job_id,
            task_index,
            MIN(timestamp) AS schedule_ts,
            arg_min(machine_id, timestamp) AS schedule_machine
        FROM task_events
        WHERE event_type = 1
        GROUP BY job_id, task_index
        """
    )

    # Task-level attributes at (or before) submission.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE task_attributes AS
        SELECT
            job_id,
            task_index,
            arg_min(priority, timestamp) AS priority,
            arg_min(scheduling_class, timestamp) AS scheduling_class,
            arg_min(cpu_request, timestamp) AS cpu_request,
            arg_min(ram_request, timestamp) AS ram_request,
            arg_min(disk_request, timestamp) AS disk_request,
            COALESCE(MAX(CASE WHEN different_machine_constraint THEN 1 ELSE 0 END), 0) AS different_machine_constraint,
            COALESCE(MAX(CASE WHEN missing_info IS NOT NULL THEN 1 ELSE 0 END), 0) AS has_missing_info
        FROM task_events
        GROUP BY job_id, task_index
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE job_task_counts AS
        SELECT job_id, COUNT(*) AS job_task_count
        FROM submits
        GROUP BY job_id
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE machine_capacity AS
        SELECT
            machine_id,
            arg_max(capacity_cpu, timestamp) AS capacity_cpu,
            arg_max(capacity_memory, timestamp) AS capacity_memory
        FROM machine_events
        GROUP BY machine_id
        """
    )

    # Main feature table with deterministic train/valid/test split.
    con.execute(
        """
        CREATE OR REPLACE TABLE features AS
        WITH base AS (
            SELECT
                s.job_id,
                s.task_index,
                s.submit_ts,
                sch.schedule_ts,
                sch.schedule_machine,
                ta.priority,
                ta.scheduling_class,
                COALESCE(ta.cpu_request, 0.0) AS cpu_request,
                COALESCE(ta.ram_request, 0.0) AS ram_request,
                COALESCE(ta.disk_request, 0.0) AS disk_request,
                COALESCE(ta.different_machine_constraint, 0) AS different_machine_constraint,
                COALESCE(ta.has_missing_info, 0) AS has_missing_info,
                COALESCE(jt.job_task_count, 0) AS job_task_count
            FROM submits s
            JOIN schedules sch ON s.job_id = sch.job_id AND s.task_index = sch.task_index
            LEFT JOIN task_attributes ta ON s.job_id = ta.job_id AND s.task_index = ta.task_index
            LEFT JOIN job_task_counts jt ON s.job_id = jt.job_id
            WHERE sch.schedule_ts IS NOT NULL
              AND s.submit_ts IS NOT NULL
              AND sch.schedule_ts >= s.submit_ts
        ),
        enriched AS (
            SELECT
                b.*,
                (b.schedule_ts - b.submit_ts) / 1e6::DOUBLE AS queue_delay_seconds,
                CAST(floor(b.submit_ts / 1e6 / 86400) AS INTEGER) AS submit_day,
                CAST(floor(b.submit_ts / 1e6 / 3600) AS INTEGER) AS submit_hour,
                CAST(floor(b.submit_ts / 1e6 / 60) AS INTEGER) AS submit_minute,
                COALESCE(mc.capacity_cpu, 0.0) AS machine_cpu_capacity,
                COALESCE(mc.capacity_memory, 0.0) AS machine_memory_capacity,
                COALESCE(be.pending_estimate, 0) AS backlog_estimate,
                COALESCE(me.submits, 0) AS submits_same_minute,
                COALESCE(me.schedules, 0) AS schedules_same_minute,
                b.cpu_request + b.ram_request + b.disk_request AS request_sum
            FROM base b
            LEFT JOIN machine_capacity mc ON b.schedule_machine = mc.machine_id
            LEFT JOIN backlog be ON CAST(floor(b.submit_ts / 60000000) AS BIGINT) = be.minute_bucket
            LEFT JOIN minute_events me ON CAST(floor(b.submit_ts / 60000000) AS BIGINT) = me.minute_bucket
        )
        SELECT
            *,
            CASE
                WHEN abs(hash(job_id)) % 100 < 80 THEN 'train'
                WHEN abs(hash(job_id)) % 100 < 90 THEN 'valid'
                ELSE 'test'
            END AS split
        FROM enriched
        """
    )

    con.execute(
        f"""
        COPY (SELECT * FROM features)
        TO '{output_path}'
        WITH (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 2000000)
        """
    )

    # Keep a small text summary for quick inspection.
    summary = con.execute(
        """
        SELECT split, COUNT(*) AS rows, AVG(queue_delay_seconds) AS avg_queue_delay
        FROM features
        GROUP BY split
        ORDER BY split
        """
    ).fetchall()
    summary_lines = [
        "split\trows\tavg_queue_delay_seconds",
        *[f"{row[0]}\t{row[1]}\t{row[2]:.4f}" for row in summary],
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the per-task queue-delay feature table with DuckDB."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "features.parquet",
        help="Path for the generated features Parquet file.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ARTIFACTS_DIR / "feature_summary.txt",
        help="Path for the feature summary text file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_feature_table(
        output_path=args.output,
        summary_path=args.summary_output,
    )
    print(f"Feature table written to {output_path}")
    print(f"Feature summary written to {args.summary_output}")
    print(f"Model features: {MODEL_FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
