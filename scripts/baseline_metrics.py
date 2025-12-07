from __future__ import annotations

from pathlib import Path

from scripts.common import ARTIFACTS_DIR, connect_duckdb


def compute_baselines(features_path: Path | None = None) -> Path:
    """
    Compute simple constant baselines (train mean/median) and distribution
    summaries so you can compare model results before training.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if features_path is None:
        features_path = ARTIFACTS_DIR / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features parquet not found at {features_path}. Build it first."
        )

    con = connect_duckdb()
    con.execute(
        f"CREATE OR REPLACE VIEW features_parquet AS SELECT * FROM read_parquet('{features_path}')"
    )

    train_stats = con.execute(
        """
        SELECT
            COUNT(*) AS train_rows,
            AVG(queue_delay_seconds) AS train_mean,
            MEDIAN(queue_delay_seconds) AS train_median,
            QUANTILE_CONT(queue_delay_seconds, 0.90) AS p90,
            QUANTILE_CONT(queue_delay_seconds, 0.95) AS p95,
            QUANTILE_CONT(queue_delay_seconds, 0.99) AS p99,
            QUANTILE_CONT(queue_delay_seconds, 0.999) AS p999,
            MAX(queue_delay_seconds) AS max_delay
        FROM features_parquet
        WHERE split = 'train'
        """
    ).fetchone()

    baseline_metrics = con.execute(
        """
        WITH train_stats AS (
            SELECT
                AVG(queue_delay_seconds) AS train_mean,
                MEDIAN(queue_delay_seconds) AS train_median
            FROM features_parquet
            WHERE split = 'train'
        )
        SELECT
            f.split,
            COUNT(*) AS rows,
            AVG(queue_delay_seconds) AS mean_delay,
            MEDIAN(queue_delay_seconds) AS median_delay,
            AVG(ABS(queue_delay_seconds - ts.train_mean)) AS mae_mean_baseline,
            SQRT(AVG(POWER(queue_delay_seconds - ts.train_mean, 2))) AS rmse_mean_baseline,
            AVG(ABS(queue_delay_seconds - ts.train_median)) AS mae_median_baseline,
            SQRT(AVG(POWER(queue_delay_seconds - ts.train_median, 2))) AS rmse_median_baseline
        FROM features_parquet f
        CROSS JOIN train_stats ts
        GROUP BY f.split
        ORDER BY f.split
        """
    ).fetchall()

    output_path = ARTIFACTS_DIR / "baseline_metrics.txt"
    lines: list[str] = []
    lines.append("Train distribution summary (seconds):")
    lines.append(
        "rows\tmean\tmedian\tp90\tp95\tp99\tp99.9\tmax"
    )
    lines.append(
        f"{train_stats[0]}\t{train_stats[1]:.3f}\t{train_stats[2]:.3f}\t"
        f"{train_stats[3]:.3f}\t{train_stats[4]:.3f}\t{train_stats[5]:.3f}\t"
        f"{train_stats[6]:.3f}\t{train_stats[7]:.3f}"
    )
    lines.append("")
    lines.append("Constant baseline metrics (using train mean/median predictors):")
    lines.append(
        "split\trows\tmean_delay\tmedian_delay\tmae_mean_baseline\trmse_mean_baseline\tmae_median_baseline\trmse_median_baseline"
    )
    for row in baseline_metrics:
        lines.append(
            f"{row[0]}\t{row[1]}\t{row[2]:.3f}\t{row[3]:.3f}\t"
            f"{row[4]:.3f}\t{row[5]:.3f}\t{row[6]:.3f}\t{row[7]:.3f}"
        )
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def main() -> None:
    path = compute_baselines()
    print(f"Wrote baseline metrics to {path}")


if __name__ == "__main__":
    main()

