from __future__ import annotations

"""
Model evaluation utilities for the queueing-delay predictor.
Computes metrics on the test split and a random sample, and emits plots.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds
import xgboost as xgb

from scripts.common import ARTIFACTS_DIR, MODEL_FEATURE_COLUMNS

LABEL_COLUMN = "queue_delay_seconds"
TOP_K_IMPORTANCE = 25


def _load_booster(model_path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def _stream_metrics(
    booster: xgb.Booster,
    dataset: ds.Dataset,
    filter_expr,
    batch_size: int,
    clip_target: float | None,
    sample_cap: int,
    log1p_target: bool,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Stream over the dataset subset defined by filter_expr, computing metrics.
    Also collect a sampled subset of predictions/labels for plotting.
    """
    batches = dataset.to_batches(
        columns=MODEL_FEATURE_COLUMNS + [LABEL_COLUMN, "split"],
        filter=filter_expr,
        batch_size=batch_size,
    )

    sse = 0.0
    sae = 0.0
    n = 0

    sample_y = []
    sample_pred = []

    iteration_limit = (
        booster.best_iteration + 1
        if booster.best_iteration is not None
        else booster.num_boosted_rounds()
    )

    for batch in batches:
        df = batch.to_pandas()
        y = df.pop(LABEL_COLUMN).to_numpy()
        if clip_target is not None:
            y = np.minimum(y, clip_target)
        df.drop(columns=["split"], inplace=True, errors="ignore")
        df = df[MODEL_FEATURE_COLUMNS]
        preds = booster.predict(xgb.DMatrix(df), iteration_range=(0, iteration_limit))
        if log1p_target:
            if clip_target is not None:
                preds = np.minimum(preds, np.log1p(clip_target))
            preds = np.expm1(preds)
        if clip_target is not None:
            preds = np.minimum(preds, clip_target)
        preds = np.maximum(preds, 0)

        residuals = preds - y
        sse += float(np.dot(residuals, residuals))
        sae += float(np.abs(residuals).sum())
        n += y.shape[0]

        # Reservoir sampling to cap sample size for plotting.
        for yy, pp in zip(y, preds):
            if len(sample_y) < sample_cap:
                sample_y.append(yy)
                sample_pred.append(pp)
            else:
                j = random.randint(0, n - 1)
                if j < sample_cap:
                    sample_y[j] = yy
                    sample_pred[j] = pp

    rmse = float(np.sqrt(sse / max(n, 1)))
    mae = float(sae / max(n, 1))
    sample_y_arr = np.array(sample_y)
    sample_pred_arr = np.array(sample_pred)
    abs_errors = np.abs(sample_pred_arr - sample_y_arr)
    quantiles = np.percentile(abs_errors, [50, 90, 95, 99]) if abs_errors.size else [0, 0, 0, 0]

    metrics = {
        "rmse_seconds": rmse,
        "mae_seconds": mae,
        "rows": n,
        "abs_error_quantiles": {
            "p50": float(quantiles[0]),
            "p90": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "p99": float(quantiles[3]),
        },
    }
    samples = {"y": sample_y_arr, "pred": sample_pred_arr, "abs_error": abs_errors}
    return metrics, samples


def _baseline_stats(dataset: ds.Dataset, clip_target: float | None) -> Dict[str, float]:
    """
    Compute train mean/median for constant baselines.
    """
    train_df = dataset.to_table(columns=[LABEL_COLUMN], filter=ds.field("split") == "train")
    arr = train_df[LABEL_COLUMN].to_numpy()
    if clip_target is not None:
        arr = np.minimum(arr, clip_target)
    return {
        "train_mean": float(np.mean(arr)),
        "train_median": float(np.median(arr)),
    }


def _baseline_metrics(
    dataset: ds.Dataset,
    filter_expr,
    batch_size: int,
    clip_target: float | None,
    train_mean: float,
    train_median: float,
) -> Dict[str, float]:
    """
    Evaluate constant baselines (mean/median) on a subset.
    """
    batches = dataset.to_batches(
        columns=[LABEL_COLUMN], filter=filter_expr, batch_size=batch_size
    )
    n = 0
    sae_mean = 0.0
    sae_med = 0.0
    sse_mean = 0.0
    sse_med = 0.0
    for batch in batches:
        y = batch.column(LABEL_COLUMN).to_numpy()
        if clip_target is not None:
            y = np.minimum(y, clip_target)
        residual_mean = y - train_mean
        residual_med = y - train_median
        sae_mean += float(np.abs(residual_mean).sum())
        sae_med += float(np.abs(residual_med).sum())
        sse_mean += float(np.dot(residual_mean, residual_mean))
        sse_med += float(np.dot(residual_med, residual_med))
        n += y.shape[0]

    return {
        "mae_mean_baseline": float(sae_mean / max(n, 1)),
        "rmse_mean_baseline": float(np.sqrt(sse_mean / max(n, 1))),
        "mae_median_baseline": float(sae_med / max(n, 1)),
        "rmse_median_baseline": float(np.sqrt(sse_med / max(n, 1))),
        "rows": n,
    }


def _make_plots(samples: Dict[str, np.ndarray], output_path: Path) -> None:
    """
    Save residual diagnostics from sampled points.
    """
    y = samples["y"]
    pred = samples["pred"]
    abs_err = samples["abs_error"]
    residual = pred - y

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Scatter predicted vs actual.
    axes[0, 0].scatter(y, pred, s=2, alpha=0.3)
    axes[0, 0].set_xlabel("Actual delay (s)")
    axes[0, 0].set_ylabel("Predicted delay (s)")
    axes[0, 0].set_title("Predicted vs actual")
    axes[0, 0].set_xscale("symlog")
    axes[0, 0].set_yscale("symlog")

    # Residual histogram.
    axes[0, 1].hist(residual, bins=100, alpha=0.8)
    axes[0, 1].set_title("Residuals (pred - actual)")
    axes[0, 1].set_xlabel("Residual (s)")

    # Absolute error histogram.
    axes[1, 0].hist(abs_err, bins=100, alpha=0.8)
    axes[1, 0].set_title("Absolute errors")
    axes[1, 0].set_xlabel("|error| (s)")
    axes[1, 0].set_yscale("log")

    # Error vs actual.
    axes[1, 1].scatter(y, residual, s=2, alpha=0.3)
    axes[1, 1].set_xscale("symlog")
    axes[1, 1].set_ylabel("Residual (s)")
    axes[1, 1].set_xlabel("Actual delay (s)")
    axes[1, 1].set_title("Residuals vs actual")

    # CDF of absolute errors.
    sorted_abs = np.sort(abs_err)
    cdf_y = np.linspace(0, 1, len(sorted_abs), endpoint=False)
    axes[0, 2].plot(sorted_abs, cdf_y)
    axes[0, 2].set_xscale("symlog")
    axes[0, 2].set_ylabel("CDF")
    axes[0, 2].set_xlabel("|error| (s)")
    axes[0, 2].set_title("Absolute error CDF")

    # Absolute error quantiles as bars.
    quantile_levels = [50, 90, 95, 99]
    quantile_vals = [float(np.percentile(abs_err, q)) for q in quantile_levels] if abs_err.size else [0] * len(quantile_levels)
    axes[1, 2].bar([str(q) for q in quantile_levels], quantile_vals)
    axes[1, 2].set_yscale("symlog")
    axes[1, 2].set_ylabel("|error| (s)")
    axes[1, 2].set_title("Absolute error quantiles")

    # Predicted/actual distributions.
    fig_dist, ax_dist = plt.subplots(figsize=(7, 4))
    ax_dist.hist(y, bins=100, alpha=0.5, label="actual")
    ax_dist.hist(pred, bins=100, alpha=0.5, label="pred")
    ax_dist.set_xscale("symlog")
    ax_dist.set_yscale("log")
    ax_dist.legend()
    ax_dist.set_title("Predicted vs actual distribution")
    dist_path = output_path.with_name(output_path.stem + "_dist.png")
    fig_dist.tight_layout()
    fig_dist.savefig(dist_path, dpi=150)
    plt.close(fig_dist)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _decile_summary(samples: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Summaries over sampled points: actual/pred quantiles and abs-error quantiles.
    """
    y = samples["y"]
    pred = samples["pred"]
    abs_err = samples["abs_error"]
    quantiles = [10, 25, 50, 75, 90, 95, 99]

    def q(arr):
        return {f"p{p}": float(np.percentile(arr, p)) for p in quantiles}

    return {
        "actual_quantiles": q(y) if y.size else {},
        "pred_quantiles": q(pred) if pred.size else {},
        "abs_error_quantiles": q(abs_err) if abs_err.size else {},
    }


def _feature_importance(booster: xgb.Booster) -> Dict[str, list]:
    """
    Get top-k feature importances by gain/weight/cover.
    """

    def topk(importance: Dict[str, float]) -> list:
        return [
            {"feature": k, "score": float(v)}
            for k, v in sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K_IMPORTANCE]
        ]

    return {
        "gain": topk(booster.get_score(importance_type="gain")),
        "weight": topk(booster.get_score(importance_type="weight")),
        "cover": topk(booster.get_score(importance_type="cover")),
    }


def evaluate(
    features_path: Path,
    model_path: Path,
    clip_target: float | None,
    batch_size: int,
    sample_cap: int,
    log1p_target: bool,
) -> Path:
    """
    Evaluate model on test split and random sample; save metrics and plots.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(features_path, format="parquet")
    booster = _load_booster(model_path)

    train_stats = _baseline_stats(dataset, clip_target=clip_target)

    test_metrics, test_samples = _stream_metrics(
        booster,
        dataset,
        filter_expr=ds.field("split") == "test",
        batch_size=batch_size,
        clip_target=clip_target,
        sample_cap=sample_cap,
        log1p_target=log1p_target,
    )
    baseline_test = _baseline_metrics(
        dataset,
        filter_expr=ds.field("split") == "test",
        batch_size=batch_size,
        clip_target=clip_target,
        train_mean=train_stats["train_mean"],
        train_median=train_stats["train_median"],
    )

    # Random sample over all splits for sanity (10% reservoir up to sample_cap).
    random_metrics, random_samples = _stream_metrics(
        booster,
        dataset,
        filter_expr=None,
        batch_size=batch_size,
        clip_target=clip_target,
        sample_cap=sample_cap,
        log1p_target=log1p_target,
    )

    results = {
        "clip_target": clip_target,
        "train_stats": train_stats,
        "test_metrics": test_metrics,
        "baseline_test": baseline_test,
        "random_sample_metrics": random_metrics,
        "test_deciles": _decile_summary(test_samples),
        "random_deciles": _decile_summary(random_samples),
        "feature_importance": _feature_importance(booster),
    }
    metrics_path = ARTIFACTS_DIR / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    # Plots on test sample.
    plot_path = ARTIFACTS_DIR / "evaluation_plots.png"
    _make_plots(test_samples, plot_path)
    # Additional plot on random sample.
    plot_random_path = ARTIFACTS_DIR / "evaluation_random_plots.png"
    _make_plots(random_samples, plot_random_path)

    return metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained model and generate metrics/plots."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ARTIFACTS_DIR / "features.parquet",
        help="Path to feature parquet.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ARTIFACTS_DIR / "xgb_queue_delay.json",
        help="Path to trained XGBoost model.",
    )
    parser.add_argument(
        "--clip-target",
        type=float,
        default=7200.0,
        help="Clip applied to labels/predictions for evaluation (seconds).",
    )
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument(
        "--sample-cap",
        type=int,
        default=200_000,
        help="Maximum number of sampled points for plots/quantiles.",
    )
    parser.add_argument(
        "--no-log1p-target",
        action="store_true",
        help="Disable log1p inversion (use if model was trained on raw target).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = evaluate(
        features_path=args.features,
        model_path=args.model,
        clip_target=args.clip_target,
        batch_size=args.batch_size,
        sample_cap=args.sample_cap,
        log1p_target=not args.no_log1p_target,
    )
    print(f"Wrote evaluation to {metrics_path}")
    print(f"Plots: {ARTIFACTS_DIR / 'evaluation_plots.png'} and {ARTIFACTS_DIR / 'evaluation_random_plots.png'}")


if __name__ == "__main__":
    main()
