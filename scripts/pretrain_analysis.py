from __future__ import annotations

"""
Pre-training data analysis for the queueing delay project.
Generates summary statistics and plots from the raw feature table
before model training.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds

from scripts.common import ARTIFACTS_DIR, MODEL_FEATURE_COLUMNS

LABEL_COLUMN = "queue_delay_seconds"


def _quantiles(arr: np.ndarray, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)) -> Dict[str, float]:
    return {f"p{int(q*100)}": float(np.quantile(arr, q)) for q in qs}


def analyze(features_path: Path, clip_target: float | None, sample_cap: int) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(features_path, format="parquet")

    # Basic stats
    total_rows = dataset.count_rows()
    splits = dataset.to_table(columns=["split"]).to_pandas()["split"].value_counts().to_dict()

    # Sample for plots
    batches = dataset.to_batches(
        columns=MODEL_FEATURE_COLUMNS + [LABEL_COLUMN, "split"],
        batch_size=200_000,
    )
    sample_labels = []
    sample_features = {f: [] for f in MODEL_FEATURE_COLUMNS}
    sample_splits = []
    for batch in batches:
        df = batch.to_pandas()
        label = df.pop(LABEL_COLUMN).to_numpy()
        if clip_target is not None:
            label = np.minimum(label, clip_target)
        split = df.pop("split")
        for f in MODEL_FEATURE_COLUMNS:
            sample_features[f].extend(df[f].tolist())
        sample_labels.extend(label.tolist())
        sample_splits.extend(split.tolist())
        if len(sample_labels) >= sample_cap:
            break

    sample_labels_arr = np.array(sample_labels[:sample_cap])
    sample_split_arr = np.array(sample_splits[:sample_cap])

    # Quantiles by split
    quantiles_by_split = {}
    for s in ["train", "valid", "test"]:
        mask = sample_split_arr == s
        if mask.any():
            quantiles_by_split[s] = _quantiles(sample_labels_arr[mask])

    # Overall quantiles
    label_quantiles = _quantiles(sample_labels_arr)

    # Plots
    plots_dir = ARTIFACTS_DIR
    # Label distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sample_labels_arr, bins=200, log=True)
    ax.set_xscale("symlog")
    ax.set_xlabel("queue_delay_seconds")
    ax.set_ylabel("count")
    ax.set_title("Queue delay distribution (sampled)")
    label_plot = plots_dir / "pretrain_queue_delay_dist.png"
    fig.tight_layout()
    fig.savefig(label_plot, dpi=150)
    plt.close(fig)

    # Feature histograms (top few)
    feature_plots = {}
    for f in MODEL_FEATURE_COLUMNS[:8]:
        fig, ax = plt.subplots(figsize=(7, 4))
        arr = np.array(sample_features[f][:sample_cap])
        ax.hist(arr, bins=100, log=True)
        ax.set_title(f"Feature distribution: {f}")
        feat_path = plots_dir / f"pretrain_feature_{f}.png"
        fig.tight_layout()
        fig.savefig(feat_path, dpi=150)
        plt.close(fig)
        feature_plots[f] = str(feat_path)

    # Correlation heatmap (sample)
    try:
        import seaborn as sns  # type: ignore

        import pandas as pd

        df_corr = pd.DataFrame(sample_features).iloc[:sample_cap]
        corr = df_corr.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0)
        ax.set_title("Feature correlation (sample)")
        corr_path = plots_dir / "pretrain_feature_corr.png"
        fig.tight_layout()
        fig.savefig(corr_path, dpi=150)
        plt.close(fig)
    except Exception:
        corr_path = None

    # Save metrics
    metrics = {
        "rows_total": total_rows,
        "rows_by_split": splits,
        "label_quantiles_sample": label_quantiles,
        "label_quantiles_by_split_sample": quantiles_by_split,
        "clip_target": clip_target,
        "sample_size": int(min(len(sample_labels_arr), sample_cap)),
    }
    metrics_path = ARTIFACTS_DIR / "pretrain_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-training analysis for queue delay data.")
    parser.add_argument(
        "--features",
        type=Path,
        default=ARTIFACTS_DIR / "features.parquet",
        help="Path to features parquet.",
    )
    parser.add_argument("--clip-target", type=float, default=7200.0, help="Optional label clip for plots/stats.")
    parser.add_argument("--sample-cap", type=int, default=500_000, help="Max sampled rows for plots/stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = analyze(
        features_path=args.features,
        clip_target=args.clip_target,
        sample_cap=args.sample_cap,
    )
    print(f"Wrote pre-training analysis to {metrics_path}")
    print("Plots: pretrain_queue_delay_dist.png, pretrain_feature_*.png, pretrain_feature_corr.png")


if __name__ == "__main__":
    main()

