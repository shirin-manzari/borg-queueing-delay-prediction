from __future__ import annotations

"""
Training script (from scratch) for queueing-delay prediction using XGBoost.

Design goals based on explanation.md:
- Handles the full dataset by streaming Parquet batches (no in-memory load).
- Targets are clipped and log-transformed to reduce heavy tails.
- Uses histogram-based trees with early stopping and robust defaults.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pyarrow.dataset as ds
import xgboost as xgb

from scripts.common import ARTIFACTS_DIR, MODEL_FEATURE_COLUMNS

LABEL_COLUMN = "queue_delay_seconds"


def _transform_labels(series, clip_target: float | None, log1p_target: bool) -> np.ndarray:
    """
    Apply optional clipping and log1p transform for heavy-tailed targets.
    Returns a NumPy array suitable for feeding into XGBoost.
    """
    arr = series.to_numpy()
    if clip_target is not None:
        arr = np.minimum(arr, clip_target)
    if log1p_target:
        arr = np.log1p(arr)
    return arr


class ParquetIterator(xgb.DataIter):
    """
    Stream Parquet batches into XGBoost without loading the full dataset.
    """

    def __init__(
        self,
        path: Path,
        split: str,
        feature_cols: Iterable[str],
        label_col: str,
        batch_size: int,
        clip_target: float | None,
        log1p_target: bool,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.split = split
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.batch_size = batch_size
        self.clip_target = clip_target
        self.log1p_target = log1p_target
        self.dataset = ds.dataset(self.path, format="parquet")
        self._batches = None
        self.reset()

    def reset(self) -> None:
        flt = ds.field("split") == self.split
        cols = self.feature_cols + [self.label_col, "split"]
        self._batches = iter(
            self.dataset.to_batches(
                columns=cols, filter=flt, batch_size=self.batch_size
            )
        )

    def next(self, input_data) -> int:
        try:
            batch = next(self._batches)
        except StopIteration:
            return 0
        df = batch.to_pandas()
        labels = _transform_labels(
            df.pop(self.label_col),
            clip_target=self.clip_target,
            log1p_target=self.log1p_target,
        )
        df.drop(columns=["split"], inplace=True, errors="ignore")
        df = df[self.feature_cols]
        input_data(data=df, label=labels)
        return 1


def train(
    features_path: Path,
    num_boost_round: int = 800,
    learning_rate: float = 0.05,
    max_depth: int = 10,
    min_child_weight: float = 2.0,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    batch_size: int = 250_000,
    early_stopping_rounds: int = 50,
    clip_target: float = 7200.0,
    log1p_target: bool = True,
) -> Tuple[xgb.Booster, Dict[str, Dict[str, float]]]:
    """
    Train an XGBoost regressor with sensible defaults for the heavy-tailed
    queue-delay target. Returns the fitted booster and per-split metrics.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_iter = ParquetIterator(
        features_path,
        split="train",
        feature_cols=MODEL_FEATURE_COLUMNS,
        label_col=LABEL_COLUMN,
        batch_size=batch_size,
        clip_target=clip_target,
        log1p_target=log1p_target,
    )
    valid_iter = ParquetIterator(
        features_path,
        split="valid",
        feature_cols=MODEL_FEATURE_COLUMNS,
        label_col=LABEL_COLUMN,
        batch_size=max(100_000, batch_size // 2),
        clip_target=clip_target,
        log1p_target=log1p_target,
    )

    params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "tree_method": "hist",
        "max_bin": 256,
        "eta": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "lambda": 1.0,
        "alpha": 0.0,
        "verbosity": 1,
        "seed": 42,
    }

    dtrain = xgb.QuantileDMatrix(train_iter)
    dvalid = xgb.QuantileDMatrix(valid_iter)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )

    metrics = {
        "valid": evaluate_split(
            booster,
            features_path,
            "valid",
            batch_size=max(100_000, batch_size // 2),
            clip_target=clip_target,
            log1p_target=log1p_target,
        ),
        "test": evaluate_split(
            booster,
            features_path,
            "test",
            batch_size=max(100_000, batch_size // 2),
            clip_target=clip_target,
            log1p_target=log1p_target,
        ),
    }

    model_path = ARTIFACTS_DIR / "xgb_queue_delay.json"
    booster.save_model(model_path)
    metrics_path = ARTIFACTS_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return booster, metrics


def evaluate_split(
    booster: xgb.Booster,
    features_path: Path,
    split: str,
    batch_size: int,
    clip_target: float | None,
    log1p_target: bool,
) -> Dict[str, float]:
    """
    Compute RMSE and MAE in *seconds* on a given split by streaming batches.
    If trained on log1p targets, predictions are expm1-transformed back to seconds and clipped.
    """
    dataset = ds.dataset(features_path, format="parquet")
    flt = ds.field("split") == split
    cols = MODEL_FEATURE_COLUMNS + [LABEL_COLUMN, "split"]
    batches = dataset.to_batches(columns=cols, filter=flt, batch_size=batch_size)

    sse = 0.0
    sae = 0.0
    n = 0
    iteration_limit = (
        booster.best_iteration + 1
        if booster.best_iteration is not None
        else booster.num_boosted_rounds()
    )

    for batch in batches:
        df = batch.to_pandas()
        y_raw = df.pop(LABEL_COLUMN).to_numpy()
        if clip_target is not None:
            y_raw = np.minimum(y_raw, clip_target)
        df.drop(columns=["split"], inplace=True, errors="ignore")
        df = df[MODEL_FEATURE_COLUMNS]

        preds = booster.predict(
            xgb.DMatrix(df), iteration_range=(0, iteration_limit)
        )
        if log1p_target:
            if clip_target is not None:
                preds = np.minimum(preds, np.log1p(clip_target))
            preds = np.expm1(preds)
        if clip_target is not None:
            preds = np.minimum(preds, clip_target)
            y_raw = np.minimum(y_raw, clip_target)

        y_eval = y_raw

        residuals = preds - y_eval
        sse += float(np.dot(residuals, residuals))
        sae += float(np.abs(residuals).sum())
        n += y_eval.shape[0]

    rmse = float(np.sqrt(sse / max(n, 1)))
    mae = float(sae / max(n, 1))
    return {"split": split, "rmse_seconds": rmse, "mae_seconds": mae, "rows": n}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost regressor on queue delay (Google trace)."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ARTIFACTS_DIR / "features.parquet",
        help="Path to the features Parquet dataset generated by build_features.py",
    )
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-child-weight", type=float, default=2.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--clip-target",
        type=float,
        default=7200.0,
        help="Upper clip (seconds) applied to labels.",
    )
    parser.add_argument(
        "--no-log1p-target",
        action="store_true",
        help="Disable log1p transform; use raw clipped seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    booster, metrics = train(
        features_path=args.features,
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        batch_size=args.batch_size,
        early_stopping_rounds=args.early_stopping_rounds,
        clip_target=args.clip_target,
        log1p_target=not args.no_log1p_target,
    )
    trees = (
        booster.best_iteration + 1
        if booster.best_iteration is not None
        else booster.num_boosted_rounds()
    )
    print(f"Saved model with {trees} trees")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
