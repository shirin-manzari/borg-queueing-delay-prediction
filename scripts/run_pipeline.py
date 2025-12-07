from __future__ import annotations

import argparse
from pathlib import Path

from scripts.build_features import build_feature_table
from scripts.train_model import train
from scripts.validate_raw import run_validation
from scripts.evaluate_model import evaluate
from scripts.common import ARTIFACTS_DIR
from scripts.pretrain_analysis import analyze


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for Google cluster queue delay modeling."
    )
    parser.add_argument("--skip-validate", action="store_true", help="Skip raw data validation stage")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature build stage")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training stage")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training evaluation stage")
    parser.add_argument("--skip-pretrain-analysis", action="store_true", help="Skip pre-training analysis stage")
    parser.add_argument("--features-path", type=Path, default=None, help="Optional precomputed features parquet path")
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-child-weight", type=float, default=2.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--clip-target", type=float, default=7200.0)
    parser.add_argument("--no-log1p-target", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features_path = args.features_path

    if not args.skip_validate:
        validation_path = run_validation()
        print(f"Validation report: {validation_path}")

    if not args.skip_features:
        features_path = build_feature_table()
        print(f"Built features at: {features_path}")
    if features_path is None:
        raise SystemExit("Feature path not provided and --skip-features was set.")

    if not args.skip_pretrain_analysis:
        pretrain_metrics = analyze(
            features_path=features_path,
            clip_target=args.clip_target,
            sample_cap=500_000,
        )
        print(f"Pre-training analysis written to {pretrain_metrics}")

    if args.skip_train:
        return

    _, metrics = train(
        features_path=features_path,
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
    print(f"Training complete. Metrics: {metrics}")

    if args.skip_eval:
        return

    eval_path = evaluate(
        features_path=features_path,
        model_path=Path(ARTIFACTS_DIR / "xgb_queue_delay.json"),
        clip_target=args.clip_target,
        batch_size=args.batch_size,
        sample_cap=200_000,
        log1p_target=not args.no_log1p_target,
    )
    print(f"Evaluation complete. Metrics: {eval_path}")
    print(f"Additional plots in artifacts/evaluation_plots.png, artifacts/evaluation_random_plots.png, and their *_dist variants.")


if __name__ == "__main__":
    main()
