"""Train Fresh50K benchmark and ML forecasting models."""

import argparse

import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, TABLES_DIR, ensure_directories
from src.models import load_feature_columns, train_lightgbm_model, train_ridge_model
from src.split import load_feature_table, time_based_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fresh50K forecasting models.")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 24])
    parser.add_argument("--sample_frac", type=float, default=0.1)
    parser.add_argument("--val_days", type=int, default=7)
    parser.add_argument("--test_days", type=int, default=14)
    parser.add_argument("--max_train_rows", type=int, default=1_000_000)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["ridge", "lightgbm"],
        default=["ridge", "lightgbm"],
    )
    parser.add_argument(
        "--features_path",
        default=str(PROCESSED_DATA_DIR / "fresh50k_features.parquet"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    feature_columns = load_feature_columns(TABLES_DIR / "feature_columns.csv")
    required_columns = [
        "dt",
        "series_id",
        "sale_amount",
        *feature_columns,
        f"target_h{args.horizon}",
        f"target_stockout_flag_h{args.horizon}",
    ]
    df = load_feature_table(args.features_path)
    df = df[[column for column in required_columns if column in df.columns]]
    train_df, val_df, test_df = time_based_split(
        df,
        val_days=args.val_days,
        test_days=args.test_days,
        output_path=TABLES_DIR / "split_summary.csv",
    )

    metrics = []
    if "ridge" in args.models:
        _, _, ridge_metrics = train_ridge_model(
            train_df,
            test_df,
            feature_columns,
            horizon=args.horizon,
            max_train_rows=args.max_train_rows,
            models_dir=MODELS_DIR,
            tables_dir=TABLES_DIR,
        )
        metrics.append(ridge_metrics)

    if "lightgbm" in args.models:
        _, _, lgbm_metrics = train_lightgbm_model(
            train_df,
            val_df,
            test_df,
            feature_columns,
            horizon=args.horizon,
            max_train_rows=args.max_train_rows,
            models_dir=MODELS_DIR,
            tables_dir=TABLES_DIR,
        )
        metrics.append(lgbm_metrics)

    metrics_df = pd.DataFrame(metrics)
    metrics_path = TABLES_DIR / f"ml_metrics_h{args.horizon}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Loaded feature table: {args.features_path}")
    print(f"Horizon: {args.horizon}")
    print(f"Train rows: {len(train_df):,}; validation rows: {len(val_df):,}; test rows: {len(test_df):,}")
    print(f"Max train rows used per model: {args.max_train_rows:,}")
    print(f"Saved models to: {MODELS_DIR}")
    print(f"Saved ML metrics: {metrics_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
