"""Run Fresh50K Phase 5 time-based split."""

from __future__ import annotations

import argparse

from src.config import PROCESSED_DATA_DIR, TABLES_DIR, ensure_directories
from src.split import load_feature_table, time_based_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create time-based train/validation/test split summary.")
    parser.add_argument("--val_days", type=int, default=7)
    parser.add_argument("--test_days", type=int, default=14)
    parser.add_argument(
        "--features_path",
        default=str(PROCESSED_DATA_DIR / "fresh50k_features.parquet"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    df = load_feature_table(args.features_path)
    train_df, val_df, test_df = time_based_split(
        df,
        val_days=args.val_days,
        test_days=args.test_days,
        output_path=TABLES_DIR / "split_summary.csv",
    )
    print(f"Loaded feature table: {args.features_path}")
    print(f"Saved split summary: {TABLES_DIR / 'split_summary.csv'}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")


if __name__ == "__main__":
    main()
