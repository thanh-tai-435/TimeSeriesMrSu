"""Run Fresh50K Phase 6 benchmark models."""

from __future__ import annotations

import argparse

from src.benchmarks import run_benchmarks
from src.config import PROCESSED_DATA_DIR, TABLES_DIR, ensure_directories
from src.split import load_feature_table, time_based_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fresh50K naive benchmark models.")
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
    _, _, test_df = time_based_split(
        df,
        val_days=args.val_days,
        test_days=args.test_days,
        output_path=TABLES_DIR / "split_summary.csv",
    )
    metrics = run_benchmarks(test_df, tables_dir=TABLES_DIR)
    print(f"Loaded feature table: {args.features_path}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Saved benchmark predictions to: {TABLES_DIR}")
    print(f"Saved benchmark metrics: {TABLES_DIR / 'benchmark_metrics.csv'}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
