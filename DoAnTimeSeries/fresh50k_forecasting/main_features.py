"""Run Fresh50K Phase 4 feature engineering."""

from __future__ import annotations

import argparse

from src.config import PROCESSED_DATA_DIR, TABLES_DIR, ensure_directories
from src.data_loader import hourly_sample_output_path, sample_output_path
from src.features import run_feature_engineering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Fresh50K feature table.")
    parser.add_argument("--sample_frac", type=float, default=0.1)
    parser.add_argument("--frequency", choices=["hourly", "daily"], default="hourly")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    input_path = (
        hourly_sample_output_path(args.sample_frac)
        if args.frequency == "hourly"
        else sample_output_path(args.sample_frac)
    )
    if not input_path.exists():
        raise FileNotFoundError(
            f"Sample file not found: {input_path}. "
            f"Run `python main_eda.py --sample_frac {args.sample_frac} --frequency {args.frequency}` first."
        )

    output_path = PROCESSED_DATA_DIR / "fresh50k_features.parquet"
    feature_columns_path = TABLES_DIR / "feature_columns.csv"
    features = run_feature_engineering(
        input_path=input_path,
        output_path=output_path,
        feature_columns_path=feature_columns_path,
    )

    print(f"Loaded sample: {input_path}")
    print(f"Saved feature table: {output_path}")
    print(f"Saved feature list: {feature_columns_path}")
    print(f"Feature rows: {len(features):,}")
    print(f"Feature columns: {features.shape[1]:,}")


if __name__ == "__main__":
    main()
