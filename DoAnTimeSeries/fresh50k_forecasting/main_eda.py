"""Run Fresh50K EDA, stationarity checks, and diagnostic plots."""

import argparse

from src.config import PROCESSED_DATA_DIR, TABLES_DIR, ensure_directories
from src.data_loader import (
    create_data_quality_report,
    expand_daily_to_hourly,
    hourly_sample_output_path,
    load_fresh50k,
    sample_by_series,
    sample_output_path,
)
from src.eda import run_eda
from src.stationarity import run_stationarity_phase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fresh50K EDA pipeline.")
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=0.1,
        help="Fraction of series_id values to keep for EDA. Use 1.0 for full data.",
    )
    parser.add_argument(
        "--frequency",
        choices=["hourly", "daily"],
        default="hourly",
        help="Run EDA on hourly-expanded data or daily rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    base_df = load_fresh50k(output_path=PROCESSED_DATA_DIR / "fresh50k_base.parquet")
    daily_sample = sample_by_series(
        base_df,
        sample_frac=args.sample_frac,
        seed=args.seed,
        output_path=sample_output_path(args.sample_frac),
    )
    if args.frequency == "hourly":
        df = expand_daily_to_hourly(
            daily_sample,
            output_path=hourly_sample_output_path(args.sample_frac),
        )
        frequency_label = "Hourly rows expanded from hours_sale and hours_stock_status"
        sample_path = hourly_sample_output_path(args.sample_frac)
    else:
        df = daily_sample
        frequency_label = "Daily rows with 24-hour sale/status sequences"
        sample_path = sample_output_path(args.sample_frac)

    create_data_quality_report(
        df,
        output_path=TABLES_DIR / "data_quality_summary.csv",
        sample_frac=args.sample_frac,
        frequency=frequency_label,
    )
    representatives = run_eda(df)
    run_stationarity_phase(df, representatives)
    print(f"Saved normalized data: {PROCESSED_DATA_DIR / 'fresh50k_base.parquet'}")
    print(f"Saved daily series-level sample: {sample_output_path(args.sample_frac)}")
    print(f"Using {args.frequency} sample: {sample_path}")
    print(f"Saved data quality report: {TABLES_DIR / 'data_quality_summary.csv'}")
    print("Saved Phase 2 EDA tables and figures.")
    print("Saved Phase 3 stationarity tables and ACF/PACF figures.")


if __name__ == "__main__":
    main()
