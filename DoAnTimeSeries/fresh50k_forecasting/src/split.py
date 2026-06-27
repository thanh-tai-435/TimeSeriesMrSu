"""Time-based train/validation/test split utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROCESSED_DATA_DIR, TABLES_DIR


def _split_summary(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, split_df in split_frames.items():
        rows.append(
            {
                "Split": split_name,
                "Start date": split_df["dt"].min(),
                "End date": split_df["dt"].max(),
                "Rows": len(split_df),
                "Series": split_df["series_id"].nunique() if "series_id" in split_df.columns else None,
            }
        )
    return pd.DataFrame(rows)


def save_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Save train/validation/test split summary."""
    summary = _split_summary({"Train": train_df, "Validation": val_df, "Test": test_df})
    output = Path(output_path) if output_path is not None else TABLES_DIR / "split_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return summary


def time_based_split(
    df: pd.DataFrame,
    val_days: int = 7,
    test_days: int = 14,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by timestamp without random shuffling."""
    if "dt" not in df.columns:
        raise ValueError("Dataframe must contain a 'dt' column.")
    if val_days <= 0 or test_days <= 0:
        raise ValueError("val_days and test_days must be positive.")

    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt")

    max_dt = df["dt"].max()
    test_start = max_dt - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train_df = df[df["dt"] < val_start]
    val_df = df[(df["dt"] >= val_start) & (df["dt"] < test_start)]
    test_df = df[df["dt"] >= test_start]

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "One or more splits are empty. Use smaller val_days/test_days or a longer time range."
        )
    if train_df["dt"].max() >= val_df["dt"].min() or val_df["dt"].max() >= test_df["dt"].min():
        raise ValueError("Time splits overlap.")

    save_split_summary(train_df, val_df, test_df, output_path)
    return train_df, val_df, test_df


def load_feature_table(path: str | Path | None = None) -> pd.DataFrame:
    """Load the engineered feature table."""
    feature_path = Path(path) if path is not None else PROCESSED_DATA_DIR / "fresh50k_features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_path}")
    return pd.read_parquet(feature_path)
