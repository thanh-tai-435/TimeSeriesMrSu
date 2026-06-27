"""Feature engineering for global ML forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR, SERIES_COL, TABLES_DIR, TARGET_COL


CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "week_of_year",
    "month",
    "sin_hour",
    "cos_hour",
    "sin_dayofweek",
    "cos_dayofweek",
]
LAG_FEATURES = [f"sale_lag_{lag}" for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]]
ROLLING_FEATURES = [
    "sale_roll_mean_3",
    "sale_roll_mean_6",
    "sale_roll_mean_12",
    "sale_roll_mean_24",
    "sale_roll_mean_168",
    "sale_roll_std_24",
    "sale_roll_std_168",
    "sale_roll_min_24",
    "sale_roll_max_24",
]
STOCKOUT_FEATURES = [
    "stockout_lag_1",
    "stockout_lag_2",
    "stockout_lag_3",
    "stockout_lag_24",
    "stockout_lag_168",
    "stockout_roll_sum_24",
    "stockout_roll_mean_24",
    "stockout_roll_sum_168",
    "stockout_roll_mean_168",
]
OPTIONAL_EXTERNAL_BASE_FEATURES = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precip",
    "temp",
    "humidity",
    "wind",
]
EXTERNAL_DERIVED_FEATURES = [
    "discount_lag_1",
    "discount_lag_24",
    "discount_roll_mean_24",
    "temp_lag_1",
    "temp_lag_24",
    "precip_lag_1",
]
TARGET_FEATURES = ["target_h1", "target_h24", "target_stockout_flag_h1", "target_stockout_flag_h24"]


def _sort_for_features(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [column for column in ["city_id", "store_id", "product_id", "dt"] if column in df.columns]
    if not sort_cols:
        sort_cols = [SERIES_COL, "dt"]
    return df.sort_values(sort_cols).reset_index(drop=True)


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage before creating wide feature tables."""
    for column in df.select_dtypes(include=["float64"]).columns:
        df[column] = df[column].astype("float32")
    for column in df.select_dtypes(include=["int64"]).columns:
        max_value = df[column].max()
        min_value = df[column].min()
        if min_value >= np.iinfo(np.int16).min and max_value <= np.iinfo(np.int16).max:
            df[column] = df[column].astype("int16")
        elif min_value >= np.iinfo(np.int32).min and max_value <= np.iinfo(np.int32).max:
            df[column] = df[column].astype("int32")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"])
    df["hour"] = df["dt"].dt.hour.astype("int16")
    df["day_of_week"] = df["dt"].dt.dayofweek.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["day_of_month"] = df["dt"].dt.day.astype("int16")
    df["week_of_year"] = df["dt"].dt.isocalendar().week.astype("int16")
    df["month"] = df["dt"].dt.month.astype("int16")
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add per-series lag features."""
    df = df.copy()
    lags = lags or [1, 2, 3, 6, 12, 24, 48, 72, 168]
    grouped_sales = df.groupby(SERIES_COL, sort=False)[TARGET_COL]
    for lag in lags:
        df[f"sale_lag_{lag}"] = grouped_sales.shift(lag).astype("float32")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe rolling features."""
    df = df.copy()
    shifted = df.groupby(SERIES_COL, sort=False)[TARGET_COL].shift(1)
    grouped_shifted = shifted.groupby(df[SERIES_COL], sort=False)

    for window in [3, 6, 12, 24, 168]:
        df[f"sale_roll_mean_{window}"] = (
            grouped_shifted.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).astype("float32")
        )
    for window in [24, 168]:
        df[f"sale_roll_std_{window}"] = (
            grouped_shifted.rolling(window, min_periods=2).std().reset_index(level=0, drop=True).astype("float32")
        )
    df["sale_roll_min_24"] = (
        grouped_shifted.rolling(24, min_periods=1).min().reset_index(level=0, drop=True).astype("float32")
    )
    df["sale_roll_max_24"] = (
        grouped_shifted.rolling(24, min_periods=1).max().reset_index(level=0, drop=True).astype("float32")
    )
    return df


def add_stockout_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stockout-aware features."""
    df = df.copy()
    if "stockout_flag" not in df.columns:
        if "stockout_rate" in df.columns:
            df["stockout_flag"] = (df["stockout_rate"].fillna(0) > 0).astype("int8")
        elif "stock_hour6_22_cnt" in df.columns:
            df["stockout_flag"] = (df["stock_hour6_22_cnt"].fillna(0) > 0).astype("int8")
        else:
            df["stockout_flag"] = 0

    grouped_stockout = df.groupby(SERIES_COL, sort=False)["stockout_flag"]
    for lag in [1, 2, 3, 24, 168]:
        df[f"stockout_lag_{lag}"] = grouped_stockout.shift(lag).astype("float32")

    shifted = grouped_stockout.shift(1)
    grouped_shifted = shifted.groupby(df[SERIES_COL], sort=False)
    for window in [24, 168]:
        df[f"stockout_roll_sum_{window}"] = (
            grouped_shifted.rolling(window, min_periods=1).sum().reset_index(level=0, drop=True).astype("float32")
        )
        df[f"stockout_roll_mean_{window}"] = (
            grouped_shifted.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).astype("float32")
        )
    return df


def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add promotion and weather features when available."""
    df = df.copy()
    grouped = df.groupby(SERIES_COL, sort=False)

    if "discount" in df.columns:
        df["discount_lag_1"] = grouped["discount"].shift(1).astype("float32")
        df["discount_lag_24"] = grouped["discount"].shift(24).astype("float32")
        shifted_discount = grouped["discount"].shift(1)
        df["discount_roll_mean_24"] = (
            shifted_discount.groupby(df[SERIES_COL], sort=False)
            .rolling(24, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

    if "temp" in df.columns:
        df["temp_lag_1"] = grouped["temp"].shift(1).astype("float32")
        df["temp_lag_24"] = grouped["temp"].shift(24).astype("float32")

    if "precip" in df.columns:
        df["precip_lag_1"] = grouped["precip"].shift(1).astype("float32")
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add forecasting targets."""
    df = df.copy()
    grouped = df.groupby(SERIES_COL, sort=False)
    df["target_h1"] = grouped[TARGET_COL].shift(-1).astype("float32")
    df["target_h24"] = grouped[TARGET_COL].shift(-24).astype("float32")
    if "stockout_flag" in df.columns:
        df["target_stockout_flag_h1"] = grouped["stockout_flag"].shift(-1).astype("float32")
        df["target_stockout_flag_h24"] = grouped["stockout_flag"].shift(-24).astype("float32")
    return df


def build_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns that exist in the dataframe."""
    candidate_columns = (
        CALENDAR_FEATURES
        + LAG_FEATURES
        + ROLLING_FEATURES
        + ["stockout_flag", "stockout_rate"]
        + STOCKOUT_FEATURES
        + OPTIONAL_EXTERNAL_BASE_FEATURES
        + EXTERNAL_DERIVED_FEATURES
    )
    id_category_columns = [
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
    ]
    return [column for column in id_category_columns + candidate_columns if column in df.columns]


def save_feature_columns(feature_columns: list[str], output_path: str | Path | None = None) -> pd.DataFrame:
    """Save the feature column list."""
    output = Path(output_path) if output_path is not None else TABLES_DIR / "feature_columns.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_df = pd.DataFrame({"feature": feature_columns})
    feature_df.to_csv(output, index=False)
    return feature_df


def run_feature_engineering(
    input_path: str | Path,
    output_path: str | Path | None = None,
    feature_columns_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run the complete Phase 4 feature engineering pipeline."""
    df = pd.read_parquet(input_path)
    df = _sort_for_features(df)
    df = _optimize_dtypes(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_stockout_features(df)
    df = add_external_features(df)
    df = add_targets(df)

    # Sequence columns are useful for raw EDA but too large for model tables.
    df = df.drop(columns=[column for column in ["hours_sale", "hours_stock_status"] if column in df.columns])
    df = df.loc[df["target_h1"].notna() & df["target_h24"].notna()]

    feature_columns = build_feature_columns(df)
    save_feature_columns(feature_columns, feature_columns_path)

    keep_columns = [
        column
        for column in ["dt", "series_id", "sale_amount", *feature_columns, *TARGET_FEATURES]
        if column in df.columns
    ]
    df = df.loc[:, keep_columns]

    output = Path(output_path) if output_path is not None else PROCESSED_DATA_DIR / "fresh50k_features.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return df
