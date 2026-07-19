"""Data loading utilities for Fresh50K."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SAMPLE_DATA_DIR, TABLES_DIR


WEATHER_RENAME_MAP = {
    "precpt": "precip",
    "avg_temperature": "temp",
    "avg_humidity": "humidity",
    "avg_wind_level": "wind",
}


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file format: {path}")


def _default_input_files(raw_dir: Path) -> list[Path]:
    files = []
    for pattern in ("*.parquet", "*.csv"):
        files.extend(sorted(raw_dir.glob(pattern)))
    return files


def _series_id(df: pd.DataFrame) -> pd.Series:
    id_cols = ["store_id", "product_id"]
    if "city_id" in df.columns:
        id_cols = ["city_id", *id_cols]
    return df[id_cols].astype(str).agg("_".join, axis=1)


def _stockout_rate_from_sequence(value) -> float:
    if value is None:
        return 0.0
    try:
        values = list(value)
    except TypeError:
        return float(value)
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _sequence_to_array(value, length: int = 24, dtype: str = "float32") -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=dtype)
    try:
        values = list(value)
    except TypeError:
        values = [value]
    if len(values) < length:
        values = values + [0] * (length - len(values))
    return np.asarray(values[:length], dtype=dtype)


def normalize_fresh50k(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns, dates, identifiers, and stockout flags."""
    df = df.copy()
    df = df.rename(columns=WEATHER_RENAME_MAP)

    if "dt" not in df.columns:
        raise ValueError("Fresh50K data must contain a 'dt' column.")
    if "sale_amount" not in df.columns:
        raise ValueError("Fresh50K data must contain a 'sale_amount' column.")
    if "store_id" not in df.columns or "product_id" not in df.columns:
        raise ValueError("Fresh50K data must contain 'store_id' and 'product_id'.")

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    if df["dt"].isna().any():
        bad_count = int(df["dt"].isna().sum())
        raise ValueError(f"Found {bad_count} rows with invalid dt values.")

    df["series_id"] = _series_id(df)

    if "hours_stock_status" in df.columns:
        stockout_rate = df["hours_stock_status"].map(_stockout_rate_from_sequence)
        df["stockout_rate"] = stockout_rate
        df["stockout_flag"] = (stockout_rate > 0).astype("int8")
    elif "stock_hour6_22_cnt" in df.columns:
        df["stockout_flag"] = (df["stock_hour6_22_cnt"].fillna(0) > 0).astype("int8")
        df["stockout_rate"] = df["stockout_flag"].astype(float)
    elif "stockout_flag" not in df.columns:
        df["stockout_flag"] = 0
        df["stockout_rate"] = 0.0

    sort_cols = [column for column in ["city_id", "store_id", "product_id", "dt"] if column in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def load_fresh50k(
    input_files: Iterable[str | Path] | None = None,
    output_path: str | Path | None = None,
    add_source_split: bool = True,
) -> pd.DataFrame:
    """Load, normalize, sort, and save Fresh50K base data."""
    paths = [Path(path) for path in input_files] if input_files is not None else _default_input_files(RAW_DATA_DIR)
    if not paths:
        raise FileNotFoundError(f"No parquet/csv files found in {RAW_DATA_DIR}")

    frames = []
    for path in paths:
        df = _read_table(path)
        if add_source_split:
            df["source_split"] = path.stem
        frames.append(df)

    base = normalize_fresh50k(pd.concat(frames, ignore_index=True))
    output = Path(output_path) if output_path is not None else PROCESSED_DATA_DIR / "fresh50k_base.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(output, index=False)
    return base


def sample_by_series(
    df: pd.DataFrame,
    sample_frac: float = 0.1,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Sample complete time series, preserving each selected series timeline."""
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in the interval (0, 1].")
    if "series_id" not in df.columns:
        raise ValueError("Data must contain 'series_id' before sampling.")

    if sample_frac >= 1:
        sampled = df.copy()
    else:
        series_ids = df["series_id"].drop_duplicates()
        sampled_ids = series_ids.sample(frac=sample_frac, random_state=seed)
        sampled = df[df["series_id"].isin(sampled_ids)].copy()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sampled.to_parquet(output, index=False)
    return sampled


def sample_output_path(sample_frac: float) -> Path:
    """Return a stable sample parquet path for a sample fraction."""
    percent = int(round(sample_frac * 100))
    return SAMPLE_DATA_DIR / f"fresh50k_sample_{percent:03d}.parquet"


def hourly_sample_output_path(sample_frac: float) -> Path:
    """Return a stable hourly sample parquet path for a sample fraction."""
    percent = int(round(sample_frac * 100))
    return SAMPLE_DATA_DIR / f"fresh50k_hourly_sample_{percent:03d}.parquet"


def expand_daily_to_hourly(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Expand Fresh50K daily rows with 24-hour sequences into hourly rows."""
    if "hours_sale" not in df.columns:
        raise ValueError("Hourly expansion requires the 'hours_sale' column.")

    df = df.sort_values(["series_id", "dt"]).reset_index(drop=True)
    n_rows = len(df)
    hours = np.tile(np.arange(24, dtype="int16"), n_rows)

    sale_values = np.concatenate([_sequence_to_array(value, dtype="float32") for value in df["hours_sale"]])
    if "hours_stock_status" in df.columns:
        stockout_values = np.concatenate(
            [_sequence_to_array(value, dtype="int8") for value in df["hours_stock_status"]]
        )
    else:
        stockout_values = np.repeat(df.get("stockout_flag", pd.Series(0, index=df.index)).to_numpy(dtype="int8"), 24)

    drop_columns = ["hours_sale", "hours_stock_status", "sale_amount", "stockout_flag", "stockout_rate"]
    scalar_columns = [column for column in df.columns if column not in drop_columns]
    hourly = df[scalar_columns].loc[df.index.repeat(24)].reset_index(drop=True)
    hourly["date"] = hourly["dt"]
    hourly["dt"] = hourly["dt"] + pd.to_timedelta(hours, unit="h")
    hourly["hour"] = hours
    hourly["daily_sale_amount"] = np.repeat(df["sale_amount"].to_numpy(dtype="float32"), 24)
    hourly["sale_amount"] = sale_values
    hourly["stockout_flag"] = stockout_values.astype("int8")
    hourly["stockout_rate"] = hourly["stockout_flag"].astype("float32")

    sort_cols = [column for column in ["city_id", "store_id", "product_id", "dt"] if column in hourly.columns]
    hourly = hourly.sort_values(sort_cols).reset_index(drop=True)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_parquet(output, index=False)
    return hourly


def create_data_quality_report(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    sample_frac: float | None = None,
    frequency: str | None = None,
) -> pd.DataFrame:
    """Create and save the data quality summary table."""
    duplicate_cols = [column for column in ["series_id", "dt"] if column in df.columns]
    missing_dt_count = int(df["dt"].isna().sum()) if "dt" in df.columns else 0
    duplicate_rows = int(df.duplicated(duplicate_cols).sum()) if duplicate_cols else int(df.duplicated().sum())

    metrics = [
        ("Sample fraction", sample_frac if sample_frac is not None else "full"),
        ("Number of rows", len(df)),
        ("Number of series", df["series_id"].nunique() if "series_id" in df.columns else None),
        ("Number of stores", df["store_id"].nunique() if "store_id" in df.columns else None),
        ("Number of products", df["product_id"].nunique() if "product_id" in df.columns else None),
        ("Number of cities", df["city_id"].nunique() if "city_id" in df.columns else None),
        ("Start date", df["dt"].min() if "dt" in df.columns else None),
        ("End date", df["dt"].max() if "dt" in df.columns else None),
        ("Frequency", frequency or "Daily rows with 24-hour sale/status sequences"),
        ("Missing dt count", missing_dt_count),
        ("Duplicate rows", duplicate_rows),
        ("Missing sale_amount", int(df["sale_amount"].isna().sum()) if "sale_amount" in df.columns else None),
        ("Stockout rate", float(df["stockout_flag"].mean()) if "stockout_flag" in df.columns else None),
        ("Average sale_amount", float(df["sale_amount"].mean()) if "sale_amount" in df.columns else None),
        ("Median sale_amount", float(df["sale_amount"].median()) if "sale_amount" in df.columns else None),
    ]
    report = pd.DataFrame(metrics, columns=["Metric", "Value"])

    output = Path(output_path) if output_path is not None else TABLES_DIR / "data_quality_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    return report
