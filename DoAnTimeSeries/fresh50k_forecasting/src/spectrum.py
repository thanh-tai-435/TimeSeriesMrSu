"""Frequency-domain diagnostics for seasonality validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram

from .config import FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _prepare_signal(series: pd.Series) -> np.ndarray:
    values = series.astype(float).to_numpy()
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = values - values.mean()
    std = values.std()
    if std > 0:
        values = values / std
    return values


def _spectrum(values: np.ndarray, sample_spacing: float) -> pd.DataFrame:
    freq, power = periodogram(values, fs=1.0 / sample_spacing, detrend="linear", scaling="spectrum")
    result = pd.DataFrame({"frequency": freq, "power": power})
    result = result[result["frequency"] > 0].copy()
    result["period"] = 1.0 / result["frequency"]
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def _top_peaks(spec: pd.DataFrame, period_unit: str, min_period: float, max_period: float, top_n: int = 8) -> pd.DataFrame:
    window = spec[(spec["period"] >= min_period) & (spec["period"] <= max_period)].copy()
    if window.empty:
        return pd.DataFrame(columns=["Rank", "Period", "Frequency", "Power", "Period unit"])
    peaks, _ = find_peaks(window["power"].to_numpy())
    candidates = window.iloc[peaks] if len(peaks) else window
    top = candidates.sort_values("power", ascending=False).head(top_n).copy()
    top.insert(0, "Rank", range(1, len(top) + 1))
    top["Period unit"] = period_unit
    top = top.rename(columns={"period": "Period", "frequency": "Frequency", "power": "Power"})
    return top[["Rank", "Period", "Frequency", "Power", "Period unit"]]


def _plot_period_spectrum(
    spec: pd.DataFrame,
    output_path: Path,
    title: str,
    xlabel: str,
    expected_periods: dict[float, str],
    min_period: float,
    max_period: float,
) -> None:
    plot_df = spec[(spec["period"] >= min_period) & (spec["period"] <= max_period)].sort_values("period")
    plt.figure(figsize=(11, 5))
    plt.plot(plot_df["period"], plot_df["power"], color="#2563eb", linewidth=1.2)
    for period, label in expected_periods.items():
        plt.axvline(period, color="#c2410c", linestyle="--", linewidth=1.1)
        ymax = float(plot_df["power"].max()) if not plot_df.empty else 1.0
        plt.text(period, ymax * 0.92, label, rotation=90, va="top", ha="right", color="#7f1d1d", fontsize=9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Power")
    plt.grid(alpha=0.25)
    _save(output_path)


def run_spectrum_analysis(
    features_path: Path = PROCESSED_DATA_DIR / "fresh50k_features.parquet",
    daily_recovered_path: Path = PROCESSED_DATA_DIR / "fresh50k_owner_daily_recovered_forecasting.parquet",
    figures_dir: Path = FIGURES_DIR,
    tables_dir: Path = TABLES_DIR,
) -> None:
    """Create spectrum plots and peak tables for hourly observed and daily recovered demand."""
    hourly = pd.read_parquet(features_path, columns=["dt", "sale_amount"])
    hourly["dt"] = pd.to_datetime(hourly["dt"])
    hourly_agg = hourly.groupby("dt", sort=True)["sale_amount"].sum().asfreq("h", fill_value=0)
    hourly_spec = _spectrum(_prepare_signal(hourly_agg), sample_spacing=1.0)
    hourly_peaks = _top_peaks(hourly_spec, period_unit="hours", min_period=2, max_period=240)
    hourly_peaks.to_csv(tables_dir / "spectrum_hourly_top_peaks.csv", index=False)
    _plot_period_spectrum(
        hourly_spec,
        figures_dir / "spectrum_hourly_aggregate_sales.png",
        "Hourly Aggregate Sales Spectrum",
        "Period (hours)",
        expected_periods={24: "24h", 168: "168h"},
        min_period=2,
        max_period=240,
    )

    daily = pd.read_parquet(daily_recovered_path, columns=["date", "recovered_daily"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily_agg = daily.groupby("date", sort=True)["recovered_daily"].sum().asfreq("D", fill_value=0)
    daily_spec = _spectrum(_prepare_signal(daily_agg), sample_spacing=1.0)
    daily_peaks = _top_peaks(daily_spec, period_unit="days", min_period=2, max_period=35)
    daily_peaks.to_csv(tables_dir / "spectrum_daily_recovered_top_peaks.csv", index=False)
    _plot_period_spectrum(
        daily_spec,
        figures_dir / "spectrum_daily_recovered_demand.png",
        "Daily Recovered Demand Spectrum",
        "Period (days)",
        expected_periods={7: "7d", 14: "14d"},
        min_period=2,
        max_period=35,
    )

