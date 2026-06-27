"""Stationarity tests and ACF/PACF utilities."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

from .config import FIGURES_DIR, TABLES_DIR


def _clean_series(series: pd.Series) -> pd.Series:
    cleaned = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    return cleaned.astype(float)


def _safe_lags(series: pd.Series, requested_lags: int = 48) -> int:
    n_obs = len(_clean_series(series))
    if n_obs < 4:
        return 1
    return max(1, min(requested_lags, (n_obs // 2) - 1))


def run_adf_test(series: pd.Series) -> dict[str, float | str | None]:
    """Run an ADF test on a series."""
    cleaned = _clean_series(series)
    if len(cleaned) < 8 or cleaned.nunique() <= 1:
        return {"statistic": None, "p_value": None, "error": "series too short or constant"}
    try:
        statistic, p_value, *_ = adfuller(cleaned, autolag="AIC")
        return {"statistic": float(statistic), "p_value": float(p_value), "error": None}
    except Exception as exc:  # statsmodels can fail on near-constant series.
        return {"statistic": None, "p_value": None, "error": str(exc)}


def run_kpss_test(series: pd.Series) -> dict[str, float | str | None]:
    """Run a KPSS test on a series."""
    cleaned = _clean_series(series)
    if len(cleaned) < 8 or cleaned.nunique() <= 1:
        return {"statistic": None, "p_value": None, "error": "series too short or constant"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value, *_ = kpss(cleaned, regression="c", nlags="auto")
        return {"statistic": float(statistic), "p_value": float(p_value), "error": None}
    except Exception as exc:
        return {"statistic": None, "p_value": None, "error": str(exc)}


def _transformations(series: pd.Series) -> dict[str, pd.Series]:
    cleaned = _clean_series(series)
    return {
        "original": cleaned,
        "log1p": np.log1p(cleaned.clip(lower=0)),
        "first_difference": cleaned.diff().dropna(),
        "seasonal_difference_lag_24": cleaned.diff(24).dropna(),
    }


def _conclusion(adf_p_value, kpss_p_value) -> str:
    adf_stationary = adf_p_value is not None and adf_p_value < 0.05
    kpss_stationary = kpss_p_value is not None and kpss_p_value >= 0.05
    if adf_stationary and kpss_stationary:
        return "stationary"
    if not adf_stationary and not kpss_stationary:
        return "non-stationary"
    if adf_p_value is None or kpss_p_value is None:
        return "insufficient evidence"
    return "mixed evidence"


def run_stationarity_report(
    series_dict: dict[str, pd.Series],
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run ADF/KPSS tests for multiple series and save a CSV report."""
    rows = []
    for series_name, series in series_dict.items():
        for transformation_name, transformed in _transformations(series).items():
            adf_result = run_adf_test(transformed)
            kpss_result = run_kpss_test(transformed)
            rows.append(
                {
                    "Series": series_name,
                    "Transformation": transformation_name,
                    "ADF statistic": adf_result["statistic"],
                    "ADF p-value": adf_result["p_value"],
                    "KPSS statistic": kpss_result["statistic"],
                    "KPSS p-value": kpss_result["p_value"],
                    "Conclusion": _conclusion(adf_result["p_value"], kpss_result["p_value"]),
                    "N": len(_clean_series(transformed)),
                }
            )

    report = pd.DataFrame(rows)
    output = Path(output_path) if output_path is not None else TABLES_DIR / "stationarity_tests.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    return report


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_acf_pacf_plots(
    series: pd.Series,
    acf_path: str | Path,
    pacf_path: str | Path,
    title_prefix: str,
    requested_lags: int = 48,
) -> None:
    """Save ACF and PACF plots for one series."""
    cleaned = _clean_series(series)
    lags = _safe_lags(cleaned, requested_lags=requested_lags)

    plt.figure(figsize=(10, 5))
    plot_acf(cleaned, lags=lags, ax=plt.gca(), zero=False)
    plt.title(f"ACF - {title_prefix}")
    _save_current_figure(Path(acf_path))

    plt.figure(figsize=(10, 5))
    plot_pacf(cleaned, lags=lags, ax=plt.gca(), zero=False, method="ywm")
    plt.title(f"PACF - {title_prefix}")
    _save_current_figure(Path(pacf_path))


def build_stationarity_series(
    df: pd.DataFrame,
    representatives: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Build aggregate and representative series for stationarity testing."""
    series_dict: dict[str, pd.Series] = {
        "aggregate_hourly_sales": df.groupby("dt", sort=True)["sale_amount"].sum()
    }

    type_map = dict(zip(representatives["type"], representatives["series_id"]))
    for series_type, series_id in type_map.items():
        series_df = df[df["series_id"] == series_id].sort_values("dt")
        series_dict[f"{series_type}_series"] = series_df.set_index("dt")["sale_amount"]
    return series_dict


def run_stationarity_phase(
    df: pd.DataFrame,
    representatives: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    tables_dir: Path = TABLES_DIR,
) -> pd.DataFrame:
    """Run Phase 3 stationarity tests and ACF/PACF plots."""
    series_dict = build_stationarity_series(df, representatives)
    report = run_stationarity_report(series_dict, tables_dir / "stationarity_tests.csv")

    save_acf_pacf_plots(
        series_dict["aggregate_hourly_sales"],
        figures_dir / "acf_aggregate_sales.png",
        figures_dir / "pacf_aggregate_sales.png",
        "Aggregate Hourly Sales",
        requested_lags=168,
    )
    if "high_volume_series" in series_dict:
        save_acf_pacf_plots(
            series_dict["high_volume_series"],
            figures_dir / "acf_high_volume_series.png",
            figures_dir / "pacf_high_volume_series.png",
            "High-Volume Series",
            requested_lags=168,
        )
    return report
