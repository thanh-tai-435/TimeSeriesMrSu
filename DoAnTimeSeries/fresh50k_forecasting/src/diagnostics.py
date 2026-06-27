"""Diagnostics and forecast interval utilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, TABLES_DIR
from .evaluation import evaluate_predictions
from .models import load_feature_columns
from .split import load_feature_table, time_based_split


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _prediction_frame(model, df: pd.DataFrame, feature_columns: list[str], horizon: int) -> pd.DataFrame:
    target_col = f"target_h{horizon}"
    pred = df[["dt", "series_id", target_col]].rename(columns={target_col: "y_true"}).copy()
    pred["y_pred"] = model.predict(df[feature_columns]).astype("float32")
    pred["residual"] = pred["y_true"] - pred["y_pred"]
    return pred


def _residual_diagnostics(pred: pd.DataFrame, horizon: int, model_name: str) -> dict:
    residual = pred["residual"].dropna()
    aggregate_residual = pred.groupby("dt", sort=True)["residual"].mean()
    lb = acorr_ljungbox(aggregate_residual, lags=[24], return_df=True)
    jb = jarque_bera(residual.sample(min(len(residual), 500_000), random_state=42))
    metrics = evaluate_predictions(pred["y_true"], pred["y_pred"])
    return {
        "Horizon": horizon,
        "Model": model_name,
        **metrics,
        "Residual mean": float(residual.mean()),
        "Residual std": float(residual.std()),
        "Ljung-Box lag": 24,
        "Ljung-Box statistic": float(lb["lb_stat"].iloc[0]),
        "Ljung-Box p-value": float(lb["lb_pvalue"].iloc[0]),
        "Jarque-Bera statistic": float(jb.statistic),
        "Jarque-Bera p-value": float(jb.pvalue),
        "N": len(pred),
    }


def _plot_residuals(pred: pd.DataFrame, horizon: int, figures_dir: Path) -> None:
    residual = pred["residual"].dropna()
    sample = residual.sample(min(len(residual), 500_000), random_state=42)

    plt.figure(figsize=(9, 5))
    plt.hist(sample, bins=100, color="#2563eb", alpha=0.85)
    plt.title(f"LightGBM Residual Distribution h{horizon}")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.25)
    _save(figures_dir / f"residual_distribution_lightgbm_h{horizon}.png")

    aggregate_residual = pred.groupby("dt", sort=True)["residual"].mean()
    plt.figure(figsize=(10, 5))
    plot_acf(aggregate_residual, lags=168, zero=False, ax=plt.gca())
    plt.title(f"Aggregate Residual ACF - LightGBM h{horizon}")
    _save(figures_dir / f"residual_acf_lightgbm_h{horizon}.png")


def _prediction_intervals(
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    abs_residual = val_pred["residual"].abs().dropna()
    q80 = float(abs_residual.quantile(0.80))
    q95 = float(abs_residual.quantile(0.95))
    intervals = test_pred.copy()
    intervals["lower_80"] = (intervals["y_pred"] - q80).clip(lower=0)
    intervals["upper_80"] = intervals["y_pred"] + q80
    intervals["lower_95"] = (intervals["y_pred"] - q95).clip(lower=0)
    intervals["upper_95"] = intervals["y_pred"] + q95
    intervals["horizon"] = horizon
    return intervals


def _plot_forecast_interval(
    intervals: pd.DataFrame,
    representative_series: pd.DataFrame,
    horizon: int,
    figures_dir: Path,
) -> None:
    high_volume_id = representative_series.loc[
        representative_series["type"] == "high_volume",
        "series_id",
    ].iloc[0]
    data = intervals[intervals["series_id"] == high_volume_id].sort_values("dt").head(336)
    if data.empty:
        return
    plt.figure(figsize=(12, 5))
    x = data["dt"]
    plt.plot(x, data["y_true"], label="Actual", linewidth=1.4)
    plt.plot(x, data["y_pred"], label="LightGBM", linewidth=1.2)
    plt.fill_between(x, data["lower_80"], data["upper_80"], alpha=0.25, label="80% interval")
    plt.fill_between(x, data["lower_95"], data["upper_95"], alpha=0.15, label="95% interval")
    plt.title(f"LightGBM Forecast Intervals h{horizon}: {high_volume_id}")
    plt.xlabel("Time")
    plt.ylabel("Sale amount")
    plt.legend()
    plt.grid(alpha=0.25)
    _save(figures_dir / f"forecast_interval_lightgbm_h{horizon}.png")


def _ridge_coefficients(horizon: int, feature_columns: list[str], tables_dir: Path) -> None:
    path = MODELS_DIR / f"ridge_h{horizon}.pkl"
    if not path.exists():
        return
    model = joblib.load(path)
    ridge = model.named_steps["ridge"]
    coef = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": ridge.coef_,
            "abs_coefficient": np.abs(ridge.coef_),
            "horizon": horizon,
        }
    ).sort_values("abs_coefficient", ascending=False)
    coef.to_csv(tables_dir / f"ridge_coefficients_h{horizon}.csv", index=False)


def run_diagnostics(
    features_path: Path = PROCESSED_DATA_DIR / "fresh50k_features.parquet",
    tables_dir: Path = TABLES_DIR,
    figures_dir: Path = FIGURES_DIR,
    horizons: tuple[int, ...] = (1, 24),
) -> None:
    """Run residual diagnostics and empirical forecast intervals."""
    feature_columns = load_feature_columns(tables_dir / "feature_columns.csv")
    needed = ["dt", "series_id", *feature_columns]
    for horizon in horizons:
        needed.extend([f"target_h{horizon}", f"target_stockout_flag_h{horizon}"])
    df = load_feature_table(features_path)
    df = df[[column for column in needed if column in df.columns]]
    train_df, val_df, test_df = time_based_split(df, output_path=tables_dir / "split_summary.csv")

    rows = []
    reps = pd.read_csv(tables_dir / "representative_series.csv")
    for horizon in horizons:
        model_path = MODELS_DIR / f"lightgbm_h{horizon}.pkl"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        val_pred = _prediction_frame(model, val_df, feature_columns, horizon)
        test_pred = _prediction_frame(model, test_df, feature_columns, horizon)
        rows.append(_residual_diagnostics(test_pred, horizon, "LightGBM"))
        _plot_residuals(test_pred, horizon, figures_dir)

        intervals = _prediction_intervals(val_pred, test_pred, horizon)
        intervals.to_csv(tables_dir / f"prediction_intervals_lightgbm_h{horizon}.csv", index=False)
        _plot_forecast_interval(intervals, reps, horizon, figures_dir)
        _ridge_coefficients(horizon, feature_columns, tables_dir)

    pd.DataFrame(rows).to_csv(tables_dir / "residual_diagnostics.csv", index=False)
