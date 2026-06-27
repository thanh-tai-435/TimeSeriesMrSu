"""Plotting utilities for reports and slides."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR, TABLES_DIR


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(df[x].astype(str), df[y], color="#2563eb")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    _save(path)


def save_model_comparison_charts(tables_dir: Path = TABLES_DIR, figures_dir: Path = FIGURES_DIR) -> None:
    """Save model comparison charts."""
    h1_path = tables_dir / "model_comparison_h1.csv"
    if h1_path.exists():
        h1 = pd.read_csv(h1_path)
        _bar(h1, "Model", "WAPE", "Model Comparison WAPE h1", figures_dir / "model_comparison_wape_h1.png")
        _bar(h1, "Model", "RMSE", "Model Comparison RMSE h1", figures_dir / "model_comparison_rmse_h1.png")

    ablation_path = tables_dir / "ablation_results.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        h1 = ablation[ablation["Horizon"] == 1]
        if not h1.empty:
            _bar(h1, "Feature set", "WAPE", "Ablation WAPE h1", figures_dir / "ablation_wape_h1.png")

    segment_path = tables_dir / "stockout_segment_evaluation.csv"
    if segment_path.exists():
        segment = pd.read_csv(segment_path)
        subset = segment[
            (segment["Horizon"] == 1)
            & (segment["Model"].isin(["LightGBM", "Ridge"]))
            & (segment["Segment"].isin(["stockout_period", "non_stockout_period"]))
        ].copy()
        if not subset.empty:
            subset["Label"] = subset["Model"] + " - " + subset["Segment"]
            _bar(subset, "Label", "WAPE", "Stockout Segment WAPE h1", figures_dir / "stockout_segment_wape.png")


def save_forecast_examples(tables_dir: Path = TABLES_DIR, figures_dir: Path = FIGURES_DIR) -> None:
    """Save actual-vs-forecast plots for representative series."""
    reps_path = tables_dir / "representative_series.csv"
    lgbm_path = tables_dir / "predictions_lightgbm_h1.csv"
    seasonal_path = tables_dir / "predictions_seasonal_naive_168_h1.csv"
    if not (reps_path.exists() and lgbm_path.exists() and seasonal_path.exists()):
        return

    reps = pd.read_csv(reps_path)
    wanted = reps["series_id"].tolist()
    lgbm = pd.read_csv(lgbm_path, parse_dates=["dt"])
    seasonal = pd.read_csv(seasonal_path, parse_dates=["dt"])
    lgbm = lgbm[lgbm["series_id"].isin(wanted)]
    seasonal = seasonal[seasonal["series_id"].isin(wanted)][["dt", "series_id", "y_pred"]].rename(
        columns={"y_pred": "seasonal_pred"}
    )
    merged = lgbm.merge(seasonal, on=["dt", "series_id"], how="left")

    file_map = {
        "high_volume": "forecast_high_volume_series.png",
        "intermittent": "forecast_intermittent_series.png",
        "stockout_heavy": "forecast_stockout_heavy_series.png",
    }
    for row in reps.itertuples(index=False):
        data = merged[merged["series_id"] == row.series_id].sort_values("dt").head(336)
        if data.empty:
            continue
        plt.figure(figsize=(12, 5))
        plt.plot(data["dt"], data["y_true"], label="Actual", linewidth=1.4)
        plt.plot(data["dt"], data["y_pred"], label="LightGBM", linewidth=1.2)
        plt.plot(data["dt"], data["seasonal_pred"], label="Seasonal naive 168h", linewidth=1.2)
        plt.title(f"Forecast Example - {row.type}: {row.series_id}")
        plt.xlabel("Time")
        plt.ylabel("Sale amount")
        plt.legend()
        plt.grid(alpha=0.25)
        _save(figures_dir / file_map[row.type])


def save_feature_importance_plot(tables_dir: Path = TABLES_DIR, figures_dir: Path = FIGURES_DIR) -> None:
    """Save LightGBM top-20 feature importance chart."""
    path = tables_dir / "lightgbm_feature_importance.csv"
    if not path.exists():
        return
    importance = pd.read_csv(path).sort_values("importance", ascending=False).head(20)
    plt.figure(figsize=(10, 7))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1], color="#059669")
    plt.title("LightGBM Feature Importance Top 20")
    plt.xlabel("Importance")
    plt.grid(axis="x", alpha=0.25)
    _save(figures_dir / "lightgbm_feature_importance_top20.png")


def run_plots(tables_dir: Path = TABLES_DIR, figures_dir: Path = FIGURES_DIR) -> None:
    """Run Phase 10 plotting."""
    save_model_comparison_charts(tables_dir, figures_dir)
    save_forecast_examples(tables_dir, figures_dir)
    save_feature_importance_plot(tables_dir, figures_dir)
