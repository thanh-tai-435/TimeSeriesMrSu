"""EDA routines for Fresh50K time series."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, TABLES_DIR


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _as_sequence(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    try:
        return list(value)
    except TypeError:
        return []


def plot_aggregate_sales(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> None:
    total_sales = df.groupby("dt", sort=True)["sale_amount"].sum()

    plt.figure(figsize=(12, 5))
    plt.plot(total_sales.index, total_sales.values, linewidth=1.5)
    plt.title("Aggregate Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total sales")
    plt.grid(alpha=0.25)
    _save_current_figure(figures_dir / "aggregate_sales_over_time.png")


def plot_stockout_rate(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> None:
    stockout_col = "stockout_rate" if "stockout_rate" in df.columns else "stockout_flag"
    stockout_rate = df.groupby("dt", sort=True)[stockout_col].mean()

    plt.figure(figsize=(12, 5))
    plt.plot(stockout_rate.index, stockout_rate.values, color="#c2410c", linewidth=1.5)
    plt.title("Stockout Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average stockout rate")
    plt.ylim(bottom=0)
    plt.grid(alpha=0.25)
    _save_current_figure(figures_dir / "stockout_rate_over_time.png")


def plot_sale_distributions(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> None:
    sales = df["sale_amount"].dropna()

    plt.figure(figsize=(9, 5))
    plt.hist(sales, bins=80, color="#2563eb", alpha=0.85)
    plt.title("Sale Amount Distribution")
    plt.xlabel("Sale amount")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.25)
    _save_current_figure(figures_dir / "sale_amount_distribution.png")

    plt.figure(figsize=(9, 5))
    plt.hist(np.log1p(sales), bins=80, color="#059669", alpha=0.85)
    plt.title("Log1p Sale Amount Distribution")
    plt.xlabel("log1p(sale_amount)")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.25)
    _save_current_figure(figures_dir / "log_sale_amount_distribution.png")


def plot_seasonality(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> None:
    tmp = df[["dt", "sale_amount"]].copy()
    tmp["day_of_week"] = tmp["dt"].dt.dayofweek
    dow_sales = tmp.groupby("day_of_week", sort=True)["sale_amount"].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(dow_sales.index.astype(str), dow_sales.values, color="#7c3aed")
    plt.title("Average Sales by Day of Week")
    plt.xlabel("Day of week (0=Monday)")
    plt.ylabel("Average sales")
    plt.grid(axis="y", alpha=0.25)
    _save_current_figure(figures_dir / "sales_by_day_of_week.png")

    if "hours_sale" in df.columns:
        hourly_sum = np.zeros(24, dtype=float)
        hourly_count = np.zeros(24, dtype=float)
        for value in df["hours_sale"]:
            seq = _as_sequence(value)
            if len(seq) >= 24:
                arr = np.asarray(seq[:24], dtype=float)
                hourly_sum += arr
                hourly_count += 1
        hourly_avg = np.divide(hourly_sum, hourly_count, out=np.zeros_like(hourly_sum), where=hourly_count > 0)
    else:
        tmp["hour"] = tmp["dt"].dt.hour
        hourly_avg = tmp.groupby("hour", sort=True)["sale_amount"].mean().reindex(range(24), fill_value=0).values

    plt.figure(figsize=(10, 5))
    plt.bar(range(24), hourly_avg, color="#0891b2")
    plt.title("Average Sales by Hour of Day")
    plt.xlabel("Hour of day")
    plt.ylabel("Average sales")
    plt.xticks(range(24))
    plt.grid(axis="y", alpha=0.25)
    _save_current_figure(figures_dir / "sales_by_hour_of_day.png")


def select_representative_series(
    df: pd.DataFrame,
    tables_dir: Path = TABLES_DIR,
) -> pd.DataFrame:
    stockout_col = "stockout_rate" if "stockout_rate" in df.columns else "stockout_flag"
    summary = (
        df.groupby("series_id", sort=False)
        .agg(
            total_sales=("sale_amount", "sum"),
            stockout_rate=(stockout_col, "mean"),
            n_obs=("sale_amount", "size"),
            mean_sales=("sale_amount", "mean"),
            zero_rate=("sale_amount", lambda x: float((x == 0).mean())),
        )
        .reset_index()
    )

    high = summary.sort_values(["total_sales", "n_obs"], ascending=[False, False]).iloc[0]
    intermittent_pool = summary[summary["series_id"] != high["series_id"]]
    intermittent = intermittent_pool.sort_values(
        ["zero_rate", "total_sales", "n_obs"],
        ascending=[False, True, False],
    ).iloc[0]
    used = {high["series_id"], intermittent["series_id"]}
    stockout_pool = summary[~summary["series_id"].isin(used)]
    stockout_heavy = stockout_pool.sort_values(
        ["stockout_rate", "total_sales", "n_obs"],
        ascending=[False, False, False],
    ).iloc[0]

    reps = pd.DataFrame(
        [
            {
                "series_id": high["series_id"],
                "type": "high_volume",
                "total_sales": high["total_sales"],
                "stockout_rate": high["stockout_rate"],
                "n_obs": high["n_obs"],
            },
            {
                "series_id": intermittent["series_id"],
                "type": "intermittent",
                "total_sales": intermittent["total_sales"],
                "stockout_rate": intermittent["stockout_rate"],
                "n_obs": intermittent["n_obs"],
            },
            {
                "series_id": stockout_heavy["series_id"],
                "type": "stockout_heavy",
                "total_sales": stockout_heavy["total_sales"],
                "stockout_rate": stockout_heavy["stockout_rate"],
                "n_obs": stockout_heavy["n_obs"],
            },
        ]
    )
    output_path = tables_dir / "representative_series.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reps.to_csv(output_path, index=False)
    return reps


def plot_representative_series(
    df: pd.DataFrame,
    representatives: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    name_map = {
        "high_volume": "representative_series_high_volume.png",
        "intermittent": "representative_series_intermittent.png",
        "stockout_heavy": "representative_series_stockout_heavy.png",
    }
    title_map = {
        "high_volume": "Representative High-Volume Series",
        "intermittent": "Representative Intermittent Series",
        "stockout_heavy": "Representative Stockout-Heavy Series",
    }

    for row in representatives.itertuples(index=False):
        series_df = df[df["series_id"] == row.series_id].sort_values("dt")
        plt.figure(figsize=(12, 5))
        plt.plot(series_df["dt"], series_df["sale_amount"], marker="o", markersize=2.5, linewidth=1.2)
        plt.title(f"{title_map[row.type]}: {row.series_id}")
        plt.xlabel("Date")
        plt.ylabel("Sale amount")
        plt.grid(alpha=0.25)
        _save_current_figure(figures_dir / name_map[row.type])


def run_eda(
    df: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    tables_dir: Path = TABLES_DIR,
) -> pd.DataFrame:
    """Run Phase 2 EDA tables and plots."""
    plot_aggregate_sales(df, figures_dir)
    plot_stockout_rate(df, figures_dir)
    plot_sale_distributions(df, figures_dir)
    plot_seasonality(df, figures_dir)
    representatives = select_representative_series(df, tables_dir)
    plot_representative_series(df, representatives, figures_dir)
    return representatives
