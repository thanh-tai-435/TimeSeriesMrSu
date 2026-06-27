"""Quality diagnostics for latent-demand imputation."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, TABLES_DIR
from .evaluation import evaluate_predictions
from .owner_approach import load_recovery_frame


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _sample_non_stockout_validation(df: pd.DataFrame, max_rows: int = 300_000) -> pd.DataFrame:
    max_dt = df["dt"].max()
    test_start = max_dt - pd.Timedelta(days=14) + pd.Timedelta(hours=1)
    val_start = test_start - pd.Timedelta(days=7)
    pool = df[(df["dt"] >= val_start) & (df["dt"] < test_start) & (df["stockout_flag"] == 0)].copy()
    if len(pool) > max_rows:
        pool = pool.sample(max_rows, random_state=42)
    return pool


def _plot_top_uplift(series_summary: pd.DataFrame) -> None:
    top = series_summary.sort_values("Recovered lift over observed", ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top["series_id"].astype(str), top["Recovered lift over observed"], color="#c2410c")
    plt.gca().invert_yaxis()
    plt.title("Top Series by Recovered Lift over Observed Sales")
    plt.xlabel("Recovered lift over observed")
    plt.ylabel("series_id")
    plt.grid(axis="x", alpha=0.25)
    _save(FIGURES_DIR / "imputation_top_series_lift.png")


def _plot_lift_distribution(series_summary: pd.DataFrame) -> None:
    lift = series_summary["Recovered lift over observed"].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(9, 5))
    plt.hist(lift.clip(upper=lift.quantile(0.99)), bins=80, color="#2563eb", alpha=0.85)
    plt.title("Distribution of Series-Level Recovered Lift")
    plt.xlabel("Recovered lift over observed")
    plt.ylabel("Series count")
    plt.grid(axis="y", alpha=0.25)
    _save(FIGURES_DIR / "imputation_series_lift_distribution.png")


def _plot_daily_uplift(daily: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(daily["date"], daily["observed_sales"], label="Observed sales", linewidth=1.3)
    plt.plot(daily["date"], daily["recovered_demand"], label="Recovered demand", linewidth=1.3)
    plt.bar(daily["date"], daily["lost_demand_recovered"], label="Recovered lost demand", alpha=0.25, color="#c2410c")
    plt.title("Daily Observed Sales vs Recovered Demand")
    plt.xlabel("Date")
    plt.ylabel("Aggregate demand")
    plt.legend()
    plt.grid(alpha=0.25)
    _save(FIGURES_DIR / "imputation_daily_uplift.png")


def run_imputation_quality_checks(
    features_path: Path = PROCESSED_DATA_DIR / "fresh50k_features.parquet",
    recovered_path: Path = PROCESSED_DATA_DIR / "fresh50k_recovered_hourly.parquet",
    recovery_model_path: Path = MODELS_DIR / "latent_demand_recovery_lightgbm.pkl",
) -> None:
    """Check whether imputation is accurate and whether it over-drives total demand."""
    recovered = pd.read_parquet(recovered_path)
    recovered["dt"] = pd.to_datetime(recovered["dt"])

    total_observed = float(recovered["sale_amount"].sum())
    total_recovered = float(recovered["recovered_demand"].sum())
    total_lost = float(recovered["lost_demand_recovered"].sum())
    raw_lost_col = "lost_demand_raw" if "lost_demand_raw" in recovered.columns else "lost_demand_recovered"
    total_raw_lost = float(recovered[raw_lost_col].sum())
    stockout_rows = recovered[recovered["stockout_flag"] == 1]
    uplift_summary = pd.DataFrame(
        [
            ("Observed sales", total_observed),
            ("Recovered demand", total_recovered),
            ("Recovered lost demand", total_lost),
            ("Recovered lift over observed", total_lost / max(total_observed, 1e-9)),
            ("Raw calibrated lost demand before cap", total_raw_lost),
            ("Cap retained share of raw lost demand", total_lost / max(total_raw_lost, 1e-9)),
            ("Stockout row share", float(recovered["stockout_flag"].mean())),
            ("Recovered lost demand from stockout rows", float(stockout_rows["lost_demand_recovered"].sum())),
            (
                "Mean stockout uplift per stockout row",
                float(stockout_rows["lost_demand_recovered"].mean()) if len(stockout_rows) else 0.0,
            ),
        ],
        columns=["Metric", "Value"],
    )
    uplift_summary.to_csv(TABLES_DIR / "imputation_uplift_summary.csv", index=False)

    series_summary = (
        recovered.groupby("series_id", as_index=False)
        .agg(
            observed_sales=("sale_amount", "sum"),
            recovered_demand=("recovered_demand", "sum"),
            recovered_lost_demand=("lost_demand_recovered", "sum"),
            stockout_rate=("stockout_flag", "mean"),
            stockout_rows=("stockout_flag", "sum"),
            n_rows=("sale_amount", "size"),
        )
    )
    series_summary["Recovered lift over observed"] = series_summary["recovered_lost_demand"] / series_summary[
        "observed_sales"
    ].clip(lower=1e-9)
    series_summary["Recovered share from imputation"] = series_summary["recovered_lost_demand"] / series_summary[
        "recovered_demand"
    ].clip(lower=1e-9)
    series_summary = series_summary.sort_values("Recovered lift over observed", ascending=False)
    series_summary.to_csv(TABLES_DIR / "imputation_series_uplift.csv", index=False)

    daily = (
        recovered.assign(date=recovered["dt"].dt.floor("D"))
        .groupby("date", as_index=False)
        .agg(
            observed_sales=("sale_amount", "sum"),
            recovered_demand=("recovered_demand", "sum"),
            lost_demand_recovered=("lost_demand_recovered", "sum"),
            stockout_rate=("stockout_flag", "mean"),
        )
    )
    daily["Recovered lift over observed"] = daily["lost_demand_recovered"] / daily["observed_sales"].clip(lower=1e-9)
    daily.to_csv(TABLES_DIR / "imputation_daily_uplift.csv", index=False)

    df, recovery_features = load_recovery_frame(features_path)
    df["dt"] = pd.to_datetime(df["dt"])
    validation = _sample_non_stockout_validation(df)
    model = joblib.load(recovery_model_path)
    validation_pred = validation[["dt", "series_id", "sale_amount"]].copy()
    validation_pred["imputed_demand"] = np.clip(model.predict(validation[recovery_features]), 0, None).astype("float32")
    validation_pred["error"] = validation_pred["sale_amount"] - validation_pred["imputed_demand"]
    validation_pred.to_csv(TABLES_DIR / "imputation_pseudo_stockout_validation_predictions.csv", index=False)

    metrics = evaluate_predictions(validation_pred["sale_amount"], validation_pred["imputed_demand"])
    pseudo_summary = pd.DataFrame(
        [
            ("Validation rows", int(len(validation_pred))),
            ("Pseudo-stockout source", "non-stockout rows from validation period"),
            ("RMSE", metrics["RMSE"]),
            ("MAE", metrics["MAE"]),
            ("WAPE", metrics["WAPE"]),
            ("sMAPE", metrics["sMAPE"]),
            (
                "Mean prediction / actual ratio",
                float(validation_pred["imputed_demand"].sum() / max(validation_pred["sale_amount"].sum(), 1e-9)),
            ),
            ("Mean error actual-minus-imputed", float(validation_pred["error"].mean())),
        ],
        columns=["Metric", "Value"],
    )
    pseudo_summary.to_csv(TABLES_DIR / "imputation_pseudo_stockout_validation.csv", index=False)

    aggregate_rows = []
    for level_name, group_cols in [
        ("Hourly aggregate", ["dt"]),
        ("Daily aggregate", ["date"]),
        ("Series daily aggregate", ["series_id", "date"]),
    ]:
        eval_frame = validation_pred.copy()
        eval_frame["date"] = pd.to_datetime(eval_frame["dt"]).dt.floor("D")
        aggregate = (
            eval_frame.groupby(group_cols, as_index=False)
            .agg(
                actual_sales=("sale_amount", "sum"),
                imputed_demand=("imputed_demand", "sum"),
            )
        )
        aggregate_metrics = evaluate_predictions(aggregate["actual_sales"], aggregate["imputed_demand"])
        aggregate_rows.append(
            {
                "Validation level": level_name,
                **aggregate_metrics,
                "Prediction / actual ratio": float(
                    aggregate["imputed_demand"].sum() / max(aggregate["actual_sales"].sum(), 1e-9)
                ),
                "N": int(len(aggregate)),
            }
        )
    aggregate_validation = pd.DataFrame(aggregate_rows)
    aggregate_validation.to_csv(TABLES_DIR / "imputation_pseudo_stockout_aggregate_validation.csv", index=False)

    caps = [0.5, 0.75, 0.9, 1.0]
    sensitivity_rows = []
    baseline_lost = recovered[raw_lost_col].copy()
    positive_lost = baseline_lost[baseline_lost > 0]
    for cap in caps:
        cap_value = float(positive_lost.quantile(cap)) if len(positive_lost) else 0.0
        capped_lost = baseline_lost.clip(upper=cap_value)
        capped_recovered = recovered["sale_amount"] + capped_lost
        sensitivity_rows.append(
            {
                "Scenario": f"Cap lost-demand at q{int(cap * 100)}",
                "Cap value": cap_value,
                "Recovered demand": float(capped_recovered.sum()),
                "Recovered lost demand": float(capped_lost.sum()),
                "Recovered lift over observed": float(capped_lost.sum() / max(total_observed, 1e-9)),
                "Share of original recovered lost demand": float(capped_lost.sum() / max(baseline_lost.sum(), 1e-9)),
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(TABLES_DIR / "imputation_cap_sensitivity.csv", index=False)

    _plot_top_uplift(series_summary)
    _plot_lift_distribution(series_summary)
    _plot_daily_uplift(daily)
