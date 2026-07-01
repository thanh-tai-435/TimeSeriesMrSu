"""Quality diagnostics for latent-demand imputation."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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


def _write_substitution_and_velocity_diagnostics(recovered: pd.DataFrame, features_path: Path) -> None:
    schema_cols = pq.read_schema(features_path).names
    wanted = [
        "dt",
        "series_id",
        "sale_amount",
        "stockout_flag",
        "sale_roll_mean_3",
        "sale_roll_mean_24",
        "sale_velocity_ratio_3_24",
        "sale_momentum_3_6",
        "peer_sales_same_group",
        "peer_sales_roll_mean_24",
        "peer_sales_velocity_ratio_3_24",
        "peer_stockout_rate_same_group",
    ]
    cols = [column for column in wanted if column in schema_cols]
    if not {"dt", "series_id", "stockout_flag"}.issubset(cols):
        return

    features = pd.read_parquet(features_path, columns=cols)
    features["dt"] = pd.to_datetime(features["dt"])

    stockout = features[features["stockout_flag"] == 1].copy()
    non_stockout = features[features["stockout_flag"] == 0].copy()
    rows = []
    for label, frame in [("stockout_rows", stockout), ("non_stockout_rows", non_stockout)]:
        if frame.empty:
            continue
        rows.append(
            {
                "Segment": label,
                "Rows": len(frame),
                "Mean sale velocity ratio 3h/24h": float(frame.get("sale_velocity_ratio_3_24", pd.Series(0)).mean()),
                "Mean sale momentum 3h-6h": float(frame.get("sale_momentum_3_6", pd.Series(0)).mean()),
                "Mean peer sales same group": float(frame.get("peer_sales_same_group", pd.Series(0)).mean()),
                "Mean peer velocity ratio 3h/24h": float(frame.get("peer_sales_velocity_ratio_3_24", pd.Series(0)).mean()),
                "Mean peer stockout rate": float(frame.get("peer_stockout_rate_same_group", pd.Series(0)).mean()),
            }
        )
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(TABLES_DIR / "stockout_substitution_velocity_diagnostics.csv", index=False)

    if not diagnostics.empty:
        plot_df = diagnostics.set_index("Segment")
        metric_cols = [column for column in plot_df.columns if column != "Rows"]
        plot_df[metric_cols].T.plot(kind="bar", figsize=(10, 5), color=["#c2410c", "#059669"])
        plt.title("Stockout vs Non-stockout: Velocity and Peer Signals")
        plt.xlabel("Signal")
        plt.ylabel("Mean value")
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.25)
        _save(FIGURES_DIR / "stockout_substitution_velocity_diagnostics.png")

    if {"peer_sales_same_group", "peer_sales_roll_mean_24"}.issubset(features.columns):
        merged = recovered[["dt", "series_id", "lost_demand_recovered"]].merge(
            features[["dt", "series_id", "peer_sales_same_group", "peer_sales_roll_mean_24", "stockout_flag"]],
            on=["dt", "series_id"],
            how="left",
        )
        merged["peer_uplift_vs_24h"] = (
            merged["peer_sales_same_group"] - merged["peer_sales_roll_mean_24"]
        ).astype("float32")
        case = (
            merged[merged["stockout_flag"] == 1]
            .groupby("series_id", as_index=False)
            .agg(
                stockout_rows=("stockout_flag", "sum"),
                recovered_lost_demand=("lost_demand_recovered", "sum"),
                mean_peer_uplift_vs_24h=("peer_uplift_vs_24h", "mean"),
                mean_peer_sales_same_group=("peer_sales_same_group", "mean"),
            )
            .sort_values(["mean_peer_uplift_vs_24h", "recovered_lost_demand"], ascending=False)
            .head(30)
        )
        case.to_csv(TABLES_DIR / "stockout_peer_substitution_cases.csv", index=False)


def _read_recovery_metric(metric_name: str, default: float = 1.0) -> float:
    path = TABLES_DIR / "owner_latent_recovery_summary.csv"
    if not path.exists():
        return default
    summary = pd.read_csv(path)
    match = summary[summary["Metric"] == metric_name]
    if match.empty:
        return default
    try:
        return float(match["Value"].iloc[0])
    except (TypeError, ValueError):
        return default


def _read_recovery_timestamp(metric_name: str) -> pd.Timestamp | None:
    path = TABLES_DIR / "owner_latent_recovery_summary.csv"
    if not path.exists():
        return None
    summary = pd.read_csv(path)
    match = summary[summary["Metric"] == metric_name]
    if match.empty:
        return None
    value = pd.to_datetime(match["Value"].iloc[0], errors="coerce")
    if pd.isna(value):
        return None
    return value


def run_imputation_quality_checks(
    features_path: Path = PROCESSED_DATA_DIR / "fresh50k_features.parquet",
    recovered_path: Path = PROCESSED_DATA_DIR / "fresh50k_recovered_hourly.parquet",
    recovery_model_path: Path = MODELS_DIR / "latent_demand_recovery_lightgbm.pkl",
) -> None:
    """Check whether imputation is accurate and whether it over-drives total demand."""
    recovered = pd.read_parquet(recovered_path)
    recovered["dt"] = pd.to_datetime(recovered["dt"])
    _write_substitution_and_velocity_diagnostics(recovered, features_path)

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
    calibration_factor = max(_read_recovery_metric("Imputation calibration factor", 1.0), 1.0)
    validation_pred = validation[["dt", "series_id", "sale_amount"]].copy()
    raw_prediction = np.clip(model.predict(validation[recovery_features]), 0, None)
    validation_pred["imputed_demand"] = np.clip(raw_prediction / calibration_factor, 0, None).astype("float32")
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
    val_start = _read_recovery_timestamp("Validation start")
    if val_start is not None:
        cap_source_lost = baseline_lost[(recovered["dt"] < val_start) & (baseline_lost > 0)]
        cap_source = "train-period positive raw lost demand"
    else:
        cap_source_lost = baseline_lost[baseline_lost > 0]
        cap_source = "all positive raw lost demand"
    for cap in caps:
        cap_value = float(cap_source_lost.quantile(cap)) if len(cap_source_lost) else 0.0
        capped_lost = baseline_lost.clip(upper=cap_value)
        capped_recovered = recovered["sale_amount"] + capped_lost
        sensitivity_rows.append(
            {
                "Scenario": f"Cap lost-demand at q{int(cap * 100)}",
                "Cap source": cap_source,
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
