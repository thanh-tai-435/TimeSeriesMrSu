"""Owner-aligned two-stage censored demand modeling.

The FreshRetailNet-50K paper/repo frames the dataset around:
1. latent demand recovery during stockouts;
2. demand forecasting on recovered/de-biased demand.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, TABLES_DIR
from .evaluation import evaluate_predictions
from .models import load_feature_columns


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _wpe(y_true, y_pred) -> float:
    denominator = float(np.sum(y_true))
    if denominator == 0:
        return np.nan
    return float(np.sum(np.asarray(y_pred) - np.asarray(y_true)) / denominator)


def _write_model_feature_importance(model, features: list[str], output_stem: str, title: str) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    importance = (
        pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(TABLES_DIR / f"{output_stem}.csv", index=False)
    top = importance.head(20).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"], color="#0f766e")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.25)
    _save(FIGURES_DIR / f"{output_stem}_top20.png")


def load_recovery_frame(features_path: Path = PROCESSED_DATA_DIR / "fresh50k_features.parquet") -> pd.DataFrame:
    """Load only columns required for latent demand recovery."""
    feature_columns = load_feature_columns(TABLES_DIR / "feature_columns.csv")
    recovery_features = [
        col
        for col in feature_columns
        if col not in {"stockout_flag", "stockout_rate"}
        and not col.startswith("stockout_lag")
        and not col.startswith("stockout_roll")
    ]
    schema_cols = pq.read_schema(features_path).names
    needed = ["dt", "series_id", "sale_amount", "stockout_flag", *recovery_features]
    df = pd.read_parquet(features_path, columns=[col for col in needed if col in schema_cols])
    return df, recovery_features


def recover_hourly_demand(
    features_path: Path = PROCESSED_DATA_DIR / "fresh50k_features.parquet",
    max_train_rows: int = 400_000,
    val_days: int = 7,
    test_days: int = 14,
    warmup_days: int = 14,
    block_days: int = 7,
) -> pd.DataFrame:
    """Recover latent hourly demand with expanding-window, point-in-time blocks."""
    from lightgbm import LGBMRegressor

    df, recovery_features = load_recovery_frame(features_path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    max_dt = df["dt"].max()
    test_start = max_dt - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    val_start = test_start - pd.Timedelta(days=val_days)
    data_start = df["dt"].min()
    first_block_start = min(data_start + pd.Timedelta(days=warmup_days), val_start)

    df["imputed_demand"] = df["sale_amount"].astype("float32")
    trained_blocks = []
    last_model = None

    def fit_recovery_model(train_pool: pd.DataFrame) -> object:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=48,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(train_pool[recovery_features], train_pool["sale_amount"])
        return model

    block_start = first_block_start
    while block_start < val_start:
        block_end = min(block_start + pd.Timedelta(days=block_days), val_start)
        train_mask = (df["dt"] < block_start) & (df["stockout_flag"] == 0)
        train_pool = df.loc[train_mask].sort_values("dt")
        if len(train_pool) > max_train_rows:
            train_pool = train_pool.tail(max_train_rows)
        if len(train_pool) == 0:
            block_start = block_end
            continue

        model = fit_recovery_model(train_pool)
        predict_mask = (df["dt"] >= block_start) & (df["dt"] < block_end)
        df.loc[predict_mask, "imputed_demand"] = np.clip(
            model.predict(df.loc[predict_mask, recovery_features]),
            0,
            None,
        ).astype("float32")
        last_model = model
        trained_blocks.append(
            {
                "Block start": block_start,
                "Block end": block_end,
                "Training end": train_pool["dt"].max(),
                "Training rows": len(train_pool),
                "Predicted rows": int(predict_mask.sum()),
            }
        )
        block_start = block_end

    final_train_mask = (df["dt"] < val_start) & (df["stockout_flag"] == 0)
    final_train_pool = df.loc[final_train_mask].sort_values("dt")
    if len(final_train_pool) > max_train_rows:
        final_train_pool = final_train_pool.tail(max_train_rows)
    final_model = fit_recovery_model(final_train_pool)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "latent_demand_recovery_lightgbm.pkl")
    _write_model_feature_importance(
        final_model,
        recovery_features,
        "owner_latent_recovery_feature_importance",
        "Top 20 Feature Importance - Latent Demand Recovery",
    )

    forward_mask = df["dt"] >= val_start
    df.loc[forward_mask, "imputed_demand"] = np.clip(
        final_model.predict(df.loc[forward_mask, recovery_features]),
        0,
        None,
    ).astype("float32")

    # Use only out-of-fold train-period predictions for calibration. This keeps
    # the recovered training labels independent of validation/test information.
    calibration_mask = (
        (df["dt"] >= first_block_start)
        & (df["dt"] < val_start)
        & (df["stockout_flag"] == 0)
    )
    if int(calibration_mask.sum()) == 0:
        calibration_mask = (
            (df["dt"] < val_start)
            & (df["stockout_flag"] == 0)
        )
    calibration_actual = float(df.loc[calibration_mask, "sale_amount"].sum())
    calibration_pred = float(df.loc[calibration_mask, "imputed_demand"].sum())
    calibration_factor = calibration_pred / max(calibration_actual, 1e-9)
    if not np.isfinite(calibration_factor) or calibration_factor <= 0:
        calibration_factor = 1.0
    calibration_factor = max(1.0, calibration_factor)
    df["imputed_demand_raw"] = df["imputed_demand"].astype("float32")
    df["imputed_demand"] = (df["imputed_demand_raw"] / calibration_factor).clip(lower=0).astype("float32")

    raw_lost = np.where(
        df["stockout_flag"] == 1,
        np.maximum(df["imputed_demand"] - df["sale_amount"], 0),
        0,
    ).astype("float32")
    cap_source_mask = (df["dt"] < val_start).to_numpy() & (raw_lost > 0)
    positive_lost = raw_lost[cap_source_mask]
    lost_demand_cap = float(np.quantile(positive_lost, 0.90)) if len(positive_lost) else 0.0
    df["lost_demand_raw"] = raw_lost.astype("float32")
    df["lost_demand_recovered"] = np.minimum(raw_lost, lost_demand_cap).astype("float32")
    df["recovered_demand"] = np.where(
        df["stockout_flag"] == 1,
        df["sale_amount"] + df["lost_demand_recovered"],
        df["sale_amount"],
    ).astype("float32")

    output = PROCESSED_DATA_DIR / "fresh50k_recovered_hourly.parquet"
    df[
        [
            "dt",
            "series_id",
            "sale_amount",
            "stockout_flag",
            "imputed_demand_raw",
            "imputed_demand",
            "lost_demand_raw",
            "recovered_demand",
            "lost_demand_recovered",
        ]
    ].to_parquet(
        output,
        index=False,
    )

    summary = pd.DataFrame(
        [
            ("Observed sales", float(df["sale_amount"].sum())),
            ("Recovered latent demand", float(df["recovered_demand"].sum())),
            ("Recovered lost demand", float(df["lost_demand_recovered"].sum())),
            ("Recovered lift over observed", float(df["lost_demand_recovered"].sum() / max(df["sale_amount"].sum(), 1e-9))),
            ("Stockout row rate", float(df["stockout_flag"].mean())),
            ("Recovery method", "expanding-window weekly recovery"),
            ("Imputation calibration factor", float(calibration_factor)),
            ("Lost demand cap quantile", "q90 of calibrated positive lost demand"),
            ("Lost demand cap source", "train-period calibrated positive lost demand only"),
            ("Lost demand cap value", float(lost_demand_cap)),
            ("Warmup days kept observed", int(warmup_days)),
            ("Recovery block days", int(block_days)),
            ("Recovery blocks", int(len(trained_blocks))),
            ("Final recovery train rows", int(len(final_train_pool))),
            ("Recovery train source", "past non-stockout rows only for train blocks; train-period non-stockout rows for validation/test"),
            ("Calibration source", "train-period out-of-fold non-stockout rows only"),
            ("Final recovery train end", str(final_train_pool["dt"].max())),
            ("Validation start", str(val_start)),
            ("Test start", str(test_start)),
        ],
        columns=["Metric", "Value"],
    )
    summary.to_csv(TABLES_DIR / "owner_latent_recovery_summary.csv", index=False)
    pd.DataFrame(trained_blocks).to_csv(TABLES_DIR / "owner_recovery_blocks.csv", index=False)

    aggregate = df.groupby("dt", as_index=False).agg(
        observed_sales=("sale_amount", "sum"),
        recovered_demand=("recovered_demand", "sum"),
    )
    plt.figure(figsize=(12, 5))
    plt.plot(aggregate["dt"], aggregate["observed_sales"], label="Observed sales", linewidth=1.2)
    plt.plot(aggregate["dt"], aggregate["recovered_demand"], label="Recovered latent demand", linewidth=1.2)
    plt.title("Observed Sales vs Recovered Latent Demand")
    plt.xlabel("Time")
    plt.ylabel("Aggregate demand")
    plt.legend()
    plt.grid(alpha=0.25)
    _save(FIGURES_DIR / "owner_observed_vs_recovered_demand.png")
    return df


def _future_sum(group: pd.Series, horizon_days: int = 7) -> pd.Series:
    result = None
    for step in range(1, horizon_days + 1):
        shifted = group.shift(-step)
        result = shifted if result is None else result + shifted
    return result


def _owner_prediction_frame(
    model,
    frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    target_col: str,
) -> pd.DataFrame:
    pred = frame[
        [
            "date",
            "series_id",
            "target_next7_recovered_daily",
            "target_next7_observed_daily",
            "target_next7_stockout_rate",
        ]
    ].copy()
    pred["y_pred"] = np.clip(model.predict(frame[features]), 0, None).astype("float32")
    pred["Model"] = model_name
    pred["Target used for training"] = target_col
    pred["y_true_recovered"] = pred["target_next7_recovered_daily"]
    pred["y_true_observed"] = pred["target_next7_observed_daily"]
    pred["residual_recovered"] = pred["y_true_recovered"] - pred["y_pred"]
    return pred


def _baseline_prediction_frame(
    frame: pd.DataFrame,
    model_name: str,
    y_pred: pd.Series,
) -> pd.DataFrame:
    pred = frame[
        [
            "date",
            "series_id",
            "target_next7_recovered_daily",
            "target_next7_observed_daily",
            "target_next7_stockout_rate",
        ]
    ].copy()
    pred["y_pred"] = np.clip(y_pred.to_numpy(dtype="float32"), 0, None)
    pred["Model"] = model_name
    pred["Target used for training"] = "baseline_no_training"
    pred["y_true_recovered"] = pred["target_next7_recovered_daily"]
    pred["y_true_observed"] = pred["target_next7_observed_daily"]
    pred["residual_recovered"] = pred["y_true_recovered"] - pred["y_pred"]
    return pred


def _write_daily_split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    horizon_days: int = 7,
    purge_days: int = 7,
) -> None:
    rows = []
    for split_name, split_df in [("Train", train), ("Validation", val), ("Test", test)]:
        if split_df.empty:
            continue
        target_start = split_df["date"].min() + pd.Timedelta(days=1)
        target_end = split_df["date"].max() + pd.Timedelta(days=7)
        rows.append(
            {
                "Split": split_name,
                "Origin start": split_df["date"].min(),
                "Origin end": split_df["date"].max(),
                "Target window start": target_start,
                "Target window end": target_end,
                "Forecast horizon days": horizon_days,
                "Purge gap days": purge_days,
                "Rows": len(split_df),
                "Series": split_df["series_id"].nunique(),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES_DIR / "owner_daily_split_summary.csv", index=False)


def _write_owner_feature_importance(model, features: list[str], model_name: str) -> None:
    safe_name = model_name.lower().replace(" ", "_").replace("-", "")
    _write_model_feature_importance(
        model,
        features,
        f"owner_{safe_name}_feature_importance",
        f"Top 20 Feature Importance - {model_name}",
    )


def _write_owner_two_stage_diagnostics(val_predictions: pd.DataFrame, test_predictions: pd.DataFrame) -> None:
    rows = []
    nonoverlap_rows = []
    interval_frames = []
    for model_name, test_pred in test_predictions.groupby("Model", sort=False):
        val_pred = val_predictions[val_predictions["Model"] == model_name]
        residual = test_pred["residual_recovered"].dropna()
        aggregate_residual = test_pred.groupby("date", sort=True)["residual_recovered"].mean()
        lag = min(7, max(len(aggregate_residual) - 1, 1))
        lb = acorr_ljungbox(aggregate_residual, lags=[lag], return_df=True)
        jb = jarque_bera(residual.sample(min(len(residual), 500_000), random_state=42))
        metrics = evaluate_predictions(test_pred["y_true_recovered"], test_pred["y_pred"])
        rows.append(
            {
                "Model": model_name,
                "Evaluation target": "Recovered latent demand proxy",
                **metrics,
                "WPE": _wpe(test_pred["y_true_recovered"], test_pred["y_pred"]),
                "Residual mean": float(residual.mean()),
                "Residual std": float(residual.std()),
                "Ljung-Box lag": int(lag),
                "Ljung-Box statistic": float(lb["lb_stat"].iloc[0]),
                "Ljung-Box p-value": float(lb["lb_pvalue"].iloc[0]),
                "Jarque-Bera statistic": float(jb.statistic),
                "Jarque-Bera p-value": float(jb.pvalue),
                "N": int(len(test_pred)),
            }
        )

        nonoverlap_dates = sorted(test_pred["date"].unique())[::7]
        nonoverlap_pred = test_pred[test_pred["date"].isin(nonoverlap_dates)].copy()
        nonoverlap_residual = nonoverlap_pred["residual_recovered"].dropna()
        nonoverlap_aggregate = nonoverlap_pred.groupby("date", sort=True)["residual_recovered"].mean()
        nonoverlap_lag = min(1, max(len(nonoverlap_aggregate) - 1, 1))
        nonoverlap_lb = acorr_ljungbox(nonoverlap_aggregate, lags=[nonoverlap_lag], return_df=True)
        nonoverlap_jb = jarque_bera(
            nonoverlap_residual.sample(min(len(nonoverlap_residual), 500_000), random_state=42)
        )
        nonoverlap_metrics = evaluate_predictions(nonoverlap_pred["y_true_recovered"], nonoverlap_pred["y_pred"])
        nonoverlap_rows.append(
            {
                "Model": model_name,
                "Evaluation mode": "Non-overlapping 7-day targets",
                "Evaluation dates": "; ".join(pd.to_datetime(nonoverlap_dates).strftime("%Y-%m-%d")),
                **nonoverlap_metrics,
                "WPE": _wpe(nonoverlap_pred["y_true_recovered"], nonoverlap_pred["y_pred"]),
                "Residual mean": float(nonoverlap_residual.mean()),
                "Residual std": float(nonoverlap_residual.std()),
                "Ljung-Box lag": int(nonoverlap_lag),
                "Ljung-Box statistic": float(nonoverlap_lb["lb_stat"].iloc[0]),
                "Ljung-Box p-value": float(nonoverlap_lb["lb_pvalue"].iloc[0]),
                "Jarque-Bera statistic": float(nonoverlap_jb.statistic),
                "Jarque-Bera p-value": float(nonoverlap_jb.pvalue),
                "N": int(len(nonoverlap_pred)),
                "Aggregate residual points": int(len(nonoverlap_aggregate)),
            }
        )

        abs_val_residual = val_pred["residual_recovered"].abs().dropna()
        q80 = float(abs_val_residual.quantile(0.80))
        q95 = float(abs_val_residual.quantile(0.95))
        intervals = test_pred.copy()
        intervals["lower_80"] = (intervals["y_pred"] - q80).clip(lower=0)
        intervals["upper_80"] = intervals["y_pred"] + q80
        intervals["lower_95"] = (intervals["y_pred"] - q95).clip(lower=0)
        intervals["upper_95"] = intervals["y_pred"] + q95
        interval_frames.append(intervals)

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(TABLES_DIR / "owner_two_stage_diagnostics.csv", index=False)
    pd.DataFrame(nonoverlap_rows).to_csv(TABLES_DIR / "owner_two_stage_nonoverlap_diagnostics.csv", index=False)
    intervals = pd.concat(interval_frames, ignore_index=True)
    intervals.to_csv(TABLES_DIR / "owner_two_stage_prediction_intervals.csv", index=False)
    coverage_rows = []
    for model_name, model_intervals in intervals.groupby("Model", sort=False):
        coverage_rows.append(
            {
                "Model": model_name,
                "80% interval coverage": float(
                    (
                        (model_intervals["y_true_recovered"] >= model_intervals["lower_80"])
                        & (model_intervals["y_true_recovered"] <= model_intervals["upper_80"])
                    ).mean()
                ),
                "95% interval coverage": float(
                    (
                        (model_intervals["y_true_recovered"] >= model_intervals["lower_95"])
                        & (model_intervals["y_true_recovered"] <= model_intervals["upper_95"])
                    ).mean()
                ),
                "Mean 80% interval width": float(
                    (model_intervals["upper_80"] - model_intervals["lower_80"]).mean()
                ),
                "Mean 95% interval width": float(
                    (model_intervals["upper_95"] - model_intervals["lower_95"]).mean()
                ),
                "N": int(len(model_intervals)),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(TABLES_DIR / "owner_two_stage_interval_coverage_summary.csv", index=False)

    best_model_name = diagnostics.sort_values("WAPE").iloc[0]["Model"]
    recovered = test_predictions[test_predictions["Model"] == best_model_name].copy()
    if recovered.empty:
        return
    residual = recovered["residual_recovered"].dropna()
    sample = residual.sample(min(len(residual), 500_000), random_state=42)
    plt.figure(figsize=(9, 5))
    plt.hist(sample, bins=80, color="#059669", alpha=0.85)
    plt.title(f"Two-Stage Residual Distribution - {best_model_name}")
    plt.xlabel("Residual against recovered latent demand")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.25)
    _save(FIGURES_DIR / "owner_two_stage_residual_distribution.png")

    aggregate_residual = recovered.groupby("date", sort=True)["residual_recovered"].mean()
    plt.figure(figsize=(9, 5))
    plot_acf(aggregate_residual, lags=min(10, len(aggregate_residual) - 1), zero=False, ax=plt.gca())
    plt.title(f"Two-Stage Residual ACF - {best_model_name}")
    _save(FIGURES_DIR / "owner_two_stage_residual_acf.png")

    plt.figure(figsize=(9, 5))
    ordered = diagnostics.sort_values("WAPE", ascending=False)
    colors = ["#78716c" if "forecasting" not in model.lower() else "#059669" for model in ordered["Model"]]
    plt.barh(ordered["Model"], ordered["WAPE"], color=colors)
    plt.title("Two-Stage Diagnostics: WAPE on Recovered Latent Demand")
    plt.xlabel("WAPE")
    plt.ylabel("Model")
    plt.grid(axis="x", alpha=0.25)
    _save(FIGURES_DIR / "owner_two_stage_diagnostics_wape.png")

    recovered_intervals = intervals[intervals["Model"] == best_model_name]
    aggregate = recovered_intervals.groupby("date", as_index=False).agg(
        y_true_recovered=("y_true_recovered", "sum"),
        y_pred=("y_pred", "sum"),
        lower_80=("lower_80", "sum"),
        upper_80=("upper_80", "sum"),
        lower_95=("lower_95", "sum"),
        upper_95=("upper_95", "sum"),
    )
    plt.figure(figsize=(12, 5))
    x = aggregate["date"]
    plt.plot(x, aggregate["y_true_recovered"], label="Recovered latent demand proxy", linewidth=1.8, color="black")
    plt.plot(x, aggregate["y_pred"], label=best_model_name, linewidth=1.3, color="#059669")
    plt.fill_between(x, aggregate["lower_80"], aggregate["upper_80"], alpha=0.22, color="#10b981", label="80% interval")
    plt.fill_between(x, aggregate["lower_95"], aggregate["upper_95"], alpha=0.12, color="#10b981", label="95% interval")
    plt.title("Two-Stage Forecast Interval on Recovered Latent Demand")
    plt.xlabel("Date")
    plt.ylabel("Aggregate next 7-day demand")
    plt.legend()
    plt.grid(alpha=0.25)
    _save(FIGURES_DIR / "owner_two_stage_forecast_interval.png")


def build_daily_forecasting_table(recovered_hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate recovered hourly demand to daily series and create 7-day forecast features."""
    if recovered_hourly is None:
        recovered_hourly = pd.read_parquet(PROCESSED_DATA_DIR / "fresh50k_recovered_hourly.parquet")
    recovered_hourly = recovered_hourly.copy()
    recovered_hourly["date"] = pd.to_datetime(recovered_hourly["dt"]).dt.floor("D")

    daily = (
        recovered_hourly.groupby(["series_id", "date"], as_index=False)
        .agg(
            observed_daily=("sale_amount", "sum"),
            recovered_daily=("recovered_demand", "sum"),
            stockout_rate=("stockout_flag", "mean"),
            lost_demand_recovered=("lost_demand_recovered", "sum"),
        )
        .sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )
    daily["day_of_week"] = daily["date"].dt.dayofweek.astype("int16")
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype("int8")
    daily["month"] = daily["date"].dt.month.astype("int16")

    for demand_col in ["observed_daily", "recovered_daily"]:
        grouped = daily.groupby("series_id", sort=False)[demand_col]
        for lag in [1, 2, 7, 14, 28]:
            daily[f"{demand_col}_lag_{lag}"] = grouped.shift(lag).astype("float32")
        shifted = grouped.shift(1)
        for window in [7, 14, 28]:
            daily[f"{demand_col}_roll_mean_{window}"] = (
                shifted.groupby(daily["series_id"], sort=False)
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
                .astype("float32")
            )
            daily[f"{demand_col}_roll_sum_{window}"] = (
                shifted.groupby(daily["series_id"], sort=False)
                .rolling(window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
                .astype("float32")
            )
        daily[f"target_next7_{demand_col}"] = (
            grouped.transform(lambda s: _future_sum(s, 7)).astype("float32")
        )

    daily["target_next7_stockout_rate"] = daily.groupby("series_id", sort=False)["stockout_rate"].transform(
        lambda s: _future_sum(s, 7) / 7
    )
    daily = daily.dropna(subset=["target_next7_observed_daily", "target_next7_recovered_daily"]).reset_index(drop=True)
    daily.to_parquet(PROCESSED_DATA_DIR / "fresh50k_owner_daily_recovered_forecasting.parquet", index=False)
    return daily


def _daily_split(df: pd.DataFrame, test_days: int = 14, val_days: int = 7, horizon_days: int = 7):
    """Create purged splits for next-horizon forecasting.

    The target at date t is the sum from t+1 to t+horizon_days. A standard
    split by origin date would let validation target windows overlap the test
    period. The purge gap keeps the target windows of train, validation and
    test separated.
    """
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days) + pd.Timedelta(days=1)
    val_end = test_start - pd.Timedelta(days=horizon_days + 1)
    val_start = val_end - pd.Timedelta(days=val_days) + pd.Timedelta(days=1)
    train_end = val_start - pd.Timedelta(days=horizon_days + 1)
    train = df[df["date"] <= train_end]
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
    test = df[df["date"] >= test_start]
    return train, val, test


def run_owner_daily_forecasting(daily: pd.DataFrame | None = None, max_train_rows: int = 300_000) -> pd.DataFrame:
    """Compare 7-day demand forecasting on observed vs recovered demand."""
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    if daily is None:
        daily = build_daily_forecasting_table()
    horizon_days = 7
    train, val, test = _daily_split(daily, horizon_days=horizon_days)
    _write_daily_split_summary(train, val, test, horizon_days=horizon_days, purge_days=horizon_days)

    base_features = ["day_of_week", "is_weekend", "month", "stockout_rate"]
    observed_features = base_features + [col for col in daily.columns if col.startswith("observed_daily_lag_") or col.startswith("observed_daily_roll_")]
    recovered_features = base_features + [col for col in daily.columns if col.startswith("recovered_daily_lag_") or col.startswith("recovered_daily_roll_")]

    configs = [
        ("Observed-sales forecasting", observed_features, "target_next7_observed_daily"),
        ("Recovered-demand forecasting", recovered_features, "target_next7_recovered_daily"),
    ]
    rows = []
    prediction_frames = []
    val_prediction_frames = []

    baseline_specs = [
        ("Recovered naive x7", lambda frame: frame["recovered_daily_lag_1"] * 7),
        ("Recovered seasonal naive 7-day", lambda frame: frame["recovered_daily_roll_mean_7"] * 7),
        ("Recovered rolling mean 14-day", lambda frame: frame["recovered_daily_roll_mean_14"] * 7),
    ]
    for model_name, predict_fn in baseline_specs:
        val_pred = _baseline_prediction_frame(val, model_name, predict_fn(val))
        pred = _baseline_prediction_frame(test, model_name, predict_fn(test))
        val_prediction_frames.append(val_pred)
        prediction_frames.append(pred)
        metrics_recovered = evaluate_predictions(pred["y_true_recovered"], pred["y_pred"])
        rows.append(
            {
                "Model": model_name,
                "Training target": "baseline_no_training",
                "Evaluation target": "Recovered latent demand proxy",
                **metrics_recovered,
                "WPE": _wpe(pred["y_true_recovered"], pred["y_pred"]),
                "N": len(pred),
                "Train rows used": 0,
            }
        )

    for model_name, features, target_col in configs:
        train_sample = train.sort_values("date").tail(max_train_rows)
        model = LGBMRegressor(
            objective="regression",
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=48,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            train_sample[features],
            train_sample[target_col],
            eval_set=[(val[features], val[target_col])],
            eval_metric="rmse",
            callbacks=[early_stopping(stopping_rounds=40), log_evaluation(period=0)],
        )
        safe_name = model_name.lower().replace(" ", "_").replace("-", "")
        joblib.dump(model, MODELS_DIR / f"owner_{safe_name}.pkl")
        _write_owner_feature_importance(model, features, model_name)

        val_pred = _owner_prediction_frame(model, val, features, model_name, target_col)
        pred = _owner_prediction_frame(model, test, features, model_name, target_col)
        val_prediction_frames.append(val_pred)
        prediction_frames.append(pred)

        metrics_recovered = evaluate_predictions(pred["y_true_recovered"], pred["y_pred"])
        metrics_observed = evaluate_predictions(pred["y_true_observed"], pred["y_pred"])
        rows.append(
            {
                "Model": model_name,
                "Training target": target_col,
                "Evaluation target": "Recovered latent demand proxy",
                **metrics_recovered,
                "WPE": _wpe(pred["y_true_recovered"], pred["y_pred"]),
                "N": len(pred),
                "Train rows used": len(train_sample),
            }
        )
        rows.append(
            {
                "Model": model_name,
                "Training target": target_col,
                "Evaluation target": "Observed sales",
                **metrics_observed,
                "WPE": _wpe(pred["y_true_observed"], pred["y_pred"]),
                "N": len(pred),
                "Train rows used": len(train_sample),
            }
        )

    val_predictions_so_far = pd.concat(val_prediction_frames, ignore_index=True)
    predictions_so_far = pd.concat(prediction_frames, ignore_index=True)
    seasonal_val = val_predictions_so_far[val_predictions_so_far["Model"] == "Recovered seasonal naive 7-day"].sort_values(
        ["date", "series_id"]
    )
    recovered_ml_val = val_predictions_so_far[
        val_predictions_so_far["Model"] == "Recovered-demand forecasting"
    ].sort_values(["date", "series_id"])
    seasonal_test = predictions_so_far[
        predictions_so_far["Model"] == "Recovered seasonal naive 7-day"
    ].sort_values(["date", "series_id"])
    recovered_ml_test = predictions_so_far[
        predictions_so_far["Model"] == "Recovered-demand forecasting"
    ].sort_values(["date", "series_id"])
    if len(seasonal_val) == len(recovered_ml_val) and len(seasonal_test) == len(recovered_ml_test):
        blend_rows = []
        for alpha in np.linspace(0, 1, 21):
            blended_val_pred = (1 - alpha) * seasonal_val["y_pred"].to_numpy() + alpha * recovered_ml_val[
                "y_pred"
            ].to_numpy()
            blend_rows.append(
                {
                    "LightGBM weight": float(alpha),
                    "Seasonal naive weight": float(1 - alpha),
                    "Validation WAPE": evaluate_predictions(
                        recovered_ml_val["y_true_recovered"], blended_val_pred
                    )["WAPE"],
                }
            )
        blend_summary = pd.DataFrame(blend_rows).sort_values("Validation WAPE").reset_index(drop=True)
        blend_summary.to_csv(TABLES_DIR / "owner_hybrid_blend_selection.csv", index=False)
        best_alpha = float(blend_summary.loc[0, "LightGBM weight"])
        hybrid_val = recovered_ml_val.copy()
        hybrid_val["Model"] = "Recovered seasonal-ML hybrid"
        hybrid_val["Target used for training"] = f"validation_blend_alpha_{best_alpha:.2f}"
        hybrid_val["y_pred"] = (
            (1 - best_alpha) * seasonal_val["y_pred"].to_numpy()
            + best_alpha * recovered_ml_val["y_pred"].to_numpy()
        ).astype("float32")
        hybrid_val["residual_recovered"] = hybrid_val["y_true_recovered"] - hybrid_val["y_pred"]
        hybrid_test = recovered_ml_test.copy()
        hybrid_test["Model"] = "Recovered seasonal-ML hybrid"
        hybrid_test["Target used for training"] = f"validation_blend_alpha_{best_alpha:.2f}"
        hybrid_test["y_pred"] = (
            (1 - best_alpha) * seasonal_test["y_pred"].to_numpy()
            + best_alpha * recovered_ml_test["y_pred"].to_numpy()
        ).astype("float32")
        hybrid_test["residual_recovered"] = hybrid_test["y_true_recovered"] - hybrid_test["y_pred"]
        val_prediction_frames.append(hybrid_val)
        prediction_frames.append(hybrid_test)
        hybrid_metrics = evaluate_predictions(hybrid_test["y_true_recovered"], hybrid_test["y_pred"])
        rows.append(
            {
                "Model": "Recovered seasonal-ML hybrid",
                "Training target": f"seasonal_naive_plus_recovered_lightgbm_alpha_{best_alpha:.2f}",
                "Evaluation target": "Recovered latent demand proxy",
                **hybrid_metrics,
                "WPE": _wpe(hybrid_test["y_true_recovered"], hybrid_test["y_pred"]),
                "N": len(hybrid_test),
                "Train rows used": max_train_rows,
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    val_predictions = pd.concat(val_prediction_frames, ignore_index=True)
    predictions.to_csv(TABLES_DIR / "owner_two_stage_daily_predictions.csv", index=False)
    val_predictions.to_csv(TABLES_DIR / "owner_two_stage_val_predictions.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "owner_two_stage_forecasting_comparison.csv", index=False)
    _write_owner_two_stage_diagnostics(val_predictions, predictions)

    plot_df = result[result["Evaluation target"] == "Recovered latent demand proxy"]
    plt.figure(figsize=(9, 5))
    colors = ["#78716c" if "forecasting" not in model.lower() else "#059669" for model in plot_df["Model"]]
    plt.bar(plot_df["Model"], plot_df["WPE"], color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Bias vs Recovered Latent Demand Proxy")
    plt.xlabel("Model")
    plt.ylabel("WPE")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    _save(FIGURES_DIR / "owner_two_stage_bias_comparison.png")

    aggregate_pred = predictions.groupby(["date", "Model"], as_index=False).agg(
        y_true_recovered=("y_true_recovered", "sum"),
        y_pred=("y_pred", "sum"),
    )
    plt.figure(figsize=(12, 5))
    for model_name, group in aggregate_pred.groupby("Model"):
        plt.plot(group["date"], group["y_pred"], label=model_name, linewidth=1.2)
    actual = aggregate_pred.groupby("date", as_index=False)["y_true_recovered"].first()
    plt.plot(actual["date"], actual["y_true_recovered"], label="Recovered latent demand proxy", linewidth=1.8, color="black")
    plt.title("Owner-Style 7-Day Demand Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Next 7-day total demand")
    plt.legend()
    plt.grid(alpha=0.25)
    _save(FIGURES_DIR / "owner_two_stage_forecast_comparison.png")
    return result


def run_owner_ablation(daily: pd.DataFrame | None = None, max_train_rows: int = 300_000) -> pd.DataFrame:
    """Ablation study for the seasonal-ML hybrid.

    Each variant removes one component of the current model, retrains the
    LightGBM stage with identical hyperparameters, re-selects the hybrid blend
    weight on validation, and measures the change in test WAPE against the
    recovered latent demand proxy.
    """
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    if daily is None:
        daily = build_daily_forecasting_table()
    train, val, test = _daily_split(daily)
    train_sample = train.sort_values("date").tail(max_train_rows)

    lag_cols = [col for col in daily.columns if col.startswith("recovered_daily_lag_")]
    roll_cols = [col for col in daily.columns if col.startswith("recovered_daily_roll_")]
    observed_cols = [
        col
        for col in daily.columns
        if col.startswith("observed_daily_lag_") or col.startswith("observed_daily_roll_")
    ]
    calendar_cols = ["day_of_week", "is_weekend", "month"]
    full_features = calendar_cols + ["stockout_rate"] + lag_cols + roll_cols
    recovered_target = "target_next7_recovered_daily"

    variants = [
        ("Full model", full_features, recovered_target),
        (
            "No demand recovery (observed features + target)",
            calendar_cols + ["stockout_rate"] + observed_cols,
            "target_next7_observed_daily",
        ),
        ("No lag features", [c for c in full_features if c not in lag_cols], recovered_target),
        ("No rolling features", [c for c in full_features if c not in roll_cols], recovered_target),
        ("No calendar features", [c for c in full_features if c not in calendar_cols], recovered_target),
        ("No stockout rate", [c for c in full_features if c != "stockout_rate"], recovered_target),
    ]

    y_val = val[recovered_target].to_numpy(dtype="float64")
    y_test = test[recovered_target].to_numpy(dtype="float64")
    seasonal_val = np.clip((val["recovered_daily_roll_mean_7"] * 7).to_numpy(dtype="float64"), 0, None)
    seasonal_test = np.clip((test["recovered_daily_roll_mean_7"] * 7).to_numpy(dtype="float64"), 0, None)

    rows = []
    for variant_name, features, target_col in variants:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=48,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            train_sample[features],
            train_sample[target_col],
            eval_set=[(val[features], val[target_col])],
            eval_metric="rmse",
            callbacks=[early_stopping(stopping_rounds=40), log_evaluation(period=0)],
        )
        val_pred = np.clip(model.predict(val[features]), 0, None)
        test_pred = np.clip(model.predict(test[features]), 0, None)

        blend_scores = [
            (
                evaluate_predictions(y_val, (1 - alpha) * seasonal_val + alpha * val_pred)["WAPE"],
                float(alpha),
            )
            for alpha in np.linspace(0, 1, 21)
        ]
        _, best_alpha = min(blend_scores)
        hybrid_test_pred = (1 - best_alpha) * seasonal_test + best_alpha * test_pred
        rows.append(
            {
                "Ablation variant": variant_name,
                "Features": len(features),
                "Best iteration": int(model.best_iteration_ or 0),
                "LightGBM weight (val)": best_alpha,
                "Test WAPE (LightGBM only)": evaluate_predictions(y_test, test_pred)["WAPE"],
                "Test WAPE (hybrid)": evaluate_predictions(y_test, hybrid_test_pred)["WAPE"],
            }
        )

    ablation = pd.DataFrame(rows)
    full_wape = float(ablation.loc[ablation["Ablation variant"] == "Full model", "Test WAPE (hybrid)"].iloc[0])
    ablation["Delta hybrid WAPE vs full"] = ablation["Test WAPE (hybrid)"] - full_wape
    ablation.to_csv(TABLES_DIR / "owner_ablation_study.csv", index=False)

    ordered = ablation.sort_values("Test WAPE (hybrid)", ascending=True)
    plt.figure(figsize=(10, 5))
    colors = ["#059669" if name == "Full model" else "#78716c" for name in ordered["Ablation variant"]]
    plt.barh(ordered["Ablation variant"], ordered["Test WAPE (hybrid)"], color=colors)
    plt.axvline(full_wape, color="black", linewidth=1, linestyle="--", label="Full model")
    plt.title("Ablation Study: Hybrid Test WAPE When Removing One Component")
    plt.xlabel("Test WAPE (recovered latent demand proxy)")
    plt.legend()
    plt.grid(axis="x", alpha=0.25)
    _save(FIGURES_DIR / "owner_ablation_wape.png")
    return ablation


def run_owner_supplementary_analysis(daily: pd.DataFrame | None = None) -> None:
    """Segment-level errors, representative case studies, bootstrap significance, split timeline."""
    if daily is None:
        daily = pd.read_parquet(PROCESSED_DATA_DIR / "fresh50k_owner_daily_recovered_forecasting.parquet")
    predictions = pd.read_csv(TABLES_DIR / "owner_two_stage_daily_predictions.csv", parse_dates=["date"])
    models = [
        "Recovered seasonal naive 7-day",
        "Observed-sales forecasting",
        "Recovered-demand forecasting",
        "Recovered seasonal-ML hybrid",
    ]
    preds = predictions[predictions["Model"].isin(models)].copy()

    # --- 1. segment-level WAPE (stockout and volume quartiles by series) ---
    series_stats = daily.groupby("series_id").agg(
        mean_stockout=("stockout_rate", "mean"),
        mean_demand=("recovered_daily", "mean"),
    )
    quartile_labels = ["Q1 (thap nhat)", "Q2", "Q3", "Q4 (cao nhat)"]
    series_stats["Stockout quartile"] = pd.qcut(series_stats["mean_stockout"], 4, labels=quartile_labels)
    series_stats["Volume quartile"] = pd.qcut(series_stats["mean_demand"], 4, labels=quartile_labels)
    merged = preds.merge(series_stats, on="series_id", how="left")
    rows = []
    for dim in ["Stockout quartile", "Volume quartile"]:
        for segment, seg_df in merged.groupby(dim, observed=True):
            row = {"Dimension": dim, "Segment": str(segment), "Series": seg_df["series_id"].nunique()}
            for model in models:
                m_df = seg_df[seg_df["Model"] == model]
                row[model] = evaluate_predictions(m_df["y_true_recovered"], m_df["y_pred"])["WAPE"]
            rows.append(row)
    segment_wape = pd.DataFrame(rows)
    segment_wape.to_csv(TABLES_DIR / "owner_segment_wape.csv", index=False)

    plot_df = segment_wape[segment_wape["Dimension"] == "Stockout quartile"]
    x = np.arange(len(plot_df))
    width = 0.2
    plt.figure(figsize=(10, 5))
    colors = {"Recovered seasonal naive 7-day": "#78716c", "Observed-sales forecasting": "#c2410c",
              "Recovered-demand forecasting": "#2563eb", "Recovered seasonal-ML hybrid": "#059669"}
    for k, model in enumerate(models):
        plt.bar(x + (k - 1.5) * width, plot_df[model], width, label=model, color=colors[model])
    plt.xticks(x, plot_df["Segment"])
    plt.title("Test WAPE by Series Stockout Quartile")
    plt.xlabel("Stockout quartile (series-level mean stockout rate)")
    plt.ylabel("WAPE")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    _save(FIGURES_DIR / "owner_segment_wape.png")

    # --- 2. case study on representative series ---
    rep_path = TABLES_DIR / "representative_series.csv"
    if rep_path.exists():
        reps = pd.read_csv(rep_path)
        fig, axes = plt.subplots(len(reps), 1, figsize=(11, 3.2 * len(reps)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (_, rep) in zip(axes, reps.iterrows()):
            sid = rep["series_id"]
            for model, color, lw in [
                ("Recovered seasonal naive 7-day", "#78716c", 1.1),
                ("Recovered seasonal-ML hybrid", "#059669", 1.4),
            ]:
                s = preds[(preds["series_id"] == sid) & (preds["Model"] == model)].sort_values("date")
                ax.plot(s["date"], s["y_pred"], label=model, color=color, linewidth=lw)
            actual = preds[(preds["series_id"] == sid) & (preds["Model"] == models[0])].sort_values("date")
            ax.plot(actual["date"], actual["y_true_recovered"], label="Recovered latent demand proxy",
                    color="black", linewidth=1.6)
            ax.set_title(f"{sid} ({rep['type']}, stockout rate {rep['stockout_rate']:.0%})", fontsize=10)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("Forecast origin date")
        fig.suptitle("Next-7-Day Forecasts on Representative Series (Test Set)", y=0.995)
        fig.supylabel("Next 7-day total demand")
        _save(FIGURES_DIR / "owner_case_study_forecasts.png")

    # --- 3. series-level cluster bootstrap for WAPE differences ---
    per_series = (
        preds.assign(abs_err=lambda f: (f["y_true_recovered"] - f["y_pred"]).abs(),
                     abs_y=lambda f: f["y_true_recovered"].abs())
        .groupby(["Model", "series_id"])[["abs_err", "abs_y"]]
        .sum()
    )
    series_ids = per_series.loc[models[0]].index.to_numpy()
    err = {m: per_series.loc[m, "abs_err"].reindex(series_ids).to_numpy() for m in models}
    tot = {m: per_series.loc[m, "abs_y"].reindex(series_ids).to_numpy() for m in models}
    rng = np.random.default_rng(42)
    n_boot = 2000
    idx = rng.integers(0, len(series_ids), size=(n_boot, len(series_ids)))
    comparisons = [
        ("Recovered seasonal naive 7-day", "Recovered seasonal-ML hybrid"),
        ("Observed-sales forecasting", "Recovered seasonal-ML hybrid"),
        ("Recovered-demand forecasting", "Recovered seasonal-ML hybrid"),
    ]
    boot_rows = []
    for base, challenger in comparisons:
        diffs = (err[base][idx].sum(axis=1) / tot[base][idx].sum(axis=1)
                 - err[challenger][idx].sum(axis=1) / tot[challenger][idx].sum(axis=1))
        point = err[base].sum() / tot[base].sum() - err[challenger].sum() / tot[challenger].sum()
        boot_rows.append(
            {
                "Baseline": base,
                "Challenger": challenger,
                "WAPE difference (baseline - challenger)": point,
                "Bootstrap CI 2.5%": float(np.quantile(diffs, 0.025)),
                "Bootstrap CI 97.5%": float(np.quantile(diffs, 0.975)),
                "Share of resamples <= 0": float((diffs <= 0).mean()),
                "Bootstrap resamples": n_boot,
                "Resample unit": "series_id",
            }
        )
    pd.DataFrame(boot_rows).to_csv(TABLES_DIR / "owner_model_significance_bootstrap.csv", index=False)

    # --- 4. purged split timeline figure ---
    split = pd.read_csv(TABLES_DIR / "owner_daily_split_summary.csv", parse_dates=[
        "Origin start", "Origin end", "Target window start", "Target window end"])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    colors_split = {"Train": "#2563eb", "Validation": "#c2410c", "Test": "#059669"}
    for i, (_, r) in enumerate(split.iterrows()):
        y = len(split) - i
        ax.barh(y + 0.18, (r["Origin end"] - r["Origin start"]).days + 1, left=r["Origin start"],
                height=0.32, color=colors_split[r["Split"]], label=f"{r['Split']} origins")
        ax.barh(y - 0.18, (r["Target window end"] - r["Target window start"]).days + 1,
                left=r["Target window start"], height=0.32, color=colors_split[r["Split"]], alpha=0.35,
                label=f"{r['Split']} target windows")
    for gap_start, gap_end in [
        (split.iloc[0]["Target window end"], split.iloc[1]["Origin start"]),
        (split.iloc[1]["Target window end"], split.iloc[2]["Origin start"]),
    ]:
        ax.axvspan(gap_start, gap_end, color="#facc15", alpha=0.25)
    ax.set_yticks([3, 2, 1])
    ax.set_yticklabels(split["Split"])
    ax.set_title("Purged Time Split: Origins, Target Windows and Purge Gaps (7 days)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="x", alpha=0.25)
    _save(FIGURES_DIR / "owner_purged_split_timeline.png")


def run_owner_aligned_pipeline() -> None:
    """Run the paper/repo-aligned two-stage pipeline."""
    recovered = recover_hourly_demand()
    daily = build_daily_forecasting_table(recovered)
    run_owner_daily_forecasting(daily)
    run_owner_ablation(daily)
    run_owner_supplementary_analysis(daily)
