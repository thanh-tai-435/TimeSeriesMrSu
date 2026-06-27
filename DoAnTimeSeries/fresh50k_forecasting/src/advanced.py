"""Advanced analysis for stronger time-series project evidence."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, TABLES_DIR
from .evaluation import evaluate_predictions
from .models import load_feature_columns
from .split import load_feature_table, time_based_split


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def write_model_rationale(reports_dir: Path = REPORTS_DIR) -> Path:
    """Write a defensible model-selection rationale for slides/report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    text = """# Model Rationale

## Why Global Machine Learning?

FreshRetailNet-50K contains thousands of store-product time series. Training one separate ARIMA/SARIMA model for every series would be difficult to scale, unstable for sparse series, and hard to maintain. A global model learns shared demand patterns across products, stores, categories, calendar effects, promotions, weather, and stockout behavior.

## Why LightGBM?

LightGBM is a strong fit for this project because the final table is tabular and feature-rich: lag features, rolling features, calendar encodings, promotion/weather variables, and stockout indicators. Gradient-boosted trees handle non-linear interactions without requiring manual interaction terms, tolerate mixed feature scales, and train faster than many alternatives on millions of rows.

## Why Not Only ARIMA/SARIMA?

ARIMA/SARIMA is useful for aggregate or individual low-dimensional series, and it is included as an aggregate diagnostic benchmark. However, it does not naturally scale to 5,000 sampled series with many external covariates and stockout-aware features. It is also less suitable for sparse/intermittent product-level demand.

## Why Not Deep Learning?

Deep learning models such as DeepAR, TFT, or N-BEATS can be powerful, but they require more infrastructure, careful tuning, and longer training. For a reproducible course project on a 16GB RAM machine, LightGBM gives a better accuracy-to-complexity tradeoff.

## Why Keep Ridge?

Ridge regression is kept as a transparent linear baseline. Its coefficients give a simple way to discuss feature direction and relative linear effect, while LightGBM provides the main non-linear forecasting performance.

## Why Quantile Intervals?

Point forecasts alone are incomplete. Quantile LightGBM estimates lower and upper conditional quantiles, giving uncertainty intervals that are more defensible than assuming normal residuals.

## Key Limitation

Observed sales during stockout periods are censored: true demand may be higher than observed sales. Stockout-aware features help the model recognize these periods, but a full latent-demand recovery model would be a natural next extension.
"""
    output = reports_dir / "model_rationale.md"
    output.write_text(text, encoding="utf-8")
    return output


def run_quantile_lightgbm(
    features_path: Path = PROCESSED_DATA_DIR / "fresh50k_features.parquet",
    tables_dir: Path = TABLES_DIR,
    models_dir: Path = MODELS_DIR,
    max_train_rows: int = 500_000,
    horizon: int = 1,
) -> None:
    """Train quantile LightGBM models for forecast intervals."""
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    feature_columns = load_feature_columns(tables_dir / "feature_columns.csv")
    target_col = f"target_h{horizon}"
    needed = ["dt", "series_id", target_col, *feature_columns]
    df = load_feature_table(features_path)
    df = df[[column for column in needed if column in df.columns]]
    train_df, val_df, test_df = time_based_split(df, output_path=tables_dir / "split_summary.csv")
    train_sample = train_df.sort_values("dt").tail(max_train_rows)

    predictions = test_df[["dt", "series_id", target_col]].rename(columns={target_col: "y_true"}).copy()
    models_dir.mkdir(parents=True, exist_ok=True)
    for alpha, name in [(0.10, "q10"), (0.50, "q50"), (0.90, "q90")]:
        model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=48,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            train_sample[feature_columns],
            train_sample[target_col],
            eval_set=[(val_df[feature_columns], val_df[target_col])],
            eval_metric="quantile",
            callbacks=[early_stopping(stopping_rounds=40), log_evaluation(period=0)],
        )
        joblib.dump(model, models_dir / f"lightgbm_quantile_{name}_h{horizon}.pkl")
        predictions[name] = model.predict(test_df[feature_columns]).astype("float32")

    predictions["lower_80"] = np.minimum(predictions["q10"], predictions["q90"]).clip(lower=0)
    predictions["upper_80"] = np.maximum(predictions["q10"], predictions["q90"])
    predictions["covered_80"] = (
        (predictions["y_true"] >= predictions["lower_80"]) & (predictions["y_true"] <= predictions["upper_80"])
    )
    predictions.to_csv(tables_dir / f"prediction_intervals_quantile_lightgbm_h{horizon}.csv", index=False)

    metrics = evaluate_predictions(predictions["y_true"], predictions["q50"])
    interval_width = float((predictions["upper_80"] - predictions["lower_80"]).mean())
    coverage = float(predictions["covered_80"].mean())
    pd.DataFrame(
        [
            {
                "Horizon": horizon,
                "Model": "QuantileLightGBM",
                **metrics,
                "Interval": "q10-q90",
                "Coverage": coverage,
                "Mean interval width": interval_width,
                "N": len(predictions),
            }
        ]
    ).to_csv(tables_dir / f"quantile_interval_metrics_h{horizon}.csv", index=False)

    reps = pd.read_csv(tables_dir / "representative_series.csv")
    high_id = reps.loc[reps["type"] == "high_volume", "series_id"].iloc[0]
    data = predictions[predictions["series_id"] == high_id].sort_values("dt").head(336)
    plt.figure(figsize=(12, 5))
    plt.plot(data["dt"], data["y_true"], label="Actual", linewidth=1.3)
    plt.plot(data["dt"], data["q50"], label="Quantile LightGBM median", linewidth=1.2)
    plt.fill_between(data["dt"], data["lower_80"], data["upper_80"], alpha=0.25, label="q10-q90 interval")
    plt.title(f"Quantile LightGBM Forecast Interval h{horizon}: {high_id}")
    plt.xlabel("Time")
    plt.ylabel("Sale amount")
    plt.legend()
    plt.grid(alpha=0.25)
    _save(FIGURES_DIR / f"forecast_interval_quantile_lightgbm_h{horizon}.png")


def run_rolling_window_evaluation(tables_dir: Path = TABLES_DIR) -> None:
    """Evaluate model stability across smaller time windows inside the test period."""
    rows = []
    specs = [
        ("LightGBM", 1, "predictions_lightgbm_h1.csv"),
        ("SeasonalNaive168", 1, "predictions_seasonal_naive_168_h1.csv"),
        ("LightGBM", 24, "predictions_lightgbm_h24.csv"),
        ("SeasonalNaive168", 24, "predictions_seasonal_naive_168_h24.csv"),
    ]
    for model, horizon, filename in specs:
        path = tables_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["dt"])
        for window_start, group in df.groupby(pd.Grouper(key="dt", freq="7D")):
            if group.empty:
                continue
            metrics = evaluate_predictions(group["y_true"], group["y_pred"])
            rows.append(
                {
                    "Horizon": horizon,
                    "Model": model,
                    "Window start": window_start,
                    "Window end": group["dt"].max(),
                    **metrics,
                    "N": len(group),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(tables_dir / "rolling_window_evaluation.csv", index=False)


def run_aggregate_sarimax(
    features_path: Path = PROCESSED_DATA_DIR / "fresh50k_features.parquet",
    tables_dir: Path = TABLES_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """Fit an aggregate SARIMAX model as a classical time-series comparator."""
    df = pd.read_parquet(features_path, columns=["dt", "sale_amount"])
    aggregate = df.groupby("dt", sort=True)["sale_amount"].sum().asfreq("h")
    split_point = aggregate.index.max() - pd.Timedelta(days=14) + pd.Timedelta(hours=1)
    train = aggregate[aggregate.index < split_point]
    test = aggregate[aggregate.index >= split_point]

    model = SARIMAX(
        train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=80)
    forecast_result = fitted.get_forecast(steps=len(test))
    forecast = forecast_result.predicted_mean.clip(lower=0)
    intervals = forecast_result.conf_int(alpha=0.20).clip(lower=0)

    seasonal = aggregate.shift(168).reindex(test.index)
    rows = [
        {"Model": "AggregateSARIMAX", **evaluate_predictions(test, forecast), "AIC": float(fitted.aic), "BIC": float(fitted.bic)},
        {"Model": "AggregateSeasonalNaive168", **evaluate_predictions(test, seasonal), "AIC": np.nan, "BIC": np.nan},
    ]
    pd.DataFrame(rows).to_csv(tables_dir / "aggregate_sarimax_comparison.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(test.index, test.values, label="Actual aggregate", linewidth=1.4)
    plt.plot(forecast.index, forecast.values, label="SARIMAX forecast", linewidth=1.2)
    plt.fill_between(
        forecast.index,
        intervals.iloc[:, 0],
        intervals.iloc[:, 1],
        alpha=0.20,
        label="80% SARIMAX interval",
    )
    plt.title("Aggregate SARIMAX Forecast vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Aggregate sale amount")
    plt.legend()
    plt.grid(alpha=0.25)
    _save(figures_dir / "aggregate_sarimax_forecast.png")


def run_stockout_censoring_analysis(
    features_path: Path = PROCESSED_DATA_DIR / "fresh50k_features.parquet",
    tables_dir: Path = TABLES_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """Estimate a simple lost-sales proxy to explain stockout censoring."""
    cols = ["dt", "series_id", "hour", "sale_amount", "stockout_flag", "sale_roll_mean_168"]
    df = pd.read_parquet(features_path, columns=cols)
    df["expected_uncensored_sale"] = df["sale_roll_mean_168"].fillna(df["sale_amount"]).clip(lower=0)
    df["lost_sales_proxy"] = np.where(
        df["stockout_flag"] == 1,
        np.maximum(df["expected_uncensored_sale"] - df["sale_amount"], 0),
        0,
    )
    summary = pd.DataFrame(
        [
            ("Observed sales", float(df["sale_amount"].sum())),
            ("Lost sales proxy during stockout", float(df["lost_sales_proxy"].sum())),
            ("Adjusted demand proxy", float(df["sale_amount"].sum() + df["lost_sales_proxy"].sum())),
            ("Stockout row rate", float(df["stockout_flag"].mean())),
            ("Lost sales proxy / observed sales", float(df["lost_sales_proxy"].sum() / max(df["sale_amount"].sum(), 1e-9))),
        ],
        columns=["Metric", "Value"],
    )
    summary.to_csv(tables_dir / "stockout_censoring_summary.csv", index=False)

    by_hour = df.groupby("hour", as_index=False).agg(
        observed_sales=("sale_amount", "sum"),
        lost_sales_proxy=("lost_sales_proxy", "sum"),
    )
    by_hour.to_csv(tables_dir / "stockout_censoring_by_hour.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(by_hour["hour"], by_hour["observed_sales"], label="Observed sales", alpha=0.85)
    plt.bar(
        by_hour["hour"],
        by_hour["lost_sales_proxy"],
        bottom=by_hour["observed_sales"],
        label="Lost sales proxy",
        alpha=0.75,
    )
    plt.title("Observed Sales and Stockout Lost-Sales Proxy by Hour")
    plt.xlabel("Hour of day")
    plt.ylabel("Sale amount")
    plt.xticks(range(24))
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    _save(figures_dir / "stockout_lost_sales_proxy_by_hour.png")


def run_advanced_analysis() -> None:
    """Run all advanced analyses."""
    write_model_rationale()
    run_quantile_lightgbm()
    run_rolling_window_evaluation()
    run_aggregate_sarimax()
    run_stockout_censoring_analysis()
