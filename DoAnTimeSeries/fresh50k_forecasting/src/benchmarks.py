"""Naive and seasonal-naive benchmark models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import TABLES_DIR


def _target_column(horizon: int) -> str:
    if horizon not in {1, 24}:
        raise ValueError("Only horizon 1 and 24 are supported.")
    return f"target_h{horizon}"


def _prediction_frame(
    df: pd.DataFrame,
    horizon: int,
    lag: int,
    model_name: str,
) -> pd.DataFrame:
    target_col = _target_column(horizon)
    lag_col = f"sale_lag_{lag}"
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if lag_col not in df.columns:
        raise ValueError(f"Missing lag column: {lag_col}")

    columns = ["dt", "series_id", target_col, lag_col]
    stockout_target_col = f"target_stockout_flag_h{horizon}"
    if stockout_target_col in df.columns:
        columns.append(stockout_target_col)

    predictions = df[columns].rename(columns={target_col: "y_true", lag_col: "y_pred"}).copy()
    predictions["model"] = model_name
    predictions["horizon"] = horizon
    predictions = predictions.dropna(subset=["y_true", "y_pred"])
    predictions["y_true"] = predictions["y_true"].astype("float32")
    predictions["y_pred"] = predictions["y_pred"].astype("float32")
    return predictions


def make_naive_predictions(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create naive predictions for a horizon."""
    lag = 1 if horizon == 1 else 24
    return _prediction_frame(df, horizon=horizon, lag=lag, model_name="Naive")


def make_seasonal_naive_predictions(
    df: pd.DataFrame,
    horizon: int,
    lag: int,
) -> pd.DataFrame:
    """Create seasonal naive predictions using a lag."""
    return _prediction_frame(df, horizon=horizon, lag=lag, model_name=f"SeasonalNaive{lag}")


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted absolute percentage error."""
    denominator = float(np.sum(np.abs(y_true)))
    if denominator == 0:
        return np.nan
    return float(np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denominator)


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Symmetric mean absolute percentage error."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    denominator = np.abs(y_true_arr) + np.abs(y_pred_arr)
    values = np.divide(
        2 * np.abs(y_true_arr - y_pred_arr),
        denominator,
        out=np.zeros_like(y_true_arr, dtype=float),
        where=denominator != 0,
    )
    return float(np.mean(values))


def evaluate_prediction_frame(predictions: pd.DataFrame) -> dict[str, float | int | str]:
    """Compute metrics for one prediction dataframe."""
    return {
        "Horizon": int(predictions["horizon"].iloc[0]),
        "Model": str(predictions["model"].iloc[0]),
        "RMSE": rmse(predictions["y_true"], predictions["y_pred"]),
        "MAE": mae(predictions["y_true"], predictions["y_pred"]),
        "WAPE": wape(predictions["y_true"], predictions["y_pred"]),
        "sMAPE": smape(predictions["y_true"], predictions["y_pred"]),
        "N": len(predictions),
    }


def save_predictions(predictions: pd.DataFrame, output_path: str | Path) -> None:
    """Save prediction dataframe as CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)


def run_benchmarks(
    test_df: pd.DataFrame,
    tables_dir: Path = TABLES_DIR,
) -> pd.DataFrame:
    """Run Phase 6 benchmark models on the test split."""
    specs = [
        ("predictions_naive_h1.csv", make_naive_predictions(test_df, horizon=1)),
        ("predictions_naive_h24.csv", make_naive_predictions(test_df, horizon=24)),
        (
            "predictions_seasonal_naive_24_h1.csv",
            make_seasonal_naive_predictions(test_df, horizon=1, lag=24),
        ),
        (
            "predictions_seasonal_naive_24_h24.csv",
            make_seasonal_naive_predictions(test_df, horizon=24, lag=24),
        ),
        (
            "predictions_seasonal_naive_168_h1.csv",
            make_seasonal_naive_predictions(test_df, horizon=1, lag=168),
        ),
        (
            "predictions_seasonal_naive_168_h24.csv",
            make_seasonal_naive_predictions(test_df, horizon=24, lag=168),
        ),
    ]

    metric_rows = []
    for filename, predictions in specs:
        save_predictions(predictions, tables_dir / filename)
        metric_rows.append(evaluate_prediction_frame(predictions))

    metrics = pd.DataFrame(metric_rows).sort_values(["Horizon", "WAPE", "RMSE"]).reset_index(drop=True)
    metrics.to_csv(tables_dir / "benchmark_metrics.csv", index=False)
    return metrics
