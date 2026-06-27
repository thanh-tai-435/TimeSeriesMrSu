"""Evaluation metrics and comparison tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import TABLES_DIR


PREDICTION_SPECS = [
    (1, "Naive", "predictions_naive_h1.csv"),
    (1, "SeasonalNaive24", "predictions_seasonal_naive_24_h1.csv"),
    (1, "SeasonalNaive168", "predictions_seasonal_naive_168_h1.csv"),
    (1, "Ridge", "predictions_ridge_h1.csv"),
    (1, "LightGBM", "predictions_lightgbm_h1.csv"),
    (24, "Naive", "predictions_naive_h24.csv"),
    (24, "SeasonalNaive24", "predictions_seasonal_naive_24_h24.csv"),
    (24, "SeasonalNaive168", "predictions_seasonal_naive_168_h24.csv"),
    (24, "Ridge", "predictions_ridge_h24.csv"),
    (24, "LightGBM", "predictions_lightgbm_h24.csv"),
]


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def mae(y_true, y_pred):
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def wape(y_true, y_pred):
    """Weighted absolute percentage error."""
    denominator = float(np.sum(np.abs(y_true)))
    if denominator == 0:
        return np.nan
    return float(np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denominator)


def smape(y_true, y_pred):
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


def evaluate_predictions(y_true, y_pred):
    """Return all evaluation metrics as a dictionary."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def _read_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["dt"])
    return df


def evaluate_prediction_file(path: Path, horizon: int, model: str) -> dict:
    """Evaluate one prediction file."""
    df = _read_prediction(path)
    metrics = evaluate_predictions(df["y_true"], df["y_pred"])
    return {"Horizon": horizon, "Model": model, **metrics, "N": len(df)}


def create_model_comparison_tables(tables_dir: Path = TABLES_DIR) -> pd.DataFrame:
    """Create h1/h24 model comparison tables from prediction files."""
    rows = []
    for horizon, model, filename in PREDICTION_SPECS:
        path = tables_dir / filename
        if path.exists():
            rows.append(evaluate_prediction_file(path, horizon, model))
    comparison = pd.DataFrame(rows).sort_values(["Horizon", "WAPE", "RMSE"]).reset_index(drop=True)
    for horizon in sorted(comparison["Horizon"].unique()):
        comparison[comparison["Horizon"] == horizon].to_csv(
            tables_dir / f"model_comparison_h{horizon}.csv",
            index=False,
        )
    return comparison


def create_stockout_segment_evaluation(tables_dir: Path = TABLES_DIR) -> pd.DataFrame:
    """Evaluate predictions overall, during stockout periods, and outside stockout periods."""
    rows = []
    for horizon, model, filename in PREDICTION_SPECS:
        path = tables_dir / filename
        if not path.exists():
            continue
        df = _read_prediction(path)
        stockout_col = f"target_stockout_flag_h{horizon}"
        segments = {"overall": df}
        if stockout_col in df.columns:
            segments["stockout_period"] = df[df[stockout_col] == 1]
            segments["non_stockout_period"] = df[df[stockout_col] == 0]
        for segment, segment_df in segments.items():
            if segment_df.empty:
                continue
            metrics = evaluate_predictions(segment_df["y_true"], segment_df["y_pred"])
            rows.append(
                {
                    "Horizon": horizon,
                    "Model": model,
                    "Segment": segment,
                    **metrics,
                    "N": len(segment_df),
                }
            )
    result = pd.DataFrame(rows).sort_values(["Horizon", "Model", "Segment"]).reset_index(drop=True)
    result.to_csv(tables_dir / "stockout_segment_evaluation.csv", index=False)
    return result


def create_group_evaluation(tables_dir: Path = TABLES_DIR) -> None:
    """Create optional store-level evaluation from LightGBM h1 predictions."""
    path = tables_dir / "predictions_lightgbm_h1.csv"
    if not path.exists():
        return
    df = _read_prediction(path)
    parts = df["series_id"].str.split("_", expand=True)
    if parts.shape[1] >= 2:
        df["store_id"] = parts[1]
        rows = []
        for store_id, group in df.groupby("store_id"):
            metrics = evaluate_predictions(group["y_true"], group["y_pred"])
            rows.append({"Group": store_id, **metrics, "N": len(group)})
        pd.DataFrame(rows).sort_values("WAPE").to_csv(tables_dir / "evaluation_by_store.csv", index=False)


def write_result_summary(tables_dir: Path = TABLES_DIR) -> Path:
    """Generate a short markdown summary from saved metrics."""
    reports_dir = tables_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(tables_dir / "model_comparison_h1.csv")
    best = comparison.sort_values("WAPE").iloc[0]
    seasonal = comparison[comparison["Model"] == "SeasonalNaive168"].iloc[0]
    improvement = (seasonal["WAPE"] - best["WAPE"]) / seasonal["WAPE"] * 100

    ablation_text = "Ablation results were not available."
    ablation_path = tables_dir / "ablation_results.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        h1 = ablation[ablation["Horizon"] == 1]
        if not h1.empty:
            base = h1[h1["Feature set"] == "Lag + Calendar"]
            full = h1[h1["Feature set"] == "Lag + Calendar + Promotion + Weather + Stockout"]
            if not base.empty and not full.empty:
                delta = (base.iloc[0]["WAPE"] - full.iloc[0]["WAPE"]) / base.iloc[0]["WAPE"] * 100
                ablation_text = (
                    f"Adding promotion, weather, and stockout features changed WAPE from "
                    f"{base.iloc[0]['WAPE']:.4f} to {full.iloc[0]['WAPE']:.4f}, "
                    f"an improvement of {delta:.2f}%."
                )

    segment_text = "Stockout-segment evaluation was generated in stockout_segment_evaluation.csv."
    segment_path = tables_dir / "stockout_segment_evaluation.csv"
    if segment_path.exists():
        segment = pd.read_csv(segment_path)
        lgbm_stockout = segment[
            (segment["Horizon"] == 1)
            & (segment["Model"] == "LightGBM")
            & (segment["Segment"] == "stockout_period")
        ]
        if not lgbm_stockout.empty:
            segment_text = f"During stockout periods, LightGBM h1 achieved WAPE {lgbm_stockout.iloc[0]['WAPE']:.4f}."

    content = f"""# Result Summary

Best model by WAPE on h1:
- Model: {best['Model']}
- WAPE: {best['WAPE']:.4f}
- Improvement over SeasonalNaive168: {improvement:.2f}%

Ablation result:
- {ablation_text}

Stockout-period result:
- {segment_text}
"""
    output = reports_dir / "result_summary.md"
    output.write_text(content, encoding="utf-8")
    return output


def run_evaluation(tables_dir: Path = TABLES_DIR) -> None:
    """Run Phase 9 and report summary generation."""
    create_model_comparison_tables(tables_dir)
    create_stockout_segment_evaluation(tables_dir)
    create_group_evaluation(tables_dir)
    write_result_summary(tables_dir)
