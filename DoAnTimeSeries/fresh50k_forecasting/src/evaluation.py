"""Evaluation metrics used by the final two-stage pipeline."""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error."""
    denominator = float(np.sum(np.abs(y_true)))
    if denominator == 0:
        return np.nan
    return float(np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denominator)


def smape(y_true, y_pred) -> float:
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


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    """Return the metrics used in result tables."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }
