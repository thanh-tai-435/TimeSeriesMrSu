"""Machine learning models for Fresh50K forecasting."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .benchmarks import evaluate_prediction_frame, save_predictions
from .config import MODELS_DIR, TABLES_DIR


def load_feature_columns(path: str | Path = TABLES_DIR / "feature_columns.csv") -> list[str]:
    """Load model feature column names."""
    return pd.read_csv(path)["feature"].tolist()


def _target_column(horizon: int) -> str:
    if horizon not in {1, 24}:
        raise ValueError("Only horizon 1 and 24 are supported.")
    return f"target_h{horizon}"


def _select_recent_training_rows(train_df: pd.DataFrame, max_train_rows: int | None) -> pd.DataFrame:
    if max_train_rows is None or max_train_rows <= 0 or len(train_df) <= max_train_rows:
        return train_df
    return train_df.sort_values("dt").tail(max_train_rows)


def _prediction_frame(
    model,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    target_col = _target_column(horizon)
    predictions = test_df[["dt", "series_id", target_col]].rename(columns={target_col: "y_true"}).copy()
    stockout_col = f"target_stockout_flag_h{horizon}"
    if stockout_col in test_df.columns:
        predictions[stockout_col] = test_df[stockout_col].values
    predictions["y_pred"] = model.predict(test_df[feature_columns]).astype("float32")
    predictions["model"] = model_name
    predictions["horizon"] = horizon
    return predictions


def train_ridge_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    max_train_rows: int | None = 1_000_000,
    models_dir: Path = MODELS_DIR,
    tables_dir: Path = TABLES_DIR,
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Train a Ridge baseline and save predictions."""
    target_col = _target_column(horizon)
    train_sample = _select_recent_training_rows(train_df, max_train_rows)
    train_sample = train_sample.dropna(subset=[target_col])

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", Ridge(alpha=1.0, solver="lsqr")),
        ]
    )
    model.fit(train_sample[feature_columns], train_sample[target_col])

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / f"ridge_h{horizon}.pkl")

    predictions = _prediction_frame(model, test_df, feature_columns, horizon, "Ridge")
    save_predictions(predictions, tables_dir / f"predictions_ridge_h{horizon}.csv")
    metrics = evaluate_prediction_frame(predictions)
    metrics["Train rows used"] = len(train_sample)
    return model, predictions, metrics


def train_lightgbm_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    max_train_rows: int | None = 1_000_000,
    models_dir: Path = MODELS_DIR,
    tables_dir: Path = TABLES_DIR,
) -> tuple[object, pd.DataFrame, dict]:
    """Train a LightGBM global forecasting model and save predictions."""
    try:
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    except ImportError as exc:
        raise ImportError("LightGBM is not installed. Run `pip install lightgbm`.") from exc

    target_col = _target_column(horizon)
    train_sample = _select_recent_training_rows(train_df, max_train_rows)
    train_sample = train_sample.dropna(subset=[target_col])
    val_sample = val_df.dropna(subset=[target_col])

    model = LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train_sample[feature_columns],
        train_sample[target_col],
        eval_set=[(val_sample[feature_columns], val_sample[target_col])],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=100)],
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / f"lightgbm_h{horizon}.pkl")

    predictions = _prediction_frame(model, test_df, feature_columns, horizon, "LightGBM")
    save_predictions(predictions, tables_dir / f"predictions_lightgbm_h{horizon}.csv")
    metrics = evaluate_prediction_frame(predictions)
    metrics["Train rows used"] = len(train_sample)

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
            "horizon": horizon,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(tables_dir / f"lightgbm_feature_importance_h{horizon}.csv", index=False)
    if horizon == 1:
        importance.to_csv(tables_dir / "lightgbm_feature_importance.csv", index=False)
    return model, predictions, metrics


def train_xgboost_model(*args, **kwargs):
    """Optionally train an XGBoost model."""
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return None, None, {"status": "skipped", "reason": "xgboost is not installed"}
    return None, None, {"status": "skipped", "reason": "xgboost training is not enabled by default"}
