"""Run Fresh50K Phase 8 LightGBM ablation study."""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.benchmarks import evaluate_prediction_frame, save_predictions
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, TABLES_DIR, ensure_directories
from src.models import load_feature_columns
from src.split import load_feature_table, time_based_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightGBM ablation study.")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 24])
    parser.add_argument("--val_days", type=int, default=7)
    parser.add_argument("--test_days", type=int, default=14)
    parser.add_argument("--max_train_rows", type=int, default=500_000)
    parser.add_argument("--features_path", default=str(PROCESSED_DATA_DIR / "fresh50k_features.parquet"))
    return parser.parse_args()


def feature_sets(all_features: list[str]) -> dict[str, list[str]]:
    id_cols = [
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
    ]
    calendar = [
        "hour",
        "day_of_week",
        "is_weekend",
        "day_of_month",
        "week_of_year",
        "month",
        "sin_hour",
        "cos_hour",
        "sin_dayofweek",
        "cos_dayofweek",
    ]
    lag_roll = [feature for feature in all_features if feature.startswith("sale_lag_") or feature.startswith("sale_roll_")]
    promotion = ["discount", "holiday_flag", "activity_flag", "discount_lag_1", "discount_lag_24", "discount_roll_mean_24"]
    weather = ["precip", "temp", "humidity", "wind", "temp_lag_1", "temp_lag_24", "precip_lag_1"]
    stockout = [feature for feature in all_features if feature.startswith("stockout") or feature == "stockout_rate"]

    def present(cols: list[str]) -> list[str]:
        return [col for col in cols if col in all_features]

    base = present(id_cols + calendar + lag_roll)
    plus_promo = present(base + promotion)
    plus_weather = present(plus_promo + weather)
    full = present(plus_weather + stockout)
    return {
        "Lag + Calendar": base,
        "Lag + Calendar + Promotion": plus_promo,
        "Lag + Calendar + Promotion + Weather": plus_weather,
        "Lag + Calendar + Promotion + Weather + Stockout": full,
    }


def train_lgbm(train_df, val_df, features, target_col):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

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
        train_df[features],
        train_df[target_col],
        eval_set=[(val_df[features], val_df[target_col])],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=40), log_evaluation(period=0)],
    )
    return model


def main() -> None:
    args = parse_args()
    ensure_directories()
    all_features = load_feature_columns(TABLES_DIR / "feature_columns.csv")
    target_col = f"target_h{args.horizon}"
    stockout_col = f"target_stockout_flag_h{args.horizon}"
    needed = ["dt", "series_id", target_col, stockout_col, *all_features]
    df = load_feature_table(args.features_path)
    df = df[[column for column in needed if column in df.columns]]
    train_df, val_df, test_df = time_based_split(df, args.val_days, args.test_days, TABLES_DIR / "split_summary.csv")
    train_sample = train_df.sort_values("dt").tail(args.max_train_rows)

    rows = []
    best_wape = float("inf")
    best_model = None
    best_name = None
    for name, features in feature_sets(all_features).items():
        model = train_lgbm(train_sample, val_df, features, target_col)
        pred = test_df[["dt", "series_id", target_col]].rename(columns={target_col: "y_true"}).copy()
        if stockout_col in test_df.columns:
            pred[stockout_col] = test_df[stockout_col].values
        pred["y_pred"] = model.predict(test_df[features]).astype("float32")
        pred["model"] = f"LightGBM {name}"
        pred["horizon"] = args.horizon
        metrics = evaluate_prediction_frame(pred)
        rows.append(
            {
                "Horizon": args.horizon,
                "Feature set": name,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "WAPE": metrics["WAPE"],
                "sMAPE": metrics["sMAPE"],
                "N": metrics["N"],
                "Train rows used": len(train_sample),
            }
        )
        safe_name = name.lower().replace(" + ", "_").replace(" ", "_")
        if args.horizon == 1 and name in {"Lag + Calendar", "Lag + Calendar + Promotion + Weather + Stockout"}:
            save_predictions(pred, TABLES_DIR / f"predictions_ablation_{safe_name}_h1.csv")
        if metrics["WAPE"] < best_wape:
            best_wape = metrics["WAPE"]
            best_model = model
            best_name = name

    result = pd.DataFrame(rows).sort_values(["Horizon", "WAPE"]).reset_index(drop=True)
    output = TABLES_DIR / "ablation_results.csv"
    if output.exists():
        old = pd.read_csv(output)
        old = old[old["Horizon"] != args.horizon]
        result = pd.concat([old, result], ignore_index=True).sort_values(["Horizon", "WAPE"])
    result.to_csv(output, index=False)
    if best_model is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODELS_DIR / f"lightgbm_ablation_best_h{args.horizon}.pkl")
    print(f"Saved ablation results: {output}")
    print(f"Best feature set h{args.horizon}: {best_name}")
    print(result[result["Horizon"] == args.horizon].to_string(index=False))


if __name__ == "__main__":
    main()
