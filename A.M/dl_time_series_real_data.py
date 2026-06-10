"""
=============================================================================
 Deep Learning for Time Series Forecasting - REAL DATA Module
=============================================================================
Companion to `dl_time_series_practice.py`. That file taught the patterns on a
synthetic series so everything was fast and self-contained. This file applies
the SAME MLP / CNN / LSTM patterns to REAL datasets used throughout Jason
Brownlee's "Deep Learning for Time Series Forecasting".

Datasets (downloaded from Jason Brownlee's public GitHub repo on first run,
then cached locally so you only download once):

    airline   - Monthly international airline passengers, 1949-1960 (144 obs)
                Strong upward TREND + yearly SEASONALITY. Great for showing
                why scaling matters and why a naive model struggles.

    temps     - Daily minimum temperatures in Melbourne, 1981-1990 (3650 obs)
                Seasonal, noisy, no strong trend. A larger, harder series.

WHAT THIS FILE ADDS BEYOND THE SYNTHETIC VERSION
------------------------------------------------
1. Loading a real CSV with pandas and parsing the date column.
2. SCALING the data (essential for neural nets) and, crucially,
   INVERSE-scaling predictions so errors are reported in real units
   (passengers, degrees C) - not meaningless scaled numbers.
3. Fitting the scaler on TRAIN ONLY, then applying it to test data, so we
   never leak information from the future into preprocessing.
4. A walk-forward style evaluation on a held-out chronological tail.

Run it:
    python dl_time_series_real_data.py                 # airline, all models
    python dl_time_series_real_data.py temps lstm      # temps, LSTM only
    python dl_time_series_real_data.py airline compare # head-to-head

Requirements: tensorflow (or tensorflow-cpu), numpy, pandas, scikit-learn
=============================================================================
"""

import os
import sys
import urllib.request

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Reuse the model builders and windowing from the teaching module so students
# see that NOTHING about the architectures changes for real data - only the
# loading and scaling around them does.
from dl_time_series_practice import (
    split_sequence,
    build_mlp,
    build_cnn,
    build_lstm_vanilla,
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# SECTION 1 - LOADING REAL DATA
# =============================================================================

DATASETS = {
    "airline": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
        "file": "airline-passengers.csv",
        "value_col": "Passengers",
        "date_col": "Month",
        "unit": "passengers",
    },
    "temps": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        "file": "daily-min-temperatures.csv",
        "value_col": "Temp",
        "date_col": "Date",
        "unit": "degrees C",
    },
}


def load_series(name, cache_dir="."):
    """Download (once) and load a real univariate series as a NumPy array.

    Returns
        values: 1D float32 array of the measured values, in time order
        meta:   the dataset's metadata dict (column names, units, ...)
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASETS)}.")
    meta = DATASETS[name]
    path = os.path.join(cache_dir, meta["file"])

    if not os.path.exists(path):
        print(f"Downloading {name} dataset ...")
        urllib.request.urlretrieve(meta["url"], path)
        print(f"Saved to {path}")

    df = pd.read_csv(path)
    values = df[meta["value_col"]].astype("float32").to_numpy()
    print(f"Loaded '{name}': {len(values)} observations "
          f"(min={values.min():.1f}, max={values.max():.1f} {meta['unit']})")
    return values, meta


# =============================================================================
# SECTION 2 - SCALING DONE RIGHT (no data leakage)
# =============================================================================
# Neural networks train far better when inputs are on a small, consistent
# scale (here, roughly 0-1). The golden rule: FIT the scaler on the TRAINING
# data only, then TRANSFORM both train and test with it. Fitting on the whole
# series would leak information about the future (its max/min) into training.

def chronological_split_values(values, test_frac=0.2):
    """Split the raw series into past (train) and recent tail (test)."""
    n_test = max(1, int(len(values) * test_frac))
    return values[:-n_test], values[-n_test:]


def scale_train_test(train_vals, test_vals):
    """Fit a 0-1 scaler on TRAIN only; return scaled arrays and the scaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).ravel()
    test_scaled = scaler.transform(test_vals.reshape(-1, 1)).ravel()
    return train_scaled, test_scaled, scaler


def windows_from_scaled(train_scaled, test_scaled, n_steps):
    """Build (X, y) windows for train and test in SCALED space.

    To predict the very first test points we need the last `n_steps` training
    values as context, so we prepend them to the test series before windowing.
    This mimics forecasting forward from the end of the known history.
    """
    X_tr, y_tr = split_sequence(train_scaled, n_steps)

    bridged = np.concatenate([train_scaled[-n_steps:], test_scaled])
    X_te, y_te = split_sequence(bridged, n_steps)
    return X_tr, y_tr, X_te, y_te


def report(y_true_real, y_pred_real, unit, model_name):
    """Print RMSE and MAE in the dataset's REAL units."""
    rmse = float(np.sqrt(mean_squared_error(y_true_real, y_pred_real)))
    mae = float(mean_absolute_error(y_true_real, y_pred_real))
    print(f"{model_name:<8} RMSE: {rmse:8.3f} {unit:<12} MAE: {mae:8.3f} {unit}")
    return rmse


# =============================================================================
# SECTION 3 - TRAIN ONE MODEL ON REAL DATA (full, leak-free pipeline)
# =============================================================================

def run_model(kind, values, meta, n_steps=12, epochs=200, verbose=True):
    """End-to-end: split -> scale -> window -> train -> predict -> un-scale.

    kind: 'mlp', 'cnn', or 'lstm'
    Returns the test RMSE in real units.
    """
    # 1) Chronological split BEFORE any scaling.
    train_vals, test_vals = chronological_split_values(values)

    # 2) Scale using train statistics only.
    train_scaled, test_scaled, scaler = scale_train_test(train_vals, test_vals)

    # 3) Build supervised windows in scaled space.
    X_tr, y_tr, X_te, y_te = windows_from_scaled(train_scaled, test_scaled, n_steps)

    # 4) Reshape + build the chosen architecture.
    if kind == "mlp":
        model = build_mlp(n_steps)
        Xtr, Xte = X_tr, X_te                      # MLP wants 2D input
    elif kind == "cnn":
        model = build_cnn(n_steps)
        Xtr = X_tr.reshape((X_tr.shape[0], n_steps, 1))   # CNN/LSTM want 3D
        Xte = X_te.reshape((X_te.shape[0], n_steps, 1))
    elif kind == "lstm":
        model = build_lstm_vanilla(n_steps)
        Xtr = X_tr.reshape((X_tr.shape[0], n_steps, 1))
        Xte = X_te.reshape((X_te.shape[0], n_steps, 1))
    else:
        raise ValueError("kind must be 'mlp', 'cnn', or 'lstm'")

    # 5) Train.
    model.fit(Xtr, y_tr, epochs=epochs, verbose=0)

    # 6) Predict, then INVERSE-scale back to real units before scoring.
    pred_scaled = model.predict(Xte, verbose=0).reshape(-1, 1)
    pred_real = scaler.inverse_transform(pred_scaled).ravel()
    y_te_real = scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()

    rmse = report(y_te_real, pred_real, meta["unit"], kind.upper()) if verbose else \
        float(np.sqrt(mean_squared_error(y_te_real, pred_real)))
    return rmse, y_te_real, pred_real


def naive_baseline(values, test_frac=0.2):
    """Persistence baseline: predict 'tomorrow = today'.

    Any real model MUST beat this to be worth the complexity. Always compute
    a naive baseline before celebrating a neural network's score.
    """
    train_vals, test_vals = chronological_split_values(values, test_frac)
    bridged = np.concatenate([train_vals[-1:], test_vals])
    y_true = bridged[1:]
    y_pred = bridged[:-1]                  # shifted by one step
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# =============================================================================
# SECTION 4 - DEMOS
# =============================================================================

def demo_single(dataset="airline", kind="lstm", n_steps=12, epochs=200):
    print_header(f"REAL DATA: {kind.upper()} on '{dataset}'")
    values, meta = load_series(dataset)
    base = naive_baseline(values)
    print(f"Naive persistence baseline RMSE: {base:.3f} {meta['unit']}\n")
    rmse, y_true, y_pred = run_model(kind, values, meta, n_steps, epochs)

    print("\nLast 5 test points (actual -> predicted):")
    for a, p in zip(y_true[-5:], y_pred[-5:]):
        print(f"    {a:8.2f}  ->  {p:8.2f}")
    verdict = "beats" if rmse < base else "does NOT beat"
    print(f"\n{kind.upper()} {verdict} the naive baseline "
          f"({rmse:.2f} vs {base:.2f} {meta['unit']}).")


def demo_compare(dataset="airline", n_steps=12, epochs=200):
    print_header(f"REAL DATA: MLP vs CNN vs LSTM on '{dataset}'")
    values, meta = load_series(dataset)
    base = naive_baseline(values)
    print(f"Naive persistence baseline RMSE: {base:.3f} {meta['unit']}\n")

    results = {"Naive": base}
    for kind in ("mlp", "cnn", "lstm"):
        rmse, _, _ = run_model(kind, values, meta, n_steps, epochs, verbose=True)
        results[kind.upper()] = rmse

    print("\n" + "-" * 34)
    print(f"{'Model':<10}{'RMSE (' + meta['unit'] + ')':>22}")
    print("-" * 34)
    for name, score in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"{name:<10}{score:>22.3f}")
    print("\n(Lower is better. Models below the Naive row are earning their keep.)")


# =============================================================================
# SECTION 5 - EXERCISES
# =============================================================================

def exercise_temps_window():
    """EXERCISE (open): On the 'temps' dataset, the natural cycle is ~365 days.

    Try n_steps in [7, 14, 30, 90] with a CNN and see which captures the
    seasonal structure best. Note: longer windows = fewer samples and slower
    training. Use run_model('cnn', values, meta, n_steps=...) in a loop and
    compare RMSE against naive_baseline(values).
    """
    print_header("EXERCISE 4.4: Window-size study on 'temps' dataset (CNN)")
    values, meta = load_series("temps")
    base = naive_baseline(values)
    print(f"Naive persistence baseline RMSE: {base:.3f} {meta['unit']}\n")

    windows = [7, 14, 30, 90]
    print(f"{'n_steps':>8}  {'RMSE':>8}  {'vs Naive':>10}  {'Beat?':>7}")
    print("-" * 42)
    results = {}
    for n_steps in windows:
        r, _, _ = run_model("cnn", values, meta, n_steps=n_steps, epochs=200, verbose=False)
        diff = r - base
        beat = "YES" if r < base else "NO"
        sign = "+" if diff >= 0 else ""
        print(f"{n_steps:>8}  {r:>8.3f}  {sign}{diff:>8.3f}  {beat:>7}")
        results[n_steps] = r

    best = min(results, key=results.get)
    print(f"\nBest window: n_steps={best}  RMSE={results[best]:.3f}")
    print(f"Trade-off: longer window -> richer context but fewer training samples.")
    return results


# =============================================================================
# Plumbing
# =============================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    args = [a.lower() for a in sys.argv[1:]]
    dataset = next((a for a in args if a in DATASETS), "airline")
    mode = next((a for a in args if a in ("mlp", "cnn", "lstm", "compare", "window")), "compare")

    # Daily temps benefit from a longer window than monthly airline data.
    n_steps = 30 if dataset == "temps" else 12

    if mode == "compare":
        demo_compare(dataset, n_steps=n_steps)
    elif mode == "window":
        exercise_temps_window()
    else:
        demo_single(dataset, kind=mode, n_steps=n_steps)


if __name__ == "__main__":
    main()
