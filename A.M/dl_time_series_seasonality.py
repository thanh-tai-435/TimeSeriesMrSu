"""
=============================================================================
 Deep Learning for Time Series Forecasting - SEASONALITY EXPERIMENT
=============================================================================
Companion to the EDA, practice, and real-data modules.

QUESTION: The EDA showed both real series are strongly seasonal. Does
EXPLICITLY removing that seasonality before modeling improve forecasts?

We compare two pipelines on the SAME data, split, and models:

  WITHOUT seasonality handling  -> train on the raw series (what we did before).
  WITH    seasonality handling  -> SEASONAL ADJUSTMENT:
        1. Learn an additive seasonal pattern from the TRAINING data only.
        2. Subtract it from the series  (deseasonalize / "seasonally adjust").
        3. Train the model on the smoother, deseasonalized series.
        4. Add the seasonal pattern BACK to the forecasts before scoring.

The seasonal pattern is keyed to position-in-cycle (month-of-year for the
airline data, day-of-year for the temperatures), so re-adding it on the test
set uses only information known in advance. Nothing leaks from the future.

Run it:
    python dl_time_series_seasonality.py                 # airline, all models
    python dl_time_series_seasonality.py temps           # temperatures
    python dl_time_series_seasonality.py airline lstm    # one model only

Requirements: tensorflow, numpy, pandas, scikit-learn, statsmodels, matplotlib
=============================================================================
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

from dl_time_series_real_data import (
    DATASETS, load_series, chronological_split_values,
    scale_train_test, windows_from_scaled, naive_baseline,
)
from dl_time_series_practice import build_mlp, build_cnn, build_lstm_vanilla

SEASONAL_PERIOD = {"airline": 12, "temps": 365}
OUTDIR = "seasonality_outputs"


# =============================================================================
# 1 - LEARN AND APPLY A SEASONAL PATTERN (leak-free)
# =============================================================================

def seasonal_pattern_from_train(train_vals, period):
    """Estimate an ADDITIVE seasonal pattern of length `period` from TRAIN only.

    We decompose the training series, then average the seasonal component at
    each position in the cycle (e.g. all Januaries together). The result is a
    repeating pattern indexed 0..period-1, centred to have mean zero.
    """
    seas = seasonal_decompose(
        pd.Series(train_vals), model="additive", period=period
    ).seasonal.to_numpy()
    pattern = np.array([np.nanmean(seas[j::period]) for j in range(period)])
    pattern = pattern - np.nanmean(pattern)          # centre to mean zero
    return pattern.astype("float32")


def season_at(pattern, indices):
    """Seasonal value for each global time index, by position in the cycle."""
    period = len(pattern)
    return pattern[np.asarray(indices) % period]


# =============================================================================
# 2 - ONE RUN, WITH OR WITHOUT SEASONAL ADJUSTMENT
# =============================================================================

def _build(kind, n_steps):
    if kind == "mlp":
        return build_mlp(n_steps)
    if kind == "cnn":
        return build_cnn(n_steps)
    if kind == "lstm":
        return build_lstm_vanilla(n_steps)
    raise ValueError("kind must be 'mlp', 'cnn', or 'lstm'")


def run(kind, values, period, n_steps, epochs, handle_seasonality):
    """Full pipeline returning (rmse, y_true_real, y_pred_real) in real units."""
    n = len(values)
    train_vals, test_vals = chronological_split_values(values)
    n_train = len(train_vals)

    if handle_seasonality:
        pattern = seasonal_pattern_from_train(train_vals, period)
        seas_all = season_at(pattern, np.arange(n))
        work = values - seas_all                     # deseasonalized series
        work_train, work_test = work[:n_train], work[n_train:]
    else:
        work_train, work_test = train_vals, test_vals

    # Scale on (deseasonalized) training data only, then window.
    tr_s, te_s, scaler = scale_train_test(work_train, work_test)
    X_tr, y_tr, X_te, y_te = windows_from_scaled(tr_s, te_s, n_steps)

    if kind in ("cnn", "lstm"):
        X_tr = X_tr.reshape((X_tr.shape[0], n_steps, 1))
        X_te = X_te.reshape((X_te.shape[0], n_steps, 1))

    model = _build(kind, n_steps)
    model.fit(X_tr, y_tr, epochs=epochs, verbose=0)

    pred_s = model.predict(X_te, verbose=0).reshape(-1, 1)
    pred_work = scaler.inverse_transform(pred_s).ravel()   # in (de)seasonalized real units

    # Test targets correspond to global indices n_train .. n-1.
    test_idx = np.arange(n_train, n)
    if handle_seasonality:
        pred_real = pred_work + season_at(pattern, test_idx)   # add seasonality back
    else:
        pred_real = pred_work

    y_true_real = values[n_train:n]                            # original test values
    rmse = float(np.sqrt(mean_squared_error(y_true_real, pred_real)))
    return rmse, y_true_real, pred_real


# =============================================================================
# 3 - COMPARISON DRIVER
# =============================================================================

# A short window (shorter than the seasonal period) is the interesting case:
# the model CANNOT see a full cycle, so removing seasonality should help most.
SHORT_WINDOW = {"airline": 6, "temps": 30}
SWEEP_WINDOWS = {"airline": [3, 6, 12, 24], "temps": [7, 14, 30, 90]}


def compare(dataset="airline", kinds=("mlp", "cnn", "lstm"), n_steps=None, epochs=200):
    print_header(f"SEASONALITY EXPERIMENT: with vs without, on '{dataset}'")
    values, meta = load_series(dataset)
    period = SEASONAL_PERIOD[dataset]
    if n_steps is None:
        n_steps = SHORT_WINDOW[dataset]

    base = naive_baseline(values)
    print(f"Seasonal period: {period}   |   Window n_steps: {n_steps} "
          f"(deliberately shorter than the period)")
    print(f"Naive baseline RMSE: {base:.3f} {meta['unit']}\n")

    rows = {}
    for kind in kinds:
        rmse_off, _, _ = run(kind, values, period, n_steps, epochs, handle_seasonality=False)
        rmse_on, _, _ = run(kind, values, period, n_steps, epochs, handle_seasonality=True)
        rows[kind.upper()] = (rmse_off, rmse_on)

    print(f"{'Model':<8}{'Without':>12}{'With':>12}{'Improvement':>14}")
    print("-" * 46)
    for name, (off, on) in rows.items():
        improve = 100.0 * (off - on) / off
        print(f"{name:<8}{off:>12.3f}{on:>12.3f}{improve:>13.1f}%")
    print(f"\nUnits: {meta['unit']}. 'Improvement' = % drop in RMSE from removing "
          f"seasonality before modeling (higher is better).")

    # The key insight: how the benefit depends on window length vs the period.
    window_sweep(dataset, values, period, meta, epochs)

    # Forecast-vs-actual plot at the short window, with vs without.
    _plot_with_without(dataset, meta, period, n_steps, epochs)
    return rows


def window_sweep(dataset, values, period, meta, epochs=200):
    """Show how seasonal adjustment's benefit changes with the window length.

    Lesson: when the window is SHORTER than the seasonal period, the model
    cannot see a full cycle and seasonal adjustment helps a lot. When the
    window is long enough to contain a full cycle, the raw model can learn the
    seasonality itself and the benefit shrinks or reverses.
    """
    print(f"\nWindow-length sweep (LSTM, period={period}):")
    print(f"{'n_steps':>8}{'Without':>10}{'With':>10}{'Improvement':>14}")
    print("-" * 42)
    for ns in SWEEP_WINDOWS[dataset]:
        off, _, _ = run("lstm", values, period, ns, epochs, handle_seasonality=False)
        on, _, _ = run("lstm", values, period, ns, epochs, handle_seasonality=True)
        flag = "  <- window < period" if ns < period else ""
        print(f"{ns:>8}{off:>10.2f}{on:>10.2f}{100*(off-on)/off:>12.1f}%{flag}")


def _plot_with_without(dataset, meta, period, n_steps, epochs):
    """Save a forecast-vs-actual plot for an LSTM, with and without adjustment."""
    values, _ = load_series(dataset)
    rmse_off, y_true, pred_off = run("lstm", values, period, n_steps, epochs, False)
    rmse_on, _,      pred_on  = run("lstm", values, period, n_steps, epochs, True)

    plt.figure(figsize=(11, 4.5))
    plt.plot(y_true, label="actual", color="black", linewidth=2)
    plt.plot(pred_off, label=f"LSTM without (RMSE {rmse_off:.1f})", linestyle="--")
    plt.plot(pred_on, label=f"LSTM with seasonal adj. (RMSE {rmse_on:.1f})", linestyle="-.")
    plt.title(f"{dataset}: forecast vs actual on test set (window={n_steps} < period={period})")
    plt.xlabel("test time step"); plt.ylabel(meta["unit"])
    plt.legend()
    plt.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"{dataset}_seasonality_forecast.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\nForecast comparison plot saved: {path}")


# =============================================================================
# EXERCISE (optional, for students)
# =============================================================================

def exercise_log_then_seasonal():
    """EXERCISE (advanced): The airline seasonality is MULTIPLICATIVE - its
    swings grow as the level grows. Additive adjustment only partly fixes this.

    Try: take log(values) FIRST (which turns multiplicative seasonality into
    additive), then run the WITH-seasonality pipeline on the logged series, and
    finally np.exp() the predictions back. Does the improvement increase versus
    plain additive adjustment? Compare RMSE in real units.
    Hint: re-use run(...) but pass np.log(values), and exponentiate y_true_real
    and pred_real before computing RMSE.
    """
    raise NotImplementedError("Complete the log-then-seasonal exercise")


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
    chosen = [k for k in ("mlp", "cnn", "lstm") if k in args]
    kinds = tuple(chosen) if chosen else ("mlp", "cnn", "lstm")
    compare(dataset, kinds=kinds)


if __name__ == "__main__":
    main()
