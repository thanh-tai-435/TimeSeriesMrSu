"""
=============================================================================
 Deep Learning for Time Series Forecasting - EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================
Companion to `dl_time_series_practice.py` and `dl_time_series_real_data.py`.

BEFORE you fit any model, you must UNDERSTAND the data. EDA for a time series
is not the same as EDA for an ordinary table: order matters, and the questions
are specific:

    1. What does the series LOOK like over time?         -> line plot
    2. How are the values DISTRIBUTED?                    -> histogram + boxplot
    3. Is it STATIONARY (stable mean/variance over time)? -> rolling stats + ADF
    4. Does it have TREND and SEASONALITY?                -> seasonal decomposition
    5. How does each value relate to its PAST values?     -> ACF / PACF

The answers directly inform your modeling choices: how long a window to use,
whether to difference the series, and whether a model can realistically beat a
naive baseline. This module computes all five for the real datasets and saves
the figures to ./eda_outputs/.

Run it:
    python dl_time_series_eda.py                 # airline, full EDA
    python dl_time_series_eda.py temps           # temperatures, full EDA
    python dl_time_series_eda.py airline adf     # just the stationarity test

Requirements: matplotlib, statsmodels, pandas, numpy  (plus the companion files)
=============================================================================
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save figures to files; no interactive window needed
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Reuse the dataset registry and loader so EDA and modeling see identical data.
from dl_time_series_real_data import DATASETS, load_series

OUTDIR = "eda_outputs"

# Each dataset has a natural seasonal period: 12 months for airline data,
# ~365 days for daily temperatures. Decomposition and ACF need this.
SEASONAL_PERIOD = {"airline": 12, "temps": 365}


# =============================================================================
# 1 - NUMERIC SUMMARY
# =============================================================================

def summarize(values, meta, name):
    """Print the basic shape, central tendency, spread, and data-quality checks."""
    print(f"\n--- Numeric summary: '{name}' ---")
    s = pd.Series(values)
    print(f"Observations      : {len(s)}")
    print(f"Missing values    : {int(s.isna().sum())}")
    print(f"Mean / Std        : {s.mean():.2f} / {s.std():.2f} {meta['unit']}")
    print(f"Min / Median / Max: {s.min():.2f} / {s.median():.2f} / {s.max():.2f}")
    print(f"Range             : {s.max() - s.min():.2f} {meta['unit']}")
    # A quick, order-aware check: does the second half sit higher than the first?
    # A large gap is an informal hint of TREND.
    half = len(s) // 2
    drift = s[half:].mean() - s[:half].mean()
    print(f"2nd-half minus 1st-half mean: {drift:+.2f} {meta['unit']} "
          f"(large magnitude hints at a trend)")


# =============================================================================
# 2 - LINE PLOT (the single most important time series plot)
# =============================================================================

def plot_series(values, meta, name):
    plt.figure(figsize=(11, 4))
    plt.plot(values, linewidth=0.9)
    plt.title(f"{name}: value over time")
    plt.xlabel("time step")
    plt.ylabel(meta["unit"])
    plt.tight_layout()
    return _save(f"{name}_01_series.png")


# =============================================================================
# 3 - DISTRIBUTION (histogram + boxplot)
# =============================================================================

def plot_distribution(values, meta, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(values, bins=30, edgecolor="black", alpha=0.8)
    ax1.set_title(f"{name}: histogram")
    ax1.set_xlabel(meta["unit"]); ax1.set_ylabel("frequency")
    ax2.boxplot(values, vert=True)
    ax2.set_title(f"{name}: boxplot (outliers shown as points)")
    ax2.set_ylabel(meta["unit"])
    plt.tight_layout()
    return _save(f"{name}_02_distribution.png")


# =============================================================================
# 4 - STATIONARITY: rolling statistics + Augmented Dickey-Fuller test
# =============================================================================
# A series is (weakly) STATIONARY if its mean and variance do not change over
# time. Most classical methods assume stationarity; neural nets are more
# forgiving but still benefit from it. We check two ways: visually (rolling
# mean/std should be roughly flat) and formally (the ADF test).

def plot_rolling_stats(values, meta, name, window=None):
    if window is None:
        window = SEASONAL_PERIOD.get(name, 12)
    s = pd.Series(values)
    plt.figure(figsize=(11, 4))
    plt.plot(s, label="original", linewidth=0.8, alpha=0.6)
    plt.plot(s.rolling(window).mean(), label=f"rolling mean (w={window})", linewidth=1.6)
    plt.plot(s.rolling(window).std(), label=f"rolling std (w={window})", linewidth=1.6)
    plt.title(f"{name}: rolling mean & std (flat lines => more stationary)")
    plt.xlabel("time step"); plt.ylabel(meta["unit"])
    plt.legend()
    plt.tight_layout()
    return _save(f"{name}_03_rolling.png")


def adf_test(values, name, label="original"):
    """Augmented Dickey-Fuller test.

    Null hypothesis H0: the series has a unit root (it is NON-stationary).
    If p-value < 0.05 we reject H0 and treat the series as stationary.
    """
    result = adfuller(np.asarray(values, dtype="float64"))
    stat, pvalue = result[0], result[1]
    verdict = "STATIONARY (reject H0)" if pvalue < 0.05 else "NON-stationary (fail to reject H0)"
    print(f"\nADF test on '{name}' ({label}):")
    print(f"    ADF statistic : {stat:.4f}")
    print(f"    p-value       : {pvalue:.4f}")
    print(f"    Conclusion    : {verdict}")
    return pvalue


def adf_with_differencing(values, name):
    """Run ADF on the raw series, then on the once-differenced series.

    DIFFERENCING (y[t] - y[t-1]) removes trend and often turns a non-stationary
    series into a stationary one. This shows students the standard fix.
    """
    adf_test(values, name, "original")
    differenced = np.diff(values)
    adf_test(differenced, name, "after 1st differencing")


# =============================================================================
# 5 - SEASONAL DECOMPOSITION (trend + seasonal + residual)
# =============================================================================

def plot_decomposition(values, meta, name):
    period = SEASONAL_PERIOD.get(name, 12)
    # Need at least two full cycles for a meaningful decomposition.
    if len(values) < 2 * period:
        print(f"[skip] '{name}' too short to decompose at period {period}.")
        return None
    result = seasonal_decompose(
        pd.Series(values), model="additive", period=period
    )
    fig = result.plot()
    fig.set_size_inches(11, 7)
    fig.suptitle(f"{name}: additive decomposition (period={period})", y=1.01)
    plt.tight_layout()
    return _save(f"{name}_04_decomposition.png")


# =============================================================================
# 6 - AUTOCORRELATION (ACF) and PARTIAL AUTOCORRELATION (PACF)
# =============================================================================
# ACF: how correlated is the series with itself at each lag? Slowly-decaying
#      ACF signals trend; spikes at seasonal lags signal seasonality.
# PACF: the correlation at a lag AFTER removing shorter lags. Helps decide how
#       many past steps actually carry independent information -> a data-driven
#       hint for choosing your window size (n_steps).

def plot_acf_pacf(values, name, lags=40):
    lags = min(lags, len(values) // 2 - 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(values, lags=lags, ax=ax1)
    ax1.set_title(f"{name}: ACF")
    plot_pacf(values, lags=lags, ax=ax2, method="ywm")
    ax2.set_title(f"{name}: PACF")
    plt.tight_layout()
    return _save(f"{name}_05_acf_pacf.png")


# =============================================================================
# Driver
# =============================================================================

def run_full_eda(name):
    print_header(f"EXPLORATORY DATA ANALYSIS: '{name}'")
    values, meta = load_series(name)
    summarize(values, meta, name)

    saved = [
        plot_series(values, meta, name),
        plot_distribution(values, meta, name),
        plot_rolling_stats(values, meta, name),
        plot_decomposition(values, meta, name),
        plot_acf_pacf(values, name),
    ]
    adf_with_differencing(values, name)

    print("\nFigures saved:")
    for f in saved:
        if f:
            print(f"    {f}")
    print("\nNow look at the plots and answer: Does this series have a trend? "
          "A seasonal cycle? Is it stationary? How many lags look informative?")


def _save(filename):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return path


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    args = [a.lower() for a in sys.argv[1:]]
    dataset = next((a for a in args if a in DATASETS), "airline")
    if "adf" in args:
        values, _ = load_series(dataset)
        adf_with_differencing(values, dataset)
    else:
        run_full_eda(dataset)


if __name__ == "__main__":
    main()
