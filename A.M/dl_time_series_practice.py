"""
=============================================================================
 Deep Learning for Time Series Forecasting - Teaching & Practice Module
=============================================================================
A self-contained, runnable script for students to learn and practice the three
core deep learning architectures for time series forecasting:

        MLP  (Multilayer Perceptron)
        CNN  (1D Convolutional Neural Network)
        LSTM (Long Short-Term Memory recurrent network)

Inspired by the structure of Jason Brownlee's "Deep Learning for Time Series
Forecasting". All code here is original and written for teaching: it is meant
to be read, modified, and experimented with, not just executed.

HOW TO USE
----------
1. Read each section top to bottom. Every model function is short on purpose.
2. Run the whole file:            python dl_time_series_practice.py
3. Run one section at a time:     python dl_time_series_practice.py mlp
                                  python dl_time_series_practice.py cnn
                                  python dl_time_series_practice.py lstm
                                  python dl_time_series_practice.py compare
4. Do the EXERCISES at the bottom (search for "EXERCISE").

KEY IDEA (the foundation of everything below)
---------------------------------------------
A time series is just a list of numbers ordered in time:
        [10, 20, 30, 40, 50, 60, 70, ...]
Neural networks need (input -> output) pairs. We create them with a SLIDING
WINDOW: use the last `n_steps` values to predict the next one.

        X (input)        y (target)
        [10, 20, 30]  -> 40
        [20, 30, 40]  -> 50
        [30, 40, 50]  -> 60
        ...

Once the series is in this supervised form, MLP / CNN / LSTM differ only in
how they READ the window. The data prep is shared; the architecture changes.

Requirements: tensorflow (or tensorflow-cpu), numpy, scikit-learn
=============================================================================
"""

import os
import sys

# Keep TensorFlow's startup logging quiet so students see our output clearly.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Flatten, Conv1D, MaxPooling1D,
    LSTM, Bidirectional, TimeDistributed, Input,
)
from sklearn.metrics import mean_squared_error

# Reproducibility: fix the random seeds so students get the same numbers we do.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# SECTION 1 - PREPARING DATA (turning a series into supervised samples)
# =============================================================================

def split_sequence(sequence, n_steps):
    """Univariate, one-step-ahead.

    Slide a window of length `n_steps` across the series. Each window is one
    input sample X; the value immediately after the window is the target y.

    Returns
        X: array of shape (n_samples, n_steps)
        y: array of shape (n_samples,)
    """
    X, y = [], []
    for i in range(len(sequence)):
        end = i + n_steps              # end index of the input window
        if end > len(sequence) - 1:    # stop when there is no next value to predict
            break
        X.append(sequence[i:end])
        y.append(sequence[end])
    return np.array(X), np.array(y)


def split_sequence_multistep(sequence, n_steps_in, n_steps_out):
    """Univariate, multi-step-ahead.

    Input is `n_steps_in` past values; target is the next `n_steps_out` values.
    Useful when you must forecast a whole horizon at once (e.g. next 7 days).
    """
    X, y = [], []
    for i in range(len(sequence)):
        in_end = i + n_steps_in
        out_end = in_end + n_steps_out
        if out_end > len(sequence):
            break
        X.append(sequence[i:in_end])
        y.append(sequence[in_end:out_end])
    return np.array(X), np.array(y)


def split_sequences_multivariate(data, n_steps):
    """Multivariate, one-step-ahead.

    `data` is a 2D array with several parallel series in columns; the LAST
    column is the target. Each sample is a window over ALL input columns, and
    the target is the value of the last column at the end of the window.

        data columns: [feature_1, feature_2, target]
    """
    X, y = [], []
    for i in range(len(data)):
        end = i + n_steps
        if end > len(data):
            break
        X.append(data[i:end, :-1])     # all input features in the window
        y.append(data[end - 1, -1])    # target at the last time step of the window
    return np.array(X), np.array(y)


def make_series(n=400, kind="sine"):
    """Generate a synthetic but realistic-looking series for experiments.

    kind="sine"  -> seasonal signal + slow trend + noise (good for all models)
    kind="ramp"  -> simple increasing sequence (good for sanity checks)
    """
    t = np.arange(n)
    if kind == "ramp":
        return (t * 1.0).astype("float32")
    seasonal = 10.0 * np.sin(2 * np.pi * t / 25.0)   # repeating cycle
    trend = 0.05 * t                                  # gentle upward drift
    noise = np.random.normal(0, 1.0, size=n)          # measurement noise
    return (50 + seasonal + trend + noise).astype("float32")


def train_test_split_series(X, y, test_frac=0.2):
    """Chronological split: train on the past, test on the most recent tail.

    NEVER shuffle time series before splitting - that leaks the future into
    the training set. We always keep time order.
    """
    n_test = max(1, int(len(X) * test_frac))
    return X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]


def rmse(y_true, y_pred):
    """Root Mean Squared Error - same units as the series, easy to interpret."""
    return float(np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred))))


def demo_windowing():
    """Show the sliding-window transformation on a tiny, readable example."""
    print_header("SECTION 1: From a raw series to supervised (X, y) samples")
    raw = [10, 20, 30, 40, 50, 60, 70]
    print(f"Raw series: {raw}\n")

    X, y = split_sequence(raw, n_steps=3)
    print("One-step windowing (n_steps=3):")
    for xi, yi in zip(X, y):
        print(f"    {xi}  ->  {yi}")

    Xm, ym = split_sequence_multistep(raw, n_steps_in=3, n_steps_out=2)
    print("\nMulti-step windowing (in=3, out=2):")
    for xi, yi in zip(Xm, ym):
        print(f"    {xi}  ->  {yi}")
    print()


# =============================================================================
# SECTION 2 - MLP (Multilayer Perceptron)
# =============================================================================
# The MLP treats the window as a flat vector of `n_steps` numbers. It has no
# notion of order inside the window - it just learns a function from
# n_steps inputs to the output. Simplest baseline; surprisingly strong.

def build_mlp(n_steps, n_outputs=1):
    model = Sequential([
        Input(shape=(n_steps,)),       # input is a flat vector of length n_steps
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(n_outputs),              # linear output for regression
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def demo_mlp(series=None, n_steps=20, epochs=120):
    print_header("SECTION 2: MLP - univariate one-step forecasting")
    series = make_series() if series is None else series

    X, y = split_sequence(series, n_steps)
    X_tr, X_te, y_tr, y_te = train_test_split_series(X, y)

    model = build_mlp(n_steps)
    model.fit(X_tr, y_tr, epochs=epochs, verbose=0)

    pred = model.predict(X_te, verbose=0).ravel()
    score = rmse(y_te, pred)
    print(f"MLP test RMSE: {score:.3f}")
    print(f"Example  -> actual: {y_te[0]:.2f}   predicted: {pred[0]:.2f}")
    return score


# =============================================================================
# SECTION 3 - CNN (1D Convolutional Neural Network)
# =============================================================================
# A 1D CNN slides small FILTERS along the window to detect local shapes
# (rises, dips, spikes). It respects local order and is fast to train.
# Input must be reshaped to 3D: (samples, timesteps, features).

def build_cnn(n_steps, n_features=1, n_outputs=1):
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),     # downsample, keep the strongest signals
        Flatten(),
        Dense(32, activation="relu"),
        Dense(n_outputs),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def demo_cnn(series=None, n_steps=20, epochs=120):
    print_header("SECTION 3: CNN - univariate one-step forecasting")
    series = make_series() if series is None else series

    X, y = split_sequence(series, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))   # CNN needs a feature dimension
    X_tr, X_te, y_tr, y_te = train_test_split_series(X, y)

    model = build_cnn(n_steps)
    model.fit(X_tr, y_tr, epochs=epochs, verbose=0)

    pred = model.predict(X_te, verbose=0).ravel()
    score = rmse(y_te, pred)
    print(f"CNN test RMSE: {score:.3f}")
    print(f"Example  -> actual: {y_te[0]:.2f}   predicted: {pred[0]:.2f}")
    return score


# =============================================================================
# SECTION 4 - LSTM (Long Short-Term Memory)
# =============================================================================
# An LSTM reads the window one step at a time, keeping a memory state. It is
# built for sequences and can capture longer dependencies. Like the CNN, its
# input is 3D: (samples, timesteps, features). Three common variants below.

def build_lstm_vanilla(n_steps, n_features=1, n_outputs=1):
    """Single LSTM layer - the standard starting point."""
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(50, activation="relu"),
        Dense(n_outputs),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_lstm_stacked(n_steps, n_features=1, n_outputs=1):
    """Two stacked LSTM layers - more capacity for complex patterns.
    The first layer must use return_sequences=True so it passes a sequence
    (not just its final state) up to the next LSTM layer."""
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(50, activation="relu", return_sequences=True),
        LSTM(50, activation="relu"),
        Dense(n_outputs),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_lstm_bidirectional(n_steps, n_features=1, n_outputs=1):
    """Reads the window forwards AND backwards, then combines both views."""
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        Bidirectional(LSTM(50, activation="relu")),
        Dense(n_outputs),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def demo_lstm(series=None, n_steps=20, epochs=120):
    print_header("SECTION 4: LSTM - three variants on the same data")
    series = make_series() if series is None else series

    X, y = split_sequence(series, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_tr, X_te, y_tr, y_te = train_test_split_series(X, y)

    builders = {
        "Vanilla LSTM":       build_lstm_vanilla,
        "Stacked LSTM":       build_lstm_stacked,
        "Bidirectional LSTM": build_lstm_bidirectional,
    }
    scores = {}
    for name, builder in builders.items():
        model = builder(n_steps)
        model.fit(X_tr, y_tr, epochs=epochs, verbose=0)
        pred = model.predict(X_te, verbose=0).ravel()
        scores[name] = rmse(y_te, pred)
        print(f"{name:<20} test RMSE: {scores[name]:.3f}")
    return scores


# =============================================================================
# SECTION 5 - HEAD-TO-HEAD COMPARISON
# =============================================================================
# Train MLP, CNN, and a vanilla LSTM on the SAME data and the SAME split so
# students can compare them fairly. Results vary with seed, data, and
# hyperparameters - the goal is to build intuition, not to crown a winner.

def demo_compare(n_steps=20, epochs=120):
    print_header("SECTION 5: MLP vs CNN vs LSTM on identical data")
    series = make_series()

    # Shared 2D windows for the MLP.
    X2d, y = split_sequence(series, n_steps)
    X3d = X2d.reshape((X2d.shape[0], X2d.shape[1], 1))  # 3D for CNN/LSTM

    Xtr2d, Xte2d, ytr, yte = train_test_split_series(X2d, y)
    Xtr3d, Xte3d, _,   _   = train_test_split_series(X3d, y)

    results = {}

    m = build_mlp(n_steps);            m.fit(Xtr2d, ytr, epochs=epochs, verbose=0)
    results["MLP"] = rmse(yte, m.predict(Xte2d, verbose=0).ravel())

    c = build_cnn(n_steps);            c.fit(Xtr3d, ytr, epochs=epochs, verbose=0)
    results["CNN"] = rmse(yte, c.predict(Xte3d, verbose=0).ravel())

    l = build_lstm_vanilla(n_steps);   l.fit(Xtr3d, ytr, epochs=epochs, verbose=0)
    results["LSTM"] = rmse(yte, l.predict(Xte3d, verbose=0).ravel())

    print(f"{'Model':<8}{'Test RMSE':>12}")
    print("-" * 20)
    for name, score in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"{name:<8}{score:>12.3f}")
    print("\n(Lower RMSE is better. Re-run with a different SEED to see variance.)")
    return results


# =============================================================================
# SECTION 6 - EXERCISES FOR STUDENTS
# =============================================================================
# Remove the `raise NotImplementedError` lines and fill in your own code.
# Solutions follow the same patterns as the demos above.

def exercise_1_multistep_mlp():
    """EXERCISE 1 (easy): Forecast the NEXT 5 values at once with an MLP.

    Steps:
      1. series = make_series()
      2. Use split_sequence_multistep(series, n_steps_in=20, n_steps_out=5)
      3. Build an MLP with build_mlp(n_steps=20, n_outputs=5)
      4. Train, predict on the test tail, and report RMSE.
    Hint: n_outputs must equal n_steps_out.
    """
    print_header("EXERCISE 1: Multi-step MLP (forecast next 5 values at once)")
    series = make_series()
    n_steps_in, n_steps_out = 20, 5

    X, y = split_sequence_multistep(series, n_steps_in, n_steps_out)
    X_tr, X_te, y_tr, y_te = train_test_split_series(X, y)

    model = build_mlp(n_steps_in, n_outputs=n_steps_out)
    model.fit(X_tr, y_tr, epochs=150, verbose=0)

    pred = model.predict(X_te, verbose=0)
    score = rmse(y_te.ravel(), pred.ravel())
    print(f"Multi-step MLP  n_steps_in={n_steps_in}  n_steps_out={n_steps_out}")
    print(f"Test RMSE (flattened across all 5 horizons): {score:.3f}")
    print(f"X shape: {X_tr.shape}  y shape: {y_tr.shape}")
    return score


def exercise_2_multivariate_lstm():
    """EXERCISE 2 (medium): Multivariate input.

    Build a 2-feature series where the target depends on both inputs, e.g.
        f1 = make_series(); f2 = make_series(kind='ramp')
        target = f1 + f2
        data = np.column_stack([f1, f2, target])
    Then use split_sequences_multivariate(data, n_steps=20), reshape is NOT
    needed (it is already 3D), and train a vanilla LSTM with
    build_lstm_vanilla(n_steps=20, n_features=2). Report RMSE.
    """
    print_header("EXERCISE 2: Multivariate LSTM (2 input features)")
    n_steps = 20
    # Feature 1: seasonal + noise; Feature 2: slow ramp (trend)
    f1 = make_series(400, kind="sine")
    f2 = make_series(400, kind="ramp")
    # Target is a weighted combination of both features
    target = f1 + 0.1 * f2 + np.random.normal(0, 0.5, size=400).astype("float32")

    data = np.column_stack([f1, f2, target])   # shape (400, 3); last col = target
    X, y = split_sequences_multivariate(data, n_steps)
    # X already 3D: (n_samples, n_steps, n_features=2) -- no reshape needed
    X_tr, X_te, y_tr, y_te = train_test_split_series(X, y)

    model = build_lstm_vanilla(n_steps, n_features=2)
    model.fit(X_tr, y_tr, epochs=150, verbose=0)

    pred = model.predict(X_te, verbose=0).ravel()
    score = rmse(y_te, pred)
    print(f"X shape: {X_tr.shape}  (samples, n_steps, n_features)")
    print(f"Multivariate LSTM  Test RMSE: {score:.3f}")
    return score


def exercise_3_tune_window():
    """EXERCISE 3 (open): How does the window size affect accuracy?

    Loop n_steps over [5, 10, 20, 40], retrain a CNN for each, collect the
    test RMSE, and print a small table. Which window works best for the
    seasonal series (cycle length is 25)? Why might that be?
    """
    print_header("EXERCISE 3: CNN window-size study (synthetic series, cycle=25)")
    series = make_series()
    windows = [5, 10, 20, 40]

    print(f"\n{'n_steps':>8}  {'Test RMSE':>10}  Note")
    print("-" * 48)
    results = {}
    for n_steps in windows:
        X, y = split_sequence(series, n_steps)
        X = X.reshape((X.shape[0], n_steps, 1))
        X_tr, X_te, y_tr, y_te = train_test_split_series(X, y)
        model = build_cnn(n_steps)
        model.fit(X_tr, y_tr, epochs=150, verbose=0)
        pred = model.predict(X_te, verbose=0).ravel()
        score = rmse(y_te, pred)
        results[n_steps] = score
        note = " <- window >= 1 cycle (25)" if n_steps >= 25 else " <- window < 1 cycle"
        print(f"{n_steps:>8}  {score:>10.3f}  {note}")

    best = min(results, key=results.get)
    print(f"\nBest window: n_steps={best}  RMSE={results[best]:.3f}")
    print("Window of ~25 (one full cycle) or larger gives the model enough context")
    print("to observe one complete seasonal pattern and generalise from it.")
    return results


# =============================================================================
# Plumbing
# =============================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if arg in ("all", "data"):
        demo_windowing()
    if arg in ("all", "mlp"):
        demo_mlp()
    if arg in ("all", "cnn"):
        demo_cnn()
    if arg in ("all", "lstm"):
        demo_lstm()
    if arg in ("all", "compare"):
        demo_compare()
    if arg in ("ex1",):
        exercise_1_multistep_mlp()
    if arg in ("ex2",):
        exercise_2_multivariate_lstm()
    if arg in ("ex3",):
        exercise_3_tune_window()

    if arg not in ("all", "data", "mlp", "cnn", "lstm", "compare", "ex1", "ex2", "ex3"):
        print(__doc__)
        print("Usage: python dl_time_series_practice.py "
              "[all|data|mlp|cnn|lstm|compare|ex1|ex2|ex3]")


if __name__ == "__main__":
    main()
