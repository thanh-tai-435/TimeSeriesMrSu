D:\HOCTAP\HK3\TimeSeries\A.M>python dl_time_series_eda.py airline
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097322.495408    2788 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097324.698415    2788 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

======================================================================
EXPLORATORY DATA ANALYSIS: 'airline'
======================================================================
Loaded 'airline': 144 observations (min=104.0, max=622.0 passengers)

--- Numeric summary: 'airline' ---
Observations      : 144
Missing values    : 0
Mean / Std        : 280.30 / 119.97 passengers
Min / Median / Max: 104.00 / 265.50 / 622.00
Range             : 518.00 passengers
2nd-half minus 1st-half mean: +194.79 passengers (large magnitude hints at a trend)

ADF test on 'airline' (original):
    ADF statistic : 0.8154
    p-value       : 0.9919
    Conclusion    : NON-stationary (fail to reject H0)

ADF test on 'airline' (after 1st differencing):
    ADF statistic : -2.8293
    p-value       : 0.0542
    Conclusion    : NON-stationary (fail to reject H0)

Figures saved:
    eda_outputs\airline_01_series.png
    eda_outputs\airline_02_distribution.png
    eda_outputs\airline_03_rolling.png
    eda_outputs\airline_04_decomposition.png
    eda_outputs\airline_05_acf_pacf.png

Now look at the plots and answer: Does this series have a trend? A seasonal cycle? Is it stationary? How many lags look informative?


======================================================================
EXPLORATORY DATA ANALYSIS: 'temps'
======================================================================
Loaded 'temps': 3650 observations (min=0.0, max=26.3 degrees C)

--- Numeric summary: 'temps' ---
Observations      : 3650
Missing values    : 0
Mean / Std        : 11.18 / 4.07 degrees C
Min / Median / Max: 0.00 / 11.00 / 26.30
Range             : 26.30 degrees C
2nd-half minus 1st-half mean: +0.27 degrees C (large magnitude hints at a trend)

ADF test on 'temps' (original):
    ADF statistic : -4.4448
    p-value       : 0.0002
    Conclusion    : STATIONARY (reject H0)

ADF test on 'temps' (after 1st differencing):
    ADF statistic : -18.0282
    p-value       : 0.0000
    Conclusion    : STATIONARY (reject H0)

Figures saved:
    eda_outputs\temps_01_series.png
    eda_outputs\temps_02_distribution.png
    eda_outputs\temps_03_rolling.png
    eda_outputs\temps_04_decomposition.png
    eda_outputs\temps_05_acf_pacf.png

Now look at the plots and answer: Does this series have a trend? A seasonal cycle? Is it stationary? How many lags look informative?

D:\HOCTAP\HK3\TimeSeries\A.M>python dl_time_series_practice.py data
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097393.633110   10844 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097395.949712   10844 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

======================================================================
SECTION 1: From a raw series to supervised (X, y) samples
======================================================================
Raw series: [10, 20, 30, 40, 50, 60, 70]

One-step windowing (n_steps=3):
    [10 20 30]  ->  40
    [20 30 40]  ->  50
    [30 40 50]  ->  60
    [40 50 60]  ->  70

Multi-step windowing (in=3, out=2):
    [10 20 30]  ->  [40 50]
    [20 30 40]  ->  [50 60]
    [30 40 50]  ->  [60 70]

D:\HOCTAP\HK3\TimeSeries\A.M>python dl_time_series_practice.py mlp
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097428.914972   14640 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097431.189879   14640 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

======================================================================
SECTION 2: MLP - univariate one-step forecasting
======================================================================
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
MLP test RMSE: 1.452
Example  -> actual: 62.71   predicted: 64.35

======================================================================
SECTION 3: CNN - univariate one-step forecasting
======================================================================
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
CNN test RMSE: 1.363
Example  -> actual: 62.71   predicted: 63.21

======================================================================
SECTION 4: LSTM - three variants on the same data
======================================================================
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
Vanilla LSTM         test RMSE: 1.284
Stacked LSTM         test RMSE: 1.366
WARNING:tensorflow:5 out of the last 7 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x000002A5DB7B2CA0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
WARNING:tensorflow:6 out of the last 9 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x000002A5DB7B2CA0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
Bidirectional LSTM   test RMSE: 2.353

D:\HOCTAP\HK3\TimeSeries\A.M>python dl_time_series_practice.py compare
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097572.541125   25880 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097574.946955   25880 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

======================================================================
SECTION 5: MLP vs CNN vs LSTM on identical data
======================================================================
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
WARNING:tensorflow:5 out of the last 7 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x00000170DBE94CC0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
WARNING:tensorflow:6 out of the last 9 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x00000170DBE94CC0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
Model      Test RMSE
--------------------
MLP            1.300
LSTM           1.316
CNN            1.531

(Lower RMSE is better. Re-run with a different SEED to see variance.)


D:\HOCTAP\HK3\TimeSeries\A.M>python dl_time_series_real_data.py airline compare
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097618.675349   23732 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097620.997018   23732 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

======================================================================
REAL DATA: MLP vs CNN vs LSTM on 'airline'
======================================================================
Loaded 'airline': 144 observations (min=104.0, max=622.0 passengers)
Naive persistence baseline RMSE: 53.355 passengers

WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
MLP      RMSE:   21.300 passengers   MAE:   18.046 passengers
CNN      RMSE:   19.677 passengers   MAE:   15.077 passengers
LSTM     RMSE:   39.541 passengers   MAE:   32.897 passengers

----------------------------------
Model          RMSE (passengers)
----------------------------------
CNN                       19.677
MLP                       21.300
LSTM                      39.541
Naive                     53.355

(Lower is better. Models below the Naive row are earning their keep.)


======================================================================
REAL DATA: MLP vs CNN vs LSTM on 'temps'
======================================================================
Loaded 'temps': 3650 observations (min=0.0, max=26.3 degrees C)
Naive persistence baseline RMSE: 2.481 degrees C

WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
MLP      RMSE:    2.610 degrees C    MAE:    2.057 degrees C
CNN      RMSE:    2.711 degrees C    MAE:    2.148 degrees C
LSTM     RMSE:    2.207 degrees C    MAE:    1.758 degrees C

----------------------------------
Model           RMSE (degrees C)
----------------------------------
LSTM                       2.207
Naive                      2.481
MLP                        2.610
CNN                        2.711

(Lower is better. Models below the Naive row are earning their keep.)



D:\HOCTAP\HK3\TimeSeries\A.M>python dl_time_series_practice.py ex1
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097842.251350    4360 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097844.577706    4360 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

======================================================================
EXERCISE 1: Multi-step MLP (forecast next 5 values at once)
======================================================================
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
Multi-step MLP  n_steps_in=20  n_steps_out=5
Test RMSE (flattened across all 5 horizons): 1.243
X shape: (301, 20)  y shape: (301, 5)


D:\HOCTAP\HK3\TimeSeries\A.M>python dl_time_series_practice.py ex2
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097871.761815   15104 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097874.067862   15104 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

======================================================================
EXERCISE 2: Multivariate LSTM (2 input features)
======================================================================
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
X shape: (305, 20, 2)  (samples, n_steps, n_features)
Multivariate LSTM  Test RMSE: 2.263


D:\HOCTAP\HK3\TimeSeries\A.M>python dl_time_series_practice.py ex3
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097909.105131   19632 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097911.386947   19632 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

======================================================================
EXERCISE 3: CNN window-size study (synthetic series, cycle=25)
======================================================================

 n_steps   Test RMSE  Note
------------------------------------------------
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
       5       4.554   <- window < 1 cycle
      10       2.386   <- window < 1 cycle
WARNING:tensorflow:5 out of the last 7 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x0000022271DF8540> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
WARNING:tensorflow:6 out of the last 9 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x0000022271DF8540> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
      20       1.405   <- window < 1 cycle
      40       1.441   <- window >= 1 cycle (25)

Best window: n_steps=20  RMSE=1.405
Window of ~25 (one full cycle) or larger gives the model enough context
to observe one complete seasonal pattern and generalise from it.


D:\HOCTAP\HK3\TimeSeries\A.M>python dl_time_series_real_data.py temps window
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097969.216633   13504 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781097971.713648   13504 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

======================================================================
EXERCISE 4.4: Window-size study on 'temps' dataset (CNN)
======================================================================
Loaded 'temps': 3650 observations (min=0.0, max=26.3 degrees C)
Naive persistence baseline RMSE: 2.481 degrees C

 n_steps      RMSE    vs Naive    Beat?
------------------------------------------
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
       7     2.661  +   0.180       NO
      14     2.432    -0.049      YES
      30     2.462    -0.019      YES
      90     3.412  +   0.931       NO

Best window: n_steps=14  RMSE=2.432
Trade-off: longer window -> richer context but fewer training samples.

D:\HOCTAP\HK3\TimeSeries\A.M>python dl_time_series_seasonality.py airline
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781098293.130228   27308 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1781098295.799455   27308 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

======================================================================
SEASONALITY EXPERIMENT: with vs without, on 'airline'
======================================================================
Loaded 'airline': 144 observations (min=104.0, max=622.0 passengers)
Seasonal period: 12   |   Window n_steps: 6 (deliberately shorter than the period)
Naive baseline RMSE: 53.355 passengers

WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
WARNING:tensorflow:5 out of the last 5 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x000001D10BCD5F80> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
WARNING:tensorflow:6 out of the last 6 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x000001D10E04E020> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
Model        Without        With   Improvement
----------------------------------------------
MLP           27.463      32.345        -17.8%
CNN           46.746      35.270         24.6%
LSTM          50.076      38.901         22.3%

Units: passengers. 'Improvement' = % drop in RMSE from removing seasonality before modeling (higher is better).

Window-length sweep (LSTM, period=12):
 n_steps   Without      With   Improvement
------------------------------------------
       3     51.85     39.89        23.1%  <- window < period
       6     43.10     41.71         3.2%  <- window < period
      12     32.21     39.55       -22.8%
      24     29.66     41.62       -40.3%
Loaded 'airline': 144 observations (min=104.0, max=622.0 passengers)

Forecast comparison plot saved: seasonality_outputs\airline_seasonality_forecast.png


Dataset: temps (3650 ngày, 0–26.3°C, baseline RMSE = 2.481°C)

Model	Không xử lý seasonality	Có xử lý	Cải thiện
MLP	2.494	3.010	-20.7% (tệ hơn)
CNN	3.245	2.843	+12.4% (tốt hơn)
LSTM	2.218	2.317	-4.5% (tệ hơn)
Kết luận chính:

LSTM không cần seasonal adjustment — tự học được mùa vụ, RMSE 2.218 là tốt nhất, thậm chí tốt hơn baseline
CNN được lợi khi tách seasonality ra (+12.4%)
MLP bị hại khi tách — mô hình đơn giản không tái hợp tốt
Window sweep cho thấy dù window ngắn (7–90 ngày) so với period 365, LSTM vẫn không cần seasonal adjustment
Plot so sánh đã lưu tại: A.M\seasonality_outputs\temps_seasonality_forecast.png