# Fresh50K Global ML Forecasting — Task List cho Codex

## Mục tiêu project

Xây dựng project Python hoàn chỉnh, reproducible cho bài **Time Series Forecasting trên bộ FreshRetailNet-50K** theo hướng:

> **Global Machine Learning Forecasting bằng LightGBM/XGBoost + stockout-aware features**

Project cần tạo đầy đủ:

- Code chạy được từ đầu đến cuối.
- EDA time series.
- ADF/KPSS, ACF/PACF.
- Feature engineering: lag, rolling, calendar, stockout, promotion, weather.
- Train/test split theo thời gian.
- Benchmark: Naive, Seasonal Naive 24h, Seasonal Naive 168h.
- Main model: LightGBM global forecasting.
- Optional model: XGBoost/CatBoost.
- Ablation study.
- Evaluation: RMSE, MAE, WAPE, sMAPE.
- Stockout-period vs non-stockout-period evaluation.
- Bảng CSV và hình PNG cho slide/report.
- README + requirements.txt.

---

# 1. Prompt tổng đưa cho Codex

```text
Tôi cần làm project Time Series Forecasting trên bộ FreshRetailNet-50K theo hướng Global Machine Learning Forecasting.

Hãy xây dựng một project Python hoàn chỉnh, reproducible, gồm:
1. Load và chuẩn hóa dữ liệu FreshRetailNet-50K.
2. EDA time series.
3. Phân tích stationarity bằng ADF/KPSS trên aggregate series và representative series.
4. Vẽ ACF/PACF.
5. Tạo lag features, rolling features, calendar features, stockout-aware features, promotion/weather features.
6. Chia train/validation/test theo thời gian, tuyệt đối không random split.
7. Train benchmark: Naive, Seasonal Naive 24h, Seasonal Naive 168h.
8. Train Linear/Ridge baseline nếu phù hợp.
9. Train LightGBM global forecasting model.
10. Optional: Train XGBoost/CatBoost nếu thư viện có sẵn.
11. Làm ablation study:
   - Lag + Calendar
   - + Promotion
   - + Weather
   - + Stockout features
12. Đánh giá bằng RMSE, MAE, WAPE, sMAPE.
13. Đánh giá riêng stockout-period và non-stockout-period.
14. Xuất bảng metrics ra CSV.
15. Xuất các hình ảnh PNG cho slide/report.
16. Tạo README hướng dẫn chạy code.
17. Code phải chạy lại được từ đầu đến cuối.

Yêu cầu kỹ thuật:
- Dùng Python.
- Ưu tiên pandas/polars, numpy, matplotlib, statsmodels, scikit-learn, lightgbm.
- Không dùng random train/test split.
- Không tạo leakage: rolling features phải shift(1) trước khi rolling.
- Dữ liệu lớn thì hỗ trợ chạy sample trước, sau đó chạy full.
- Lưu outputs vào thư mục outputs/.
- Lưu figures vào outputs/figures/.
- Lưu metrics vào outputs/tables/.
```

---

# 2. Project structure cần tạo

```text
fresh50k_forecasting/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_stationarity_acf_pacf.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_results_analysis.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── stationarity.py
│   ├── features.py
│   ├── split.py
│   ├── benchmarks.py
│   ├── models.py
│   ├── evaluation.py
│   ├── plots.py
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── reports/
│
├── main_eda.py
├── main_train.py
├── main_evaluate.py
├── requirements.txt
└── README.md
```

---

# 3. Phase 1 — Load data và kiểm tra dữ liệu

## Task 1.1 — Load Fresh50K

File cần viết:

```text
src/data_loader.py
```

Việc cần làm:

- Load file CSV/Parquet.
- Parse cột thời gian `dt`.
- Sort theo `store_id`, `product_id`, `dt`.
- Tạo `series_id`.
- Chuẩn hóa tên cột nếu cần.

Logic tạo `series_id`:

```python
series_id = city_id + "_" + store_id + "_" + product_id
```

Nếu không có `city_id`, dùng:

```python
series_id = store_id + "_" + product_id
```

Output:

```text
data/processed/fresh50k_base.parquet
```

Checklist:

- [ ] Load được data.
- [ ] Parse datetime.
- [ ] Sort đúng thứ tự thời gian.
- [ ] Tạo `series_id`.
- [ ] Save parquet để chạy nhanh hơn.

---

## Task 1.2 — Data quality report

Tạo bảng:

```text
outputs/tables/data_quality_summary.csv
```

Bảng gồm:

| Metric | Value |
|---|---:|
| Number of rows | ... |
| Number of series | ... |
| Number of stores | ... |
| Number of products | ... |
| Number of cities | ... |
| Start date | ... |
| End date | ... |
| Frequency | Hourly |
| Missing dt count | ... |
| Duplicate rows | ... |
| Missing sale_amount | ... |
| Stockout rate | ... |
| Average sale_amount | ... |
| Median sale_amount | ... |

Checklist:

- [ ] Count rows.
- [ ] Count unique series.
- [ ] Count stores.
- [ ] Count products.
- [ ] Count missing values.
- [ ] Count duplicate timestamps.
- [ ] Tính stockout rate.
- [ ] Save CSV.

---

# 4. Phase 2 — EDA cho slide/report

## Task 2.1 — Aggregate time series plot

Tạo aggregate sales theo giờ hoặc ngày:

```python
total_sales_by_time = df.groupby("dt")["sale_amount"].sum()
```

Xuất hình:

```text
outputs/figures/aggregate_sales_over_time.png
```

Yêu cầu:

- Line plot.
- Title rõ.
- X-axis là time.
- Y-axis là total sales.
- Có thể thêm vertical line train/test sau này.

Checklist:

- [ ] Plot tổng sales theo thời gian.
- [ ] Save PNG 300 dpi.
- [ ] Dùng matplotlib.

---

## Task 2.2 — Stockout pattern plot

Tạo stockout rate theo thời gian:

```python
stockout_rate_by_time = df.groupby("dt")["stockout_flag"].mean()
```

Nếu cột là `hours_stock_status`, cần convert thành flag:

```python
stockout_flag = 1 nếu stock_status biểu thị hết hàng
```

Xuất:

```text
outputs/figures/stockout_rate_over_time.png
```

Checklist:

- [ ] Tính stockout rate theo thời gian.
- [ ] Plot line chart.
- [ ] Save PNG.

---

## Task 2.3 — Sale distribution

Xuất:

```text
outputs/figures/sale_amount_distribution.png
outputs/figures/log_sale_amount_distribution.png
```

Checklist:

- [ ] Histogram sale_amount.
- [ ] Histogram log1p(sale_amount).
- [ ] Nhận xét skewness nếu có.
- [ ] Save PNG.

---

## Task 2.4 — Seasonality plots

Tạo:

```text
outputs/figures/sales_by_hour_of_day.png
outputs/figures/sales_by_day_of_week.png
```

Logic:

```python
df["hour"] = df["dt"].dt.hour
df["day_of_week"] = df["dt"].dt.dayofweek
```

Checklist:

- [ ] Average sales by hour.
- [ ] Average sales by day of week.
- [ ] Save PNG.

---

## Task 2.5 — Representative series selection

Chọn 3 series để plot:

1. High-volume series.
2. Low-volume/intermittent series.
3. Stockout-heavy series.

Xuất bảng:

```text
outputs/tables/representative_series.csv
```

Bảng gồm:

| series_id | type | total_sales | stockout_rate | n_obs |
|---|---|---:|---:|---:|

Xuất hình:

```text
outputs/figures/representative_series_high_volume.png
outputs/figures/representative_series_intermittent.png
outputs/figures/representative_series_stockout_heavy.png
```

Checklist:

- [ ] Chọn high-volume series.
- [ ] Chọn low-volume series.
- [ ] Chọn stockout-heavy series.
- [ ] Plot từng chuỗi.
- [ ] Save CSV + PNG.

---

# 5. Phase 3 — Stationarity, ADF/KPSS, ACF/PACF

## Task 3.1 — ADF/KPSS trên aggregate series

File cần viết:

```text
src/stationarity.py
```

Function cần có:

```python
run_adf_test(series)
run_kpss_test(series)
run_stationarity_report(series_dict)
```

Chạy trên:

- aggregate hourly sales.
- aggregate daily sales nếu cần.
- 3 representative series.

Xuất bảng:

```text
outputs/tables/stationarity_tests.csv
```

Bảng:

| Series | Transformation | ADF statistic | ADF p-value | KPSS statistic | KPSS p-value | Conclusion |
|---|---|---:|---:|---:|---:|---|

Transformations cần thử:

- original.
- log1p.
- first difference.
- seasonal difference lag 24.

Checklist:

- [ ] Chạy ADF.
- [ ] Chạy KPSS.
- [ ] Có try/except nếu series quá ngắn.
- [ ] Save CSV.
- [ ] Có conclusion tự động: stationary / non-stationary / mixed evidence.

---

## Task 3.2 — ACF/PACF plots

Xuất:

```text
outputs/figures/acf_aggregate_sales.png
outputs/figures/pacf_aggregate_sales.png
```

Với representative series:

```text
outputs/figures/acf_high_volume_series.png
outputs/figures/pacf_high_volume_series.png
```

Yêu cầu:

- Lags ít nhất 48 hoặc 168.
- Nếu hourly data, ưu tiên show lag 24 và 168.
- Save PNG.

Checklist:

- [ ] Plot ACF.
- [ ] Plot PACF.
- [ ] Save PNG.
- [ ] Dùng statsmodels.

---

# 6. Phase 4 — Feature engineering

File chính:

```text
src/features.py
```

---

## Task 4.1 — Calendar features

Tạo các cột:

```text
hour
day_of_week
is_weekend
day_of_month
week_of_year
month
sin_hour
cos_hour
sin_dayofweek
cos_dayofweek
```

Checklist:

- [ ] Tạo hour.
- [ ] Tạo day_of_week.
- [ ] Tạo is_weekend.
- [ ] Tạo cyclic encoding cho hour.
- [ ] Tạo cyclic encoding cho day_of_week.

---

## Task 4.2 — Lag features

Tạo:

```text
sale_lag_1
sale_lag_2
sale_lag_3
sale_lag_6
sale_lag_12
sale_lag_24
sale_lag_48
sale_lag_72
sale_lag_168
```

Quan trọng:

```python
df.groupby("series_id")["sale_amount"].shift(lag)
```

Checklist:

- [ ] Lag theo từng `series_id`.
- [ ] Không được lag global toàn dataset.
- [ ] Không leakage.
- [ ] Save feature list.

---

## Task 4.3 — Rolling features

Tạo:

```text
sale_roll_mean_3
sale_roll_mean_6
sale_roll_mean_12
sale_roll_mean_24
sale_roll_mean_168
sale_roll_std_24
sale_roll_std_168
sale_roll_min_24
sale_roll_max_24
```

Quan trọng: phải `shift(1)` trước khi rolling.

Ví dụ đúng:

```python
shifted = df.groupby("series_id")["sale_amount"].shift(1)
df["sale_roll_mean_24"] = (
    shifted.groupby(df["series_id"])
    .rolling(24)
    .mean()
    .reset_index(level=0, drop=True)
)
```

Checklist:

- [ ] Rolling mean.
- [ ] Rolling std.
- [ ] Rolling min/max.
- [ ] Shift trước rolling.
- [ ] Không leakage.

---

## Task 4.4 — Stockout-aware features

Tạo hoặc chuẩn hóa cột:

```text
stockout_flag
```

Sau đó tạo:

```text
stockout_lag_1
stockout_lag_2
stockout_lag_3
stockout_lag_24
stockout_lag_168
stockout_roll_sum_24
stockout_roll_mean_24
stockout_roll_sum_168
stockout_roll_mean_168
```

Nếu có `hours_stock_status` dạng sequence thì xử lý tùy format. Nếu data đã là long format theo giờ thì tạo trực tiếp.

Checklist:

- [ ] Chuẩn hóa stockout_flag.
- [ ] Tạo stockout lag.
- [ ] Tạo stockout rolling sum/mean.
- [ ] Shift trước rolling.
- [ ] Save feature list.

---

## Task 4.5 — Promotion/weather features

Dùng các cột nếu có:

```text
discount
holiday_flag
activity_flag
precip
temp
humidity
wind
```

Tạo thêm:

```text
discount_lag_1
discount_lag_24
discount_roll_mean_24
temp_lag_1
temp_lag_24
precip_lag_1
```

Checklist:

- [ ] Check cột tồn tại trước khi tạo.
- [ ] Không crash nếu thiếu cột.
- [ ] Tạo lag/rolling cho discount.
- [ ] Tạo lag weather nếu có.

---

## Task 4.6 — Target engineering

Tạo target cho horizon:

```text
target_h1 = sale_amount at t+1
target_h24 = sale_amount at t+24
```

Code:

```python
df["target_h1"] = df.groupby("series_id")["sale_amount"].shift(-1)
df["target_h24"] = df.groupby("series_id")["sale_amount"].shift(-24)
```

Output:

```text
data/processed/fresh50k_features.parquet
```

Checklist:

- [ ] Tạo target_h1.
- [ ] Tạo target_h24.
- [ ] Drop rows thiếu target.
- [ ] Save processed feature table.

---

# 7. Phase 5 — Train/validation/test split

File:

```text
src/split.py
```

Không random split.

## Option A — Percent split theo thời gian

```text
Train: first 70%
Validation: next 15%
Test: final 15%
```

## Option B — Last N days

```text
Test: last 7 days or 14 days
Validation: 7 days before test
Train: all previous data
```

Ưu tiên Option B nếu data đủ dài.

Output:

```text
outputs/tables/split_summary.csv
```

Bảng:

| Split | Start date | End date | Rows | Series |
|---|---|---|---:|---:|

Checklist:

- [ ] Split theo `dt`.
- [ ] Không random.
- [ ] Không overlap.
- [ ] Save split summary.
- [ ] Có function trả về train_df, val_df, test_df.

---

# 8. Phase 6 — Benchmark models

File:

```text
src/benchmarks.py
```

## Task 6.1 — Naive benchmark

Forecast:

```text
y_hat = sale_lag_1
```

Cho h=1.

Với h=24:

```text
y_hat = sale_lag_24
```

Output:

```text
outputs/tables/predictions_naive_h1.csv
outputs/tables/predictions_naive_h24.csv
```

---

## Task 6.2 — Seasonal naive 24h

Forecast:

```text
y_hat = sale_lag_24
```

---

## Task 6.3 — Seasonal naive 168h

Forecast:

```text
y_hat = sale_lag_168
```

Checklist benchmark:

- [ ] Naive.
- [ ] Seasonal Naive 24.
- [ ] Seasonal Naive 168.
- [ ] Không train gì cả.
- [ ] Tính metrics trên test.

---

# 9. Phase 7 — ML models

File:

```text
src/models.py
```

---

## Task 7.1 — Ridge/Linear baseline

Train simple baseline:

```text
Ridge regression
```

Feature set:

```text
lag + calendar
```

Output:

```text
outputs/tables/predictions_ridge_h1.csv
outputs/tables/predictions_ridge_h24.csv
```

Checklist:

- [ ] Handle categorical bằng one-hot hoặc bỏ category.
- [ ] StandardScaler nếu cần.
- [ ] Train h1.
- [ ] Train h24 nếu có.

---

## Task 7.2 — LightGBM main model

Train:

```python
LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

Có early stopping trên validation.

Output:

```text
outputs/models/lightgbm_h1.pkl
outputs/models/lightgbm_h24.pkl
outputs/tables/predictions_lightgbm_h1.csv
outputs/tables/predictions_lightgbm_h24.csv
```

Checklist:

- [ ] Train LightGBM h1.
- [ ] Train LightGBM h24.
- [ ] Early stopping.
- [ ] Save model.
- [ ] Save predictions.
- [ ] Save feature importance.

---

## Task 7.3 — XGBoost optional

Nếu máy chạy được thì thêm:

```text
XGBRegressor
```

Nếu không, bỏ cũng được.

Output:

```text
outputs/models/xgboost_h1.pkl
outputs/tables/predictions_xgboost_h1.csv
```

Checklist:

- [ ] Optional.
- [ ] Không để lỗi nếu thiếu xgboost.
- [ ] Có try/except hoặc config flag.

---

# 10. Phase 8 — Ablation study

Chạy LightGBM với các feature set sau.

## Feature set A — Lag + Calendar

```text
lag features
rolling features
calendar features
```

## Feature set B — A + Promotion

```text
+ discount
+ holiday_flag
+ activity_flag
```

## Feature set C — B + Weather

```text
+ temp
+ precip
+ humidity
+ wind
```

## Feature set D — C + Stockout

```text
+ stockout lag
+ stockout rolling
```

Xuất bảng:

```text
outputs/tables/ablation_results.csv
```

Bảng:

| Horizon | Feature set | RMSE | MAE | WAPE | sMAPE |
|---|---|---:|---:|---:|---:|

Checklist:

- [ ] Train 4 LightGBM models cho h1.
- [ ] Optional train 4 models cho h24.
- [ ] Save metrics.
- [ ] Save best model.
- [ ] Kết luận feature stockout có giúp không.

---

# 11. Phase 9 — Evaluation

File:

```text
src/evaluation.py
```

## Task 9.1 — Metrics functions

Cần function:

```python
rmse(y_true, y_pred)
mae(y_true, y_pred)
wape(y_true, y_pred)
smape(y_true, y_pred)
evaluate_predictions(y_true, y_pred)
```

Công thức WAPE:

```python
WAPE = sum(abs(y - yhat)) / sum(abs(y))
```

sMAPE:

```python
sMAPE = mean(2 * abs(y - yhat) / (abs(y) + abs(yhat)))
```

Checklist:

- [ ] Handle division by zero.
- [ ] Không crash khi y toàn 0.
- [ ] Return dict metrics.

---

## Task 9.2 — Overall model comparison

Xuất:

```text
outputs/tables/model_comparison_h1.csv
outputs/tables/model_comparison_h24.csv
```

Bảng:

| Horizon | Model | RMSE | MAE | WAPE | sMAPE |
|---|---|---:|---:|---:|---:|

Models:

- Naive.
- Seasonal Naive 24.
- Seasonal Naive 168.
- Ridge.
- LightGBM.
- XGBoost optional.

Checklist:

- [ ] Tính metrics cùng test set.
- [ ] Save CSV.
- [ ] Sort theo WAPE hoặc RMSE.

---

## Task 9.3 — Stockout vs non-stockout evaluation

Dựa trên `stockout_flag` tại thời điểm target.

Cần tạo:

```text
target_stockout_flag_h1
target_stockout_flag_h24
```

Hoặc dùng stockout tại timestamp tương ứng target.

Xuất:

```text
outputs/tables/stockout_segment_evaluation.csv
```

Bảng:

| Horizon | Model | Segment | RMSE | MAE | WAPE | sMAPE | N |
|---|---|---|---:|---:|---:|---:|---:|

Segments:

```text
overall
stockout_period
non_stockout_period
```

Checklist:

- [ ] Tính metrics overall.
- [ ] Tính metrics stockout_period.
- [ ] Tính metrics non_stockout_period.
- [ ] Save CSV.

---

## Task 9.4 — Category/store-level evaluation

Optional nhưng đẹp.

Xuất:

```text
outputs/tables/evaluation_by_category.csv
outputs/tables/evaluation_by_store.csv
```

Bảng:

| Group | RMSE | MAE | WAPE | N |
|---|---:|---:|---:|---:|

Checklist:

- [ ] Group by category.
- [ ] Group by store.
- [ ] Tính WAPE.
- [ ] Save CSV.

---

# 12. Phase 10 — Plots cho slide

File:

```text
src/plots.py
```

## Task 10.1 — Model comparison bar chart

Xuất:

```text
outputs/figures/model_comparison_wape_h1.png
outputs/figures/model_comparison_rmse_h1.png
```

Checklist:

- [ ] Bar chart WAPE.
- [ ] Bar chart RMSE.
- [ ] Models trên x-axis.
- [ ] Save PNG.

---

## Task 10.2 — Ablation chart

Xuất:

```text
outputs/figures/ablation_wape_h1.png
```

Checklist:

- [ ] Bar chart feature set vs WAPE.
- [ ] Save PNG.

---

## Task 10.3 — Stockout vs non-stockout chart

Xuất:

```text
outputs/figures/stockout_segment_wape.png
```

Checklist:

- [ ] So sánh LightGBM without stockout vs with stockout.
- [ ] Segment: stockout vs non-stockout.
- [ ] Save PNG.

---

## Task 10.4 — Forecast example plots

Chọn 3 representative series:

```text
high_volume
intermittent
stockout_heavy
```

Xuất:

```text
outputs/figures/forecast_high_volume_series.png
outputs/figures/forecast_intermittent_series.png
outputs/figures/forecast_stockout_heavy_series.png
```

Mỗi hình gồm:

- Actual test.
- LightGBM prediction.
- Seasonal naive prediction.
- Có title rõ.
- Có legend.

Checklist:

- [ ] Plot actual vs forecast.
- [ ] Có benchmark line.
- [ ] Save PNG.

---

## Task 10.5 — Feature importance

Xuất:

```text
outputs/tables/lightgbm_feature_importance.csv
outputs/figures/lightgbm_feature_importance_top20.png
```

Checklist:

- [ ] Extract feature importance.
- [ ] Sort descending.
- [ ] Save CSV.
- [ ] Plot top 20.

---

# 13. Phase 11 — Generate summary text cho report

Tạo file:

```text
outputs/reports/result_summary.md
```

Nội dung tự động lấy từ bảng metrics:

```text
Best model by WAPE:
- Model: ...
- WAPE: ...
- Improvement over seasonal naive: ...%

Ablation result:
- Adding stockout features changed WAPE from ... to ...
- Improvement: ...%

Stockout-period result:
- In stockout periods, LightGBM with stockout features achieved WAPE ...
- Compared with without-stockout model, improvement was ...%
```

Checklist:

- [ ] Đọc CSV metrics.
- [ ] Tính improvement %.
- [ ] Xuất markdown text.
- [ ] Có câu kết luận ngắn.

---

# 14. Phase 12 — README và reproducibility

Cần tạo:

```text
README.md
requirements.txt
```

## README cần có

```text
Project: Fresh50K Global ML Forecasting

1. Environment setup
pip install -r requirements.txt

2. Put raw data in:
data/raw/

3. Run EDA:
python main_eda.py

4. Run training:
python main_train.py --horizon 1
python main_train.py --horizon 24

5. Run evaluation:
python main_evaluate.py

6. Outputs:
outputs/figures/
outputs/tables/
outputs/models/
```

## requirements.txt tối thiểu

```text
pandas
numpy
matplotlib
scikit-learn
statsmodels
lightgbm
pyarrow
joblib
tqdm
```

Optional:

```text
xgboost
catboost
shap
polars
```

Checklist:

- [ ] Có hướng dẫn chạy.
- [ ] Có package requirements.
- [ ] Có mô tả output.
- [ ] Có note về data path.
- [ ] Có seed.

---

# 15. Phase 13 — Main scripts

## main_eda.py

Chạy:

- Load data.
- Data quality.
- EDA plots.
- Stationarity.
- ACF/PACF.

Output:

```text
outputs/tables/data_quality_summary.csv
outputs/tables/stationarity_tests.csv
outputs/figures/*.png
```

---

## main_train.py

Argument:

```bash
python main_train.py --horizon 1 --sample_frac 0.2
python main_train.py --horizon 24 --sample_frac 1.0
```

Chạy:

- Load processed features.
- Split.
- Train benchmarks.
- Train LightGBM.
- Ablation.
- Save predictions/models.

---

## main_evaluate.py

Chạy:

- Load predictions.
- Compute metrics.
- Create comparison tables.
- Create stockout-segment evaluation.
- Plot charts.
- Create result_summary.md.

---

# 16. Các bảng cần có cuối cùng

Đảm bảo tạo đủ:

```text
outputs/tables/data_quality_summary.csv
outputs/tables/representative_series.csv
outputs/tables/stationarity_tests.csv
outputs/tables/split_summary.csv
outputs/tables/model_comparison_h1.csv
outputs/tables/model_comparison_h24.csv
outputs/tables/ablation_results.csv
outputs/tables/stockout_segment_evaluation.csv
outputs/tables/lightgbm_feature_importance.csv
```

---

# 17. Các hình cần có cuối cùng

Đảm bảo tạo đủ:

```text
outputs/figures/aggregate_sales_over_time.png
outputs/figures/stockout_rate_over_time.png
outputs/figures/sale_amount_distribution.png
outputs/figures/log_sale_amount_distribution.png
outputs/figures/sales_by_hour_of_day.png
outputs/figures/sales_by_day_of_week.png
outputs/figures/acf_aggregate_sales.png
outputs/figures/pacf_aggregate_sales.png
outputs/figures/model_comparison_wape_h1.png
outputs/figures/ablation_wape_h1.png
outputs/figures/stockout_segment_wape.png
outputs/figures/forecast_high_volume_series.png
outputs/figures/forecast_intermittent_series.png
outputs/figures/forecast_stockout_heavy_series.png
outputs/figures/lightgbm_feature_importance_top20.png
```

---

# 18. Slide/report mapping

| Slide | Output cần dùng |
|---:|---|
| Dataset overview | `data_quality_summary.csv` |
| Main time plot | `aggregate_sales_over_time.png` |
| Stockout pattern | `stockout_rate_over_time.png` |
| Seasonality | `sales_by_hour_of_day.png`, `sales_by_day_of_week.png` |
| ADF/KPSS | `stationarity_tests.csv` |
| ACF/PACF | `acf_aggregate_sales.png`, `pacf_aggregate_sales.png` |
| Model comparison | `model_comparison_h1.csv` |
| Ablation | `ablation_results.csv` |
| Stockout analysis | `stockout_segment_evaluation.csv` |
| Forecast examples | `forecast_*.png` |
| Feature importance | `lightgbm_feature_importance_top20.png` |

---

# 19. Checklist cuối đưa cho Codex

```text
Sau khi code xong, hãy đảm bảo project chạy được theo thứ tự:

1. python main_eda.py
2. python main_train.py --horizon 1
3. python main_train.py --horizon 24
4. python main_evaluate.py

Kiểm tra:
- Không lỗi import.
- Không lỗi đường dẫn.
- Không dùng random split.
- Không leakage trong rolling features.
- Có đủ outputs/tables.
- Có đủ outputs/figures.
- Metrics không NaN hàng loạt.
- README có hướng dẫn chạy.
- requirements.txt đầy đủ.
```

---

# 20. Bản MVP nếu cần làm nhanh

Nếu muốn làm nhanh để kịp deadline, chỉ cần yêu cầu Codex làm bản MVP:

```text
Làm bản MVP cho Fresh50K forecasting:
1. Load data.
2. Tạo series_id.
3. EDA: data summary, aggregate plot, stockout plot.
4. Tạo calendar, lag, rolling, stockout features.
5. Split train/test theo thời gian.
6. Benchmark: Naive, Seasonal Naive 24.
7. Train LightGBM.
8. Ablation: without stockout vs with stockout.
9. Metrics: RMSE, MAE, WAPE.
10. Forecast plot cho 3 series.
11. Feature importance.
12. Save tất cả bảng/hình vào outputs/.
13. README + requirements.
```

MVP này đủ để làm slide tốt. Sau đó nếu còn thời gian thì thêm:

- ADF/KPSS.
- ACF/PACF.
- Horizon h=24.
- XGBoost.
- Stockout-segment analysis.

---

# 21. Thứ tự nên giao Codex làm

Không nên giao một cục quá to ngay. Nên giao theo thứ tự:

1. Tạo project structure + loader + README.
2. Làm EDA + data quality + plots.
3. Làm feature engineering.
4. Làm split + benchmarks.
5. Train LightGBM.
6. Tính metrics + model comparison.
7. Làm ablation with/without stockout.
8. Làm plots cho slide.
9. Clean code + test run.

Quan trọng nhất là bước 3–7. Nếu feature và evaluation đúng thì project đã có xương sống.

---

# 22. Lưu ý quan trọng để tránh bị sai bản chất

## Không random split

Sai:

```python
train_test_split(df, test_size=0.2, shuffle=True)
```

Đúng:

```python
train_df = df[df["dt"] < val_start]
val_df = df[(df["dt"] >= val_start) & (df["dt"] < test_start)]
test_df = df[df["dt"] >= test_start]
```

## Không leakage rolling features

Sai:

```python
df["roll_mean_24"] = df.groupby("series_id")["sale_amount"].rolling(24).mean()
```

Đúng:

```python
shifted = df.groupby("series_id")["sale_amount"].shift(1)
df["roll_mean_24"] = shifted.groupby(df["series_id"]).rolling(24).mean().reset_index(level=0, drop=True)
```

## Không chỉ báo cáo model tốt nhất

Cần có:

- Benchmark.
- Model comparison.
- Ablation.
- Stockout vs non-stockout evaluation.

## Không plot 50k series

Chỉ plot:

- Aggregate sales.
- 3 representative series.
- Forecast examples.

## Không dùng MAPE đơn thuần nếu sale có nhiều 0

Ưu tiên:

- WAPE.
- MAE.
- RMSE.
- sMAPE.

---

# 23. Expected final story cho slide/report

Câu chuyện mong muốn:

```text
Fresh50K là bộ dữ liệu bán lẻ nhiều chuỗi theo giờ, có đặc điểm mùa vụ, biến động, khuyến mãi, thời tiết và trạng thái stockout.

Thay vì train một mô hình riêng cho từng chuỗi, project dùng global ML forecasting model để học pattern chung giữa store-SKU series.

Các lag/rolling/calendar features giúp bắt autocorrelation và seasonality.

Các stockout-aware features giúp mô hình nhận biết những giai đoạn observed sales có thể bị giới hạn bởi tồn kho.

Kết quả được đánh giá bằng RMSE, MAE, WAPE, sMAPE và so với naive/seasonal naive benchmark.

Ablation study kiểm tra việc thêm promotion, weather và stockout features có cải thiện forecasting accuracy hay không.

Phân tích riêng stockout-period và non-stockout-period giúp đánh giá mô hình trong tình huống nghiệp vụ quan trọng.
```

---

# 24. Tên đề tài gợi ý

## Tiếng Anh

```text
Stockout-Aware Global Machine Learning Forecasting for Large-Scale Retail Time Series: Evidence from FreshRetailNet-50K
```

## Tiếng Việt

```text
Dự báo nhu cầu bán lẻ bằng mô hình học máy toàn cục có xét đến trạng thái hết hàng trên bộ dữ liệu FreshRetailNet-50K
```
