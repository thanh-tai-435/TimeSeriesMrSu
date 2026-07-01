# FreshRetailNet-50K Stockout-Aware Demand Forecasting

Project này giữ một hướng tiếp cận duy nhất, dễ bảo vệ:

> Doanh số quan sát trong bán lẻ có thể bị thấp giả tạo khi sản phẩm hết hàng. Vì vậy, pipeline khôi phục nhu cầu ẩn ở cấp giờ, tổng hợp lên cấp ngày, rồi dự báo tổng nhu cầu 7 ngày tiếp theo.

## Câu chuyện chính

1. Load dữ liệu FreshRetailNet-50K và lấy mẫu 10% theo `series_id`.
2. Bung dữ liệu từ ngày sang giờ bằng `hours_sale` và `hours_stock_status`.
3. Nhận diện stockout để thấy observed sales là dữ liệu bị kiểm duyệt.
4. Khôi phục latent demand bằng expanding-window LightGBM, chỉ dùng dữ liệu quá khứ để tránh leakage.
5. Calibration và capping để imputation không làm tổng demand tăng quá mức.
6. Tổng hợp recovered hourly demand lên daily demand.
7. Dự báo tổng demand 7 ngày tiếp theo bằng Seasonal Naive 7-day, LightGBM recovered-demand và Hybrid Seasonal-ML.
8. Đánh giá bằng WAPE, bias/WPE, residual diagnostics, interval coverage và kiểm tra pseudo-stockout.

## Kết quả chính

Trên target `Recovered latent demand proxy`:

| Model | WAPE | Ghi chú |
|---|---:|---|
| Observed-sales forecasting | 24.92% | Bị thấp do stockout bias |
| Recovered-demand LightGBM | 20.19% | ML học trên demand đã recover |
| Seasonal naive 7-day | 19.85% | Baseline rất mạnh vì chu kỳ tuần rõ |
| Seasonal-ML hybrid | 19.62% | Tốt nhất theo WAPE, kết hợp weekly pattern và ML |

Kết luận nên trình bày: contribution chính không phải là “ML thắng xa baseline”, mà là framing đúng vấn đề stockout, recovery không leakage, kiểm soát imputation, rồi so sánh công bằng với baseline mùa vụ mạnh.

## Chạy lại pipeline

```bash
pip install -r requirements.txt
python main_pipeline.py
```

Hoặc chạy từng bước:

```bash
python main_eda.py --sample_frac 0.1 --frequency hourly
python main_features.py --sample_frac 0.1 --frequency hourly
python main_split.py --val_days 7 --test_days 14
python main_owner_approach.py
python main_imputation_quality.py
python main_spectrum.py
```

## Cấu trúc đã tinh gọn

```text
data/
  raw/          dữ liệu parquet gốc
  sample/       sample 10% theo series_id
  processed/    feature table và bảng recovery trung gian

src/
  data_loader.py
  eda.py
  stationarity.py
  features.py
  split.py
  owner_approach.py
  imputation_quality.py
  spectrum.py
  evaluation.py
  models.py

outputs/
  figures/      hình phân tích và kết quả mô hình
  tables/       bảng số liệu dùng để báo cáo/bảo vệ
  models/       model artifacts

deliverables/
  figures/      bản copy hình quan trọng nếu cần nộp
  tables/       bản copy bảng quan trọng nếu cần nộp
```

## File kết quả nên dùng khi bảo vệ

- `outputs/tables/data_quality_summary.csv`
- `outputs/tables/owner_latent_recovery_summary.csv`
- `outputs/tables/imputation_pseudo_stockout_aggregate_validation.csv`
- `outputs/tables/imputation_cap_sensitivity.csv`
- `outputs/tables/owner_two_stage_forecasting_comparison.csv`
- `outputs/tables/owner_two_stage_diagnostics.csv`
- `outputs/figures/stockout_rate_over_time.png`
- `outputs/figures/owner_observed_vs_recovered_demand.png`
- `outputs/figures/owner_two_stage_forecast_comparison.png`
- `outputs/figures/owner_two_stage_residual_acf.png`

## Ghi chú khi bảo vệ

- Forecast target chính là `target_next7_recovered_daily`, không phải raw observed sales.
- Recovery trong train dùng expanding window theo block tuần: block hiện tại chỉ được học từ quá khứ.
- Validation/test được recover bằng model fit từ non-stockout rows trong train period.
- Imputation không được xem là ground truth tuyệt đối; nó là proxy được kiểm soát bằng calibration, cap và pseudo-stockout validation.
- Seasonal Naive 7-day phải được giữ làm baseline chính vì dữ liệu có weekly pattern mạnh.
- Hybrid hợp lý vì nó giữ weekly pattern của Seasonal Naive nhưng thêm khả năng học stockout history, recovery signal, peer/substitution và velocity features.
