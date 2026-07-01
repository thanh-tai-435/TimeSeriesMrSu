# FreshRetailNet-50K Two-Stage Demand Forecasting

Project này tập trung vào một câu chuyện chính:

> Observed sales trong bán lẻ có thể bị thấp giả tạo khi stockout. Vì vậy, ta recover latent demand ở cấp hourly trước, aggregate lên daily, rồi forecast tổng nhu cầu 7 ngày tiếp theo.

## Main Story

1. Load dữ liệu FreshRetailNet-50K và lấy sample 10% theo `series_id`.
2. Bung dữ liệu ngày thành hourly bằng `hours_sale` và `hours_stock_status`.
3. Recover latent demand trong các giờ stockout bằng expanding-window LightGBM, không dùng dữ liệu tương lai.
4. Kiểm soát imputation bằng calibration + cap q90 + sensitivity checks.
5. Aggregate hourly recovered demand thành daily demand.
6. Forecast tổng demand 7 ngày tiếp theo.
7. So sánh daily baselines, observed-sales LightGBM, recovered-demand LightGBM và hybrid seasonal-ML.

## Key Result

Trên recovered latent demand proxy:

| Model | WAPE |
|---|---:|
| Observed-sales LightGBM | 24.42% |
| Recovered-demand LightGBM | 20.28% |
| Seasonal naive 7-day | 19.85% |
| Seasonal-ML hybrid | 19.67% |

Kết luận chính: seasonal pattern theo tuần rất mạnh, nên mô hình cuối dùng hybrid giữa seasonal naive 7-day và LightGBM trên recovered demand. Trọng số hybrid được chọn trên validation set, không chọn bằng test.

## Minimal Run

```bash
pip install -r requirements.txt

python main_pipeline.py
```

Hoặc chạy từng bước:

```bash
python main_eda.py --sample_frac 0.1 --frequency hourly
python main_features.py --sample_frac 0.1 --frequency hourly
python main_split.py --val_days 7 --test_days 14
python main_process_diagram.py
python main_owner_approach.py
python main_imputation_quality.py
python main_spectrum.py
python main_report_vi.py
python main_report_long_vi.py
python main_slides_vi.py
```

Compile LaTeX nếu cần:

```bash
cd outputs/reports
xelatex -interaction=nonstopmode fresh50k_report_vi.tex
xelatex -interaction=nonstopmode fresh50k_slides_vi.tex
```

## Main Deliverables

Các file nên dùng để nộp/bảo vệ nằm trong:

```text
deliverables/
  reports/
  tables/
  figures/
```

Quan trọng nhất:

- `deliverables/reports/fresh50k_report_vi.pdf`
- `deliverables/reports/fresh50k_report_long_vi.pdf`
- `deliverables/reports/fresh50k_slides_vi.pdf`
- `deliverables/tables/owner_two_stage_forecasting_comparison.csv`
- `deliverables/tables/owner_latent_recovery_summary.csv`
- `deliverables/tables/imputation_cap_sensitivity.csv`
- `deliverables/tables/imputation_pseudo_stockout_aggregate_validation.csv`

## Project Structure

```text
data/
  raw/          raw train/eval parquet
  sample/       selected 10% sample
  processed/    generated feature/recovery tables

src/
  data_loader.py
  preprocessing.py
  eda.py
  features.py
  split.py
  owner_approach.py
  imputation_quality.py
  spectrum.py
  evaluation.py

outputs/
  figures/
  tables/
  models/
  reports/

deliverables/   clean files for presentation/report submission
archive_auxiliary/ old exploratory scripts/notebooks
```

## Notes For Defense

- Forecast target chính là `target_next7_recovered_daily`, không phải raw observed sales.
- Recovery trong train dùng expanding window theo block tuần, tránh lấy ngày sau để recover ngày trước.
- Validation/test recovery dùng model fit từ non-stockout rows trong train period.
- Imputation không được xem là ground truth tuyệt đối; nó là proxy được kiểm soát bằng calibration, cap và sensitivity.
- Row-level hourly imputation rất nhiễu, nên báo cáo bổ sung aggregate validation.
- Residual rolling 7-day target vẫn có autocorrelation nhẹ vì các target liên tiếp overlap mạnh.
