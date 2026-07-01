# Deliverables

Đây là thư mục tinh gọn để nộp/bảo vệ.

## Nên mở trước

1. `reports/fresh50k_slides_vi.pdf`
2. `reports/fresh50k_report_long_vi.pdf` - bản focused report, rõ contribution, ít phụ lục thừa
3. `reports/fresh50k_report_vi.pdf` nếu cần bản ngắn hơn

Nếu cần chỉnh hoặc compile lại bản report đầy đủ, dùng thư mục:

```text
report_long_package/
```

## Bảng kết quả chính

- `tables/owner_two_stage_forecasting_comparison.csv`
- `tables/owner_latent_recovery_summary.csv`
- `tables/imputation_uplift_summary.csv`
- `tables/imputation_cap_sensitivity.csv`
- `tables/imputation_pseudo_stockout_aggregate_validation.csv`

## Số cần nhớ

- Observed-sales LightGBM WAPE: 24.42%
- Recovered-demand LightGBM WAPE: 20.28%
- Seasonal naive 7-day WAPE: 19.85%
- Seasonal-ML hybrid WAPE: 19.67%
- Recovered demand lift over observed sales: khoảng 10.68%
