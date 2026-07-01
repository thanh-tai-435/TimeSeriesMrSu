# FreshRetailNet-50K: Stockout-Aware Demand Forecasting

## 1. Tổng Quan Đề Tài

Đề tài nghiên cứu bài toán dự báo nhu cầu bán lẻ trên bộ dữ liệu FreshRetailNet-50K trong bối cảnh dữ liệu doanh số quan sát có thể bị sai lệch bởi hiện tượng hết hàng. Khi một sản phẩm rơi vào trạng thái stockout, doanh số ghi nhận không còn phản ánh đầy đủ nhu cầu thực tế của khách hàng. Vì vậy, nếu mô hình dự báo được huấn luyện trực tiếp trên observed sales, kết quả có nguy cơ đánh giá thấp nhu cầu trong các giai đoạn chịu ảnh hưởng bởi thiếu hàng.

Hướng tiếp cận của đề tài là xây dựng một pipeline hai giai đoạn:

1. Khôi phục nhu cầu tiềm ẩn trong các khoảng thời gian stockout ở cấp độ giờ.
2. Tổng hợp nhu cầu đã khôi phục lên cấp độ ngày và dự báo tổng nhu cầu trong 7 ngày tiếp theo.

## 2. Dữ Liệu Và Phạm Vi Thực Nghiệm

Nguồn dữ liệu được sử dụng là FreshRetailNet-50K. Do giới hạn tài nguyên tính toán, thực nghiệm chính sử dụng mẫu 10% theo `series_id`, trong đó mỗi chuỗi thời gian tương ứng với một tổ hợp cửa hàng và sản phẩm.

Pipeline xử lý dữ liệu gồm các bước chính:

- Chuẩn hóa cột thời gian, định danh chuỗi và biến mục tiêu.
- Bung dữ liệu từ cấp độ ngày sang cấp độ giờ dựa trên `hours_sale` và `hours_stock_status`.
- Xác định các dòng chịu ảnh hưởng bởi stockout.
- Tạo đặc trưng thời gian, lag, rolling statistics, stockout history, peer/substitution signal và velocity signal.
- Chia tập dữ liệu theo trục thời gian thành train, validation và test.

## 3. Phương Pháp Nghiên Cứu

### 3.1. Latent Demand Recovery

Nhu cầu tiềm ẩn được khôi phục ở cấp độ giờ bằng LightGBM. Để tránh rò rỉ thông tin theo thời gian, phần recovery trong tập train sử dụng cơ chế expanding window theo từng block thời gian. Mỗi block chỉ được khôi phục bằng mô hình học từ các quan sát non-stockout trong quá khứ.

Sau khi dự báo nhu cầu cho các dòng stockout, pipeline áp dụng calibration và capping để hạn chế việc imputation làm tổng nhu cầu tăng quá mức. Nhu cầu khôi phục cuối cùng được xác định theo nguyên tắc không thấp hơn doanh số quan sát.

### 3.2. Daily Demand Forecasting

Sau bước recovery, dữ liệu hourly được tổng hợp lên daily demand. Mục tiêu dự báo chính là tổng nhu cầu 7 ngày tiếp theo trên recovered latent demand proxy.

Các mô hình được so sánh gồm:

- Observed-sales forecasting: mô hình học trực tiếp trên doanh số quan sát.
- Seasonal Naive 7-day: baseline mùa vụ theo chu kỳ tuần.
- Recovered-demand LightGBM: mô hình học trên nhu cầu đã khôi phục.
- Seasonal-ML Hybrid: kết hợp Seasonal Naive 7-day và Recovered-demand LightGBM, với trọng số chọn trên validation set.

## 4. Kết Quả Chính

Kết quả trên tập test với mục tiêu `Recovered latent demand proxy`:

| Mô hình | WAPE | Diễn giải |
|---|---:|---|
| Observed-sales forecasting | 24.92% | Dự báo trực tiếp trên observed sales bị ảnh hưởng bởi stockout bias. |
| Recovered-demand LightGBM | 20.19% | Mô hình học trên nhu cầu đã khôi phục cải thiện rõ so với observed-sales forecasting. |
| Seasonal Naive 7-day | 19.85% | Baseline mạnh do dữ liệu có chu kỳ tuần rõ rệt. |
| Seasonal-ML Hybrid | 19.62% | Mô hình tốt nhất theo WAPE trong thực nghiệm chính. |

Ngoài đánh giá forecast, đề tài cũng thực hiện các kiểm tra bổ sung cho bước imputation, gồm uplift analysis, pseudo-stockout validation, cap sensitivity analysis và residual diagnostics.

## 5. Cách Tái Lập Thực Nghiệm

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

Chạy toàn bộ pipeline:

```bash
python main_pipeline.py
```

Chạy từng bước:

```bash
python main_eda.py --sample_frac 0.1 --frequency hourly
python main_features.py --sample_frac 0.1 --frequency hourly
python main_split.py --val_days 7 --test_days 14
python main_owner_approach.py
python main_imputation_quality.py
python main_spectrum.py
```

## 6. Cấu Trúc Mã Nguồn

```text
fresh50k_forecasting/
  data/
    raw/          dữ liệu gốc
    sample/       dữ liệu sau khi lấy mẫu theo series_id
    processed/    dữ liệu trung gian và feature table

  src/
    data_loader.py          load, chuẩn hóa và bung dữ liệu hourly
    eda.py                  phân tích khám phá dữ liệu
    stationarity.py         kiểm định tính dừng, ACF và PACF
    features.py             tạo đặc trưng cho mô hình
    split.py                chia train/validation/test theo thời gian
    owner_approach.py       two-stage recovery và daily forecasting
    imputation_quality.py   đánh giá chất lượng imputation
    spectrum.py             phân tích phổ tần số
    evaluation.py           các thước đo đánh giá
    models.py               tiện ích liên quan tới feature columns

  outputs/
    figures/                biểu đồ kết quả
    tables/                 bảng kết quả
    models/                 mô hình đã huấn luyện

  deliverables/
    figures/                hình quan trọng để đưa vào báo cáo
    tables/                 bảng quan trọng để đưa vào báo cáo
```

## 7. Các File Kết Quả Quan Trọng

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

## 8. Hạn Chế

Nhu cầu khôi phục trong đề tài là một proxy được xây dựng từ dữ liệu quan sát và trạng thái stockout, không phải ground truth tuyệt đối của nhu cầu thực tế. Ngoài ra, thực nghiệm chính sử dụng 10% số chuỗi để phù hợp với tài nguyên tính toán, nên kết quả có thể thay đổi khi mở rộng sang toàn bộ dữ liệu.

Các hướng phát triển tiếp theo gồm mở rộng thực nghiệm trên toàn bộ dữ liệu, thử nghiệm các mô hình imputation chuỗi thời gian chuyên sâu hơn, và đánh giá trực tiếp tác động của dự báo lên quyết định nhập hàng.
