"""Generate a Vietnamese LaTeX report from saved Fresh50K results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR, TABLES_DIR, ensure_directories
from main_report import csv_table, figure, latex_escape, read_metric


def build_report_vi() -> str:
    data_quality = TABLES_DIR / "data_quality_summary.csv"
    split_summary = TABLES_DIR / "split_summary.csv"
    stationarity = TABLES_DIR / "stationarity_tests.csv"
    comparison_h1 = TABLES_DIR / "model_comparison_h1.csv"
    comparison_h24 = TABLES_DIR / "model_comparison_h24.csv"
    ablation = TABLES_DIR / "ablation_results.csv"
    stockout_eval = TABLES_DIR / "stockout_segment_evaluation.csv"
    residual_diagnostics = TABLES_DIR / "residual_diagnostics.csv"
    quantile_metrics = TABLES_DIR / "quantile_interval_metrics_h1.csv"
    rolling_eval = TABLES_DIR / "rolling_window_evaluation.csv"
    sarimax_comparison = TABLES_DIR / "aggregate_sarimax_comparison.csv"
    stockout_censoring = TABLES_DIR / "stockout_censoring_summary.csv"
    feature_importance = TABLES_DIR / "lightgbm_feature_importance.csv"
    ridge_coef = TABLES_DIR / "ridge_coefficients_h1.csv"
    owner_recovery = TABLES_DIR / "owner_latent_recovery_summary.csv"
    owner_forecast = TABLES_DIR / "owner_two_stage_forecasting_comparison.csv"
    owner_daily_split = TABLES_DIR / "owner_daily_split_summary.csv"
    owner_hybrid_blend = TABLES_DIR / "owner_hybrid_blend_selection.csv"
    owner_diagnostics = TABLES_DIR / "owner_two_stage_diagnostics.csv"
    owner_nonoverlap_diagnostics = TABLES_DIR / "owner_two_stage_nonoverlap_diagnostics.csv"
    owner_feature_importance = TABLES_DIR / "owner_recovereddemand_forecasting_feature_importance.csv"
    spectrum_hourly = TABLES_DIR / "spectrum_hourly_top_peaks.csv"
    spectrum_daily = TABLES_DIR / "spectrum_daily_recovered_top_peaks.csv"
    imputation_uplift = TABLES_DIR / "imputation_uplift_summary.csv"
    imputation_validation = TABLES_DIR / "imputation_pseudo_stockout_validation.csv"
    imputation_aggregate_validation = TABLES_DIR / "imputation_pseudo_stockout_aggregate_validation.csv"
    imputation_sensitivity = TABLES_DIR / "imputation_cap_sensitivity.csv"
    imputation_series_uplift = TABLES_DIR / "imputation_series_uplift.csv"

    sample_frac = read_metric(data_quality, "Sample fraction")
    n_rows = read_metric(data_quality, "Number of rows")
    n_series = read_metric(data_quality, "Number of series")
    start_date = read_metric(data_quality, "Start date")
    end_date = read_metric(data_quality, "End date")

    h1 = pd.read_csv(comparison_h1)
    best_h1 = h1.sort_values("WAPE").iloc[0]
    seasonal_h1 = h1[h1["Model"] == "SeasonalNaive168"].iloc[0]
    improvement = (seasonal_h1["WAPE"] - best_h1["WAPE"]) / seasonal_h1["WAPE"] * 100
    owner_cmp = pd.read_csv(owner_forecast)
    owner_main = owner_cmp[
        owner_cmp["Evaluation target"] == "Recovered latent demand proxy"
    ].sort_values("WAPE").iloc[0]
    owner_recovered = owner_cmp[
        (owner_cmp["Model"] == "Recovered-demand forecasting")
        & (owner_cmp["Evaluation target"] == "Recovered latent demand proxy")
    ].iloc[0]
    owner_observed = owner_cmp[
        (owner_cmp["Model"] == "Observed-sales forecasting")
        & (owner_cmp["Evaluation target"] == "Recovered latent demand proxy")
    ].iloc[0]
    impute_lift_pct = float(read_metric(imputation_uplift, "Recovered lift over observed")) * 100
    calibration_factor = read_metric(owner_recovery, "Imputation calibration factor")
    lost_demand_cap = read_metric(owner_recovery, "Lost demand cap value")
    owner_diag = pd.read_csv(owner_diagnostics)
    owner_recovered_diag = owner_diag[
        (owner_diag["Model"] == owner_main["Model"])
        & (owner_diag["Evaluation target"] == "Recovered latent demand proxy")
    ].iloc[0]
    nonoverlap_diag = pd.read_csv(owner_nonoverlap_diagnostics)
    nonoverlap_recovered_diag = nonoverlap_diag[
        nonoverlap_diag["Model"] == owner_main["Model"]
    ].iloc[0]

    content = rf"""\documentclass[12pt,a4paper]{{article}}
\usepackage{{fontspec}}
\setmainfont{{Times New Roman}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{caption}}
\usepackage{{array}}
\geometry{{margin=1in}}

\title{{Dự báo chuỗi thời gian bán lẻ có xét đến tình trạng hết hàng trên FreshRetailNet-50K}}
\author{{}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Báo cáo này trình bày một quy trình two-stage cho bộ dữ liệu FreshRetailNet-50K: khôi phục latent demand trong các giờ stockout, sau đó dự báo nhu cầu trên recovered demand.
Thí nghiệm sử dụng mẫu {latex_escape(sample_frac)} ở cấp \texttt{{series\_id}}, giữ nguyên toàn bộ dòng thời gian của các chuỗi được chọn.
Dữ liệu theo ngày được bung thành dữ liệu theo giờ bằng \texttt{{hours\_sale}} và \texttt{{hours\_stock\_status}}, rồi aggregate lại thành daily recovered demand cho bài toán dự báo tổng nhu cầu 7 ngày.
Tập dữ liệu cuối cùng có {latex_escape(n_rows)} quan sát theo giờ và {latex_escape(n_series)} chuỗi, từ {latex_escape(start_date)} đến {latex_escape(end_date)}.
\end{{abstract}}

\section{{Câu hỏi nghiên cứu và động lực}}
Bài toán đặt ra là dự báo lượng bán trong tương lai cho nhiều chuỗi cửa hàng--sản phẩm, đồng thời xét đến tình trạng hết hàng.
Trong bán lẻ, doanh số quan sát được trong lúc hết hàng thường bị kiểm duyệt: bán thấp không nhất thiết là nhu cầu thấp, mà có thể do không còn hàng để bán.
Vì vậy, mô hình cần tận dụng thông tin mùa vụ, lịch, khuyến mãi, thời tiết và đặc biệt là đặc trưng stockout-aware.

\section{{Dữ liệu và tiền xử lý}}
Dữ liệu gốc gồm \texttt{{train.parquet}} và \texttt{{eval.parquet}}.
Sau khi chuẩn hóa, biến \texttt{{series\_id}} được tạo từ \texttt{{city\_id}}, \texttt{{store\_id}} và \texttt{{product\_id}}.
Các chuỗi bán theo giờ trong \texttt{{hours\_sale}} được bung thành từng quan sát giờ; trạng thái hết hàng theo giờ được lấy từ \texttt{{hours\_stock\_status}}.

\begin{{table}}[H]
\centering
\caption{{Tóm tắt chất lượng dữ liệu}}
{csv_table(data_quality)}
\end{{table}}

\section{{Phân tích khám phá dữ liệu}}
Các biểu đồ EDA cho thấy dữ liệu có tính mùa vụ theo giờ, tính thưa cao, và tỷ lệ hết hàng biến động rõ theo thời gian.

{figure(FIGURES_DIR / "aggregate_sales_over_time.png", "Tổng lượng bán theo giờ theo thời gian.", "fig:vi-aggregate-sales")}
{figure(FIGURES_DIR / "stockout_rate_over_time.png", "Tỷ lệ hết hàng theo giờ theo thời gian.", "fig:vi-stockout-rate")}
{figure(FIGURES_DIR / "sale_amount_distribution.png", "Phân phối lượng bán theo giờ.", "fig:vi-sale-dist", r"0.78\linewidth")}
{figure(FIGURES_DIR / "log_sale_amount_distribution.png", "Phân phối log1p của lượng bán theo giờ.", "fig:vi-log-sale-dist", r"0.78\linewidth")}
{figure(FIGURES_DIR / "sales_by_hour_of_day.png", "Lượng bán trung bình theo giờ trong ngày.", "fig:vi-hour-seasonality", r"0.78\linewidth")}
{figure(FIGURES_DIR / "sales_by_day_of_week.png", "Lượng bán trung bình theo thứ trong tuần.", "fig:vi-dow-seasonality", r"0.78\linewidth")}

\begin{{table}}[H]
\centering
\caption{{Các chuỗi đại diện được chọn để minh họa}}
{csv_table(TABLES_DIR / "representative_series.csv")}
\end{{table}}

{figure(FIGURES_DIR / "representative_series_high_volume.png", "Chuỗi đại diện có tổng lượng bán cao.", "fig:vi-rep-high")}
{figure(FIGURES_DIR / "representative_series_intermittent.png", "Chuỗi đại diện có nhu cầu thưa/ngắt quãng.", "fig:vi-rep-intermittent")}
{figure(FIGURES_DIR / "representative_series_stockout_heavy.png", "Chuỗi đại diện có tỷ lệ hết hàng cao.", "fig:vi-rep-stockout")}

\section{{Tính dừng, ACF/PACF và nhận dạng mô hình}}
ADF và KPSS được dùng để kiểm tra tính dừng trên chuỗi tổng hợp và các chuỗi đại diện.
Kết quả cho thấy sai phân bậc một và sai phân mùa vụ lag 24 giúp cải thiện tính dừng, phù hợp với việc sử dụng các đặc trưng lag 24 giờ và lag 168 giờ.

\begin{{table}}[H]
\centering
\caption{{Kết quả kiểm định ADF/KPSS}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(stationarity, columns=["Series", "Transformation", "ADF p-value", "KPSS p-value", "Conclusion", "N"])}
}}
\end{{table}}

{figure(FIGURES_DIR / "acf_aggregate_sales.png", "ACF của chuỗi tổng hợp theo giờ.", "fig:vi-acf-aggregate", r"0.85\linewidth")}
{figure(FIGURES_DIR / "pacf_aggregate_sales.png", "PACF của chuỗi tổng hợp theo giờ.", "fig:vi-pacf-aggregate", r"0.85\linewidth")}

\subsection{{Phân tích phổ tần số}}
Ngoài ACF/PACF, báo cáo dùng periodogram để kiểm tra seasonality trong miền tần số.
Mục tiêu là xác nhận liệu các chu kỳ được dùng trong feature engineering và baseline có xuất hiện rõ trong dữ liệu hay không.
Với chuỗi hourly aggregate, peak mạnh nhất nằm ở khoảng 24 giờ, ủng hộ việc dùng lag 24 giờ và các đặc trưng theo giờ trong ngày.
Một peak yếu hơn gần 168 giờ cũng xuất hiện, phù hợp với seasonality theo tuần.
Sau khi chuyển sang daily recovered demand, peak lớn nhất nằm gần 7 ngày, củng cố lựa chọn lag 7/14/28 ngày và seasonal naive theo tuần cho bài toán two-stage.

{figure(FIGURES_DIR / "spectrum_hourly_aggregate_sales.png", "Phổ tần số của aggregate observed sales theo giờ.", "fig:vi-spectrum-hourly", r"0.88\linewidth")}

\begin{{table}}[H]
\centering
\caption{{Các peak chu kỳ lớn nhất của hourly aggregate sales}}
{csv_table(spectrum_hourly, columns=["Rank", "Period", "Frequency", "Power", "Period unit"])}
\end{{table}}

{figure(FIGURES_DIR / "spectrum_daily_recovered_demand.png", "Phổ tần số của daily recovered demand.", "fig:vi-spectrum-daily", r"0.88\linewidth")}

\begin{{table}}[H]
\centering
\caption{{Các peak chu kỳ lớn nhất của daily recovered demand}}
{csv_table(spectrum_daily, columns=["Rank", "Period", "Frequency", "Power", "Period unit"])}
\end{{table}}

\section{{Thiết kế đặc trưng và chia tập dữ liệu}}
Các đặc trưng được tạo gồm calendar features, lag features, rolling features, promotion/weather features và stockout-aware features.
Để tránh rò rỉ dữ liệu, tất cả rolling features đều được shift một bước trước khi tính rolling.
Mục tiêu dự báo là \texttt{{target\_h1}} và \texttt{{target\_h24}}, tương ứng với dự báo trước 1 giờ và 24 giờ.

\begin{{table}}[H]
\centering
\caption{{Chia train/validation/test theo thời gian}}
{csv_table(split_summary)}
\end{{table}}

Với pipeline two-stage, phần forecasting cuối cùng được định nghĩa ở cấp ngày theo \textbf{{forecast origin}}.
Mỗi dòng dự báo tại ngày $t$ có target là tổng recovered demand trong 7 ngày sau đó, tức từ $t+1$ đến $t+7$.
Vì vậy, báo cáo lưu thêm bảng split riêng cho daily forecasting để phân biệt rõ origin window và target window.

\begin{{table}}[H]
\centering
\caption{{Chia tập cho daily 7-day forecasting theo forecast origin}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(owner_daily_split)}
}}
\end{{table}}

\section{{Lý do lựa chọn mô hình}}
Mô hình chính được chọn là LightGBM vì bài toán có nhiều chuỗi store--product, dữ liệu thưa, nhiều biến ngoại sinh và nhiều đặc trưng phi tuyến.
Huấn luyện một mô hình ARIMA/SARIMA riêng cho từng chuỗi sẽ khó mở rộng, không ổn định với chuỗi thưa và khó đưa đầy đủ thông tin khuyến mãi, thời tiết, stockout.
Ridge được giữ lại như baseline tuyến tính dễ diễn giải.
SARIMAX được dùng trên chuỗi tổng hợp như một đối chứng cổ điển, nhưng không thay thế cho mô hình toàn cục ở cấp sản phẩm--cửa hàng.
Deep learning không được chọn làm mô hình chính vì yêu cầu nhiều tài nguyên và tuning hơn, trong khi LightGBM cho tỉ lệ hiệu quả/độ phức tạp tốt hơn trong bối cảnh đồ án tái lập được.

\section{{Tiếp cận theo hướng của paper/repository}}
Paper FreshRetailNet-50K và repository baseline không xem đây chỉ là bài toán dự báo doanh số quan sát được.
Hướng tiếp cận chính là two-stage censored demand modeling:
\begin{{enumerate}}
    \item Khôi phục latent demand trong các giờ stockout vì observed sales bị kiểm duyệt.
    \item Dùng recovered demand để huấn luyện mô hình dự báo nhu cầu robust hơn.
\end{{enumerate}}

Trong đồ án này, ta triển khai một phiên bản nhẹ và tái lập được của ý tưởng đó.
Để tránh leakage theo thời gian, phần recovery trong train period dùng expanding window theo block tuần: mỗi block chỉ được recover bằng mô hình học từ các giờ không stockout trong quá khứ.
Validation và test được recover bằng final recovery model train trên các giờ không stockout thuộc train period.
Với các giờ stockout, mô hình tạo \texttt{{imputed\_demand\_raw}}, sau đó hiệu chỉnh bằng calibration factor học từ validation non-stockout rows.
Phần lost demand được tính bằng \texttt{{max(calibrated imputed demand - observed sales, 0)}} và được cap tại phân vị q90 của lost demand dương để tránh một số giờ stockout làm tổng demand tăng quá mức.
Cuối cùng, \texttt{{recovered\_demand = observed sales + capped lost demand}}.
Sau đó recovered demand theo giờ được aggregate thành daily demand để làm bài toán dự báo tổng nhu cầu 7 ngày tiếp theo.

\subsection{{Cơ chế expanding-window recovery}}
Quy trình recovery được thiết kế theo nguyên tắc point-in-time: tại một thời điểm bất kỳ, mô hình không được sử dụng dữ liệu ở tương lai để khôi phục demand ở hiện tại.
Sơ đồ dưới đây tóm tắt toàn bộ luồng xử lý từ dữ liệu gốc đến latent demand recovery và forecasting.

{figure(FIGURES_DIR / "owner_expanding_window_process.png", "Sơ đồ quy trình expanding-window latent demand recovery và forecasting.", "fig:vi-expanding-window-process", r"0.88\linewidth")}

\subsubsection{{Quy trình xử lý riêng cho expanding-window recovery}}
Phần recovery được tách thành một quy trình riêng vì đây là bước dễ gây leakage nhất trong bài toán latent demand.
Ý tưởng cốt lõi là mô phỏng đúng bối cảnh triển khai thực tế: khi cần khôi phục nhu cầu ở tuần hiện tại, mô hình chỉ được biết các quan sát không stockout đã xảy ra trước tuần đó.
Do đó, ngay cả trong train period, ta không dùng dữ liệu của các ngày phía sau để recover các ngày phía trước.

{figure(FIGURES_DIR / "owner_expanding_window_recovery_detail.png", "Quy trình xử lý riêng cho expanding-window latent demand recovery.", "fig:vi-expanding-window-recovery-detail", r"0.9\linewidth")}

Quy trình chi tiết gồm các bước:
\begin{{enumerate}}
    \item \textbf{{Sắp xếp dữ liệu theo thời gian}}: toàn bộ hourly frame được sort theo \texttt{{dt}}, sau đó chia train/validation/test theo mốc thời gian cố định.
    \item \textbf{{Giữ warmup 14 ngày đầu}}: giai đoạn này chưa đủ lịch sử để fit recovery model ổn định nên \texttt{{recovered\_demand}} được đặt bằng observed sales.
    \item \textbf{{Tạo block recovery trong train period}}: phần train sau warmup được chia thành các block 7 ngày liên tiếp.
    \item \textbf{{Fit model theo nguyên tắc quá khứ}}: với block thứ $k$, tập train của recovery model chỉ gồm các dòng có \texttt{{dt < block\_start}} và \texttt{{stockout\_flag = 0}}.
    \item \textbf{{Recover block hiện tại}}: model dự đoán \texttt{{imputed\_demand}} cho block $k$; với dòng stockout, ta chỉ cộng thêm phần lost demand dương sau calibration và cap, nên nhu cầu recovered không thấp hơn doanh số đã bán thật nhưng cũng không bị phóng đại bởi các dự đoán cực trị.
    \item \textbf{{Di chuyển cửa sổ}}: sau khi xử lý xong block $k$, block này trở thành quá khứ cho block $k+1$, nhưng chỉ các dòng \texttt{{non-stockout}} mới được dùng để fit các model sau.
\end{{enumerate}}
Sau khi train period kết thúc, một final recovery model được fit trên toàn bộ non-stockout rows của train period và chỉ dùng để recover validation/test.
Cách này giữ đúng point-in-time constraint: validation/test không tham gia fit model, và mỗi block trong train cũng không nhìn thấy tương lai của chính nó.

Mốc warmup 14 ngày được chọn như một thỏa hiệp giữa độ ổn định của recovery model và lượng dữ liệu còn lại để recover.
Vì dữ liệu có mùa vụ theo giờ và theo tuần, 14 ngày tương ứng hai chu kỳ tuần đầy đủ, giúp mô hình nhìn thấy ít nhất hai lần lặp của các pattern theo thứ trong tuần và cuối tuần.
Khoảng này cũng đủ để các đặc trưng lag/rolling ngắn hạn như lag 24 giờ, lag 168 giờ, rolling 24 giờ và rolling 168 giờ bắt đầu có ý nghĩa hơn trước khi fit recovery model đầu tiên.
Nếu warmup quá ngắn, model recovery đầu tiên dễ học từ lịch sử quá ít và impute stockout thiếu ổn định; nếu quá dài, nhiều giờ stockout đầu kỳ bị giữ nguyên observed sales, làm giảm lợi ích của latent demand recovery.
Do đó, 14 ngày là lựa chọn thực dụng cho đồ án: đủ bao phủ seasonality tuần nhưng không làm mất quá nhiều train period.

Các bước cụ thể như sau:
\begin{{enumerate}}
    \item \textbf{{Warmup period}}: 14 ngày đầu được giữ nguyên observed sales vì chưa đủ lịch sử để train recovery model ổn định.
    \item \textbf{{Chia train period thành block tuần}}: sau warmup, train period được chia thành các block 7 ngày liên tiếp.
    \item \textbf{{Train recovery cho từng block}}: với mỗi block, LightGBM recovery model chỉ được train trên các dòng \texttt{{non-stockout}} nằm trước thời điểm bắt đầu block.
    \item \textbf{{Recover block hiện tại}}: model vừa train được dùng để ước lượng \texttt{{imputed\_demand}} cho block hiện tại; nếu dòng đó stockout, recovered demand bằng observed sales cộng phần lost demand đã calibration và cap.
    \item \textbf{{Validation/Test}}: sau khi kết thúc train period, một final recovery model được train trên toàn bộ non-stockout rows của train period và dùng để recover validation/test.
\end{{enumerate}}
Cách làm này tránh cả hai dạng leakage: không dùng validation/test để fit recovery model, và cũng không dùng các ngày phía sau trong train period để recover các ngày phía trước.

\begin{{table}}[H]
\centering
\caption{{Tóm tắt khôi phục latent demand}}
{csv_table(owner_recovery)}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Các block expanding-window dùng trong latent demand recovery}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(TABLES_DIR / "owner_recovery_blocks.csv", columns=["Block start", "Block end", "Training end", "Training rows", "Predicted rows"])}
}}
\end{{table}}

{figure(FIGURES_DIR / "owner_observed_vs_recovered_demand.png", "Observed sales và recovered latent demand theo thời gian.", "fig:vi-owner-recovered")}

\subsection{{Kiểm tra chất lượng imputation}}
Vì recovered demand là biến được ước lượng, báo cáo không xem imputation là ground truth tuyệt đối.
Ta kiểm tra chất lượng impute theo ba hướng.
Thứ nhất, đo tổng mức uplift để xem recovered demand có làm tổng sales tăng quá mức hay không.
Thứ hai, dùng pseudo-stockout validation: lấy các dòng non-stockout trong validation period, giả sử chúng cần impute, rồi so sánh prediction của recovery model với observed sales thật.
Thứ ba, kiểm tra sensitivity khi cap phần recovered lost demand ở các phân vị khác nhau.

\begin{{table}}[H]
\centering
\caption{{Tổng mức uplift do latent-demand imputation}}
{csv_table(imputation_uplift)}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Pseudo-stockout validation trên non-stockout rows}}
{csv_table(imputation_validation)}
\end{{table}}

Do dữ liệu hourly rất thưa và nhiều giá trị nhỏ/zero, row-level WAPE của pseudo-stockout validation có thể nhìn xấu hơn thực tế vận hành.
Vì vậy, báo cáo bổ sung kiểm tra ở cấp aggregate, gồm hourly aggregate, daily aggregate và series-daily aggregate.

\begin{{table}}[H]
\centering
\caption{{Pseudo-stockout validation ở các cấp aggregate}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(imputation_aggregate_validation)}
}}
\end{{table}}

Kết quả pseudo-stockout validation cho thấy recovery model có xu hướng dự đoán cao hơn observed sales trên các dòng non-stockout validation.
Vì vậy, phần impute được xem là một proxy có kiểm soát chứ không phải nhãn thật.
Trong pipeline chính, dự đoán raw được chia cho calibration factor {latex_escape(calibration_factor)} và phần lost demand dương được cap ở giá trị {latex_escape(lost_demand_cap)}.
Để tránh kết luận phụ thuộc quá mạnh vào impute, báo cáo thêm sensitivity theo mức cap lost demand.

\begin{{table}}[H]
\centering
\caption{{Sensitivity khi cap phần recovered lost demand}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(imputation_sensitivity, columns=["Scenario", "Cap value", "Recovered demand", "Recovered lost demand", "Recovered lift over observed", "Share of original recovered lost demand"])}
}}
\end{{table}}

Ở mức toàn bộ dataset, recovered demand tăng khoảng {impute_lift_pct:.2f}\% so với observed sales.
Mức này không làm tổng demand tăng gấp nhiều lần, nhưng một số series có stockout rate rất cao có thể có lift lớn.
Do đó, các series uplift cao được kiểm tra riêng để phát hiện trường hợp forecast phụ thuộc quá mạnh vào imputation.

{figure(FIGURES_DIR / "imputation_daily_uplift.png", "Observed sales, recovered demand và phần lost demand recovered theo ngày.", "fig:vi-imputation-daily-uplift")}
{figure(FIGURES_DIR / "imputation_series_lift_distribution.png", "Phân phối recovered lift ở cấp series.", "fig:vi-imputation-lift-dist", r"0.78\linewidth")}
{figure(FIGURES_DIR / "imputation_top_series_lift.png", "Top series có recovered lift cao nhất.", "fig:vi-imputation-top-series", r"0.78\linewidth")}

\begin{{table}}[H]
\centering
\caption{{Top series phụ thuộc nhiều nhất vào imputation}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(imputation_series_uplift, columns=["series_id", "observed_sales", "recovered_demand", "recovered_lost_demand", "stockout_rate", "Recovered lift over observed", "Recovered share from imputation"], max_rows=10)}
}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{So sánh forecast trên observed sales và recovered demand}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(owner_forecast, columns=["Model", "Training target", "Evaluation target", "RMSE", "MAE", "WAPE", "sMAPE", "WPE", "N"])}
}}
\end{{table}}

{figure(FIGURES_DIR / "owner_two_stage_bias_comparison.png", "So sánh bias WPE khi đánh giá trên recovered latent demand proxy.", "fig:vi-owner-bias", r"0.82\linewidth")}
{figure(FIGURES_DIR / "owner_two_stage_forecast_comparison.png", "Dự báo tổng nhu cầu 7 ngày theo hướng two-stage.", "fig:vi-owner-forecast")}

\section{{Kết quả mô hình two-stage}}
Sau khi khôi phục latent demand, phần dự báo chính của đồ án không còn xem observed sales là target cuối cùng.
Target chính là \texttt{{target\_next7\_recovered\_daily}}, tức tổng nhu cầu 7 ngày tiếp theo sau khi đã hiệu chỉnh các giờ stockout.
Để chứng minh lợi ích của two-stage, ta so sánh hai mô hình LightGBM có cùng dạng global model:
\begin{{enumerate}}
    \item \textbf{{Observed-sales forecasting}}: train trên doanh số quan sát được, sau đó đánh giá lại trên recovered latent demand proxy.
    \item \textbf{{Recovered-demand forecasting}}: train trực tiếp trên recovered demand và đánh giá trên recovered latent demand proxy.
\end{{enumerate}}
Ngoài ra, báo cáo thêm các baseline daily không cần train như naive, seasonal naive 7 ngày và rolling mean 14 ngày.
Seasonal naive 7 ngày là baseline rất mạnh vì phổ tần số và ACF đều cho thấy seasonality theo tuần.
Mô hình cuối cùng được dùng để kết luận là \textbf{{Recovered seasonal-ML hybrid}}: blend giữa seasonal naive 7 ngày và Recovered-demand LightGBM, với trọng số được chọn trên validation set.
Cách này giữ được nền mùa vụ mạnh của seasonal naive, đồng thời cho phép LightGBM hiệu chỉnh theo stockout, lag, rolling và calendar features.

\begin{{table}}[H]
\centering
\caption{{Kết quả chính của mô hình two-stage}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(owner_forecast, columns=["Model", "Training target", "Evaluation target", "RMSE", "MAE", "WAPE", "sMAPE", "WPE", "N"])}
}}
\end{{table}}

Kết quả cho thấy train trên recovered demand làm giảm sai số khi đánh giá trên recovered latent demand proxy, đồng thời làm bias âm nhỏ hơn.
Điều này quan trọng vì mô hình train trên observed sales có xu hướng under-forecast nhu cầu trong các giai đoạn từng bị stockout.
Baseline seasonal naive 7 ngày cũng rất cạnh tranh, cho thấy tính mùa vụ tuần là pattern chính của dữ liệu.
Hybrid seasonal-ML đạt WAPE {owner_main["WAPE"]:.4f}, là kết quả tốt nhất trong bảng, nên được dùng làm mô hình kết luận cuối.

\begin{{table}}[H]
\centering
\caption{{Chọn trọng số hybrid trên validation set}}
\resizebox{{0.72\linewidth}}{{!}}{{%
{csv_table(owner_hybrid_blend, max_rows=8)}
}}
\end{{table}}

{figure(FIGURES_DIR / "owner_two_stage_bias_comparison.png", "Bias WPE của mô hình observed-sales và recovered-demand khi đánh giá trên recovered latent demand proxy.", "fig:vi-owner-bias-main", r"0.82\linewidth")}
{figure(FIGURES_DIR / "owner_two_stage_forecast_comparison.png", "Dự báo tổng nhu cầu 7 ngày theo hướng two-stage.", "fig:vi-owner-forecast-main")}

\subsection{{Benchmark observed-sales hourly}}
Các kết quả h=1 và h=24 trên observed sales vẫn được giữ như benchmark phụ để cho thấy pipeline feature engineering và LightGBM hourly hoạt động hợp lý.
Tuy nhiên, chúng không còn là kết quả chính vì observed sales bị kiểm duyệt trong các giờ stockout.

\begin{{table}}[H]
\centering
\caption{{Benchmark phụ trên observed sales cho horizon h=1}}
{csv_table(comparison_h1, columns=["Model", "RMSE", "MAE", "WAPE", "sMAPE", "N"])}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Benchmark phụ trên observed sales cho horizon h=24}}
{csv_table(comparison_h24, columns=["Model", "RMSE", "MAE", "WAPE", "sMAPE", "N"])}
\end{{table}}

\section{{Diagnostics và khoảng dự báo two-stage}}
Diagnostics chính được thực hiện trên mô hình \textbf{{{latex_escape(owner_main["Model"])}}}, với residual được tính theo recovered latent demand proxy.
Cách đánh giá này nhất quán với mục tiêu của two-stage: sau khi hiệu chỉnh latent demand, mô hình phải dự báo nhu cầu đã khử kiểm duyệt, không chỉ dự báo observed sales.
Bảng dưới đây vẫn hiển thị cả baseline, observed-sales, recovered-demand và hybrid để so sánh trực tiếp, nhưng diễn giải chính tập trung vào mô hình có WAPE thấp nhất trên recovered latent demand proxy.

\begin{{table}}[H]
\centering
\caption{{Diagnostics của mô hình two-stage trên recovered latent demand proxy}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(owner_diagnostics, columns=["Model", "Evaluation target", "RMSE", "MAE", "WAPE", "sMAPE", "WPE", "Residual mean", "Residual std", "Ljung-Box p-value", "Jarque-Bera p-value", "N"])}
}}
\end{{table}}

{figure(FIGURES_DIR / "owner_two_stage_residual_distribution.png", "Phân phối residual của recovered-demand forecasting trên recovered latent demand proxy.", "fig:vi-owner-residual-dist", r"0.78\linewidth")}
{figure(FIGURES_DIR / "owner_two_stage_residual_acf.png", "Residual ACF của recovered-demand forecasting theo daily aggregate.", "fig:vi-owner-residual-acf", r"0.82\linewidth")}
{figure(FIGURES_DIR / "owner_two_stage_diagnostics_wape.png", "So sánh WAPE diagnostics của hai hướng train khi đánh giá trên recovered latent demand proxy.", "fig:vi-owner-diagnostics-wape", r"0.78\linewidth")}

Vì target của bài toán là tổng nhu cầu 7 ngày tiếp theo, các nhãn dự báo theo từng ngày bị overlap mạnh: forecast tại ngày $t$ và ngày $t+1$ chia sẻ 6/7 ngày trong target.
Do đó, residual theo daily rolling evaluation có thể tự tương quan một phần do thiết kế target, không chỉ do mô hình học thiếu pattern.
Để kiểm tra độc lập residual công bằng hơn, báo cáo thêm diagnostics trên các mốc non-overlapping cách nhau 7 ngày.
Với {latex_escape(owner_main["Model"])}, diagnostics chính có Ljung--Box p-value {owner_recovered_diag["Ljung-Box p-value"]:.4f}, tức vẫn còn tự tương quan nhẹ ở mức 5\% nếu đánh giá trên rolling target bị overlap.
Diagnostics non-overlapping có Ljung--Box p-value {nonoverlap_recovered_diag["Ljung-Box p-value"]:.4f}, nên không bác bỏ tự tương quan ở mức 5\% trong kiểm tra phụ.
Tuy nhiên, do test set chỉ có 14 ngày nên non-overlapping aggregate chỉ có hai mốc thời gian; kết quả này được dùng như kiểm tra phụ, không phải bằng chứng tuyệt đối.

\begin{{table}}[H]
\centering
\caption{{Diagnostics trên non-overlapping 7-day targets}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(owner_nonoverlap_diagnostics, columns=["Model", "Evaluation mode", "Evaluation dates", "RMSE", "MAE", "WAPE", "WPE", "Ljung-Box p-value", "N", "Aggregate residual points"])}
}}
\end{{table}}

Khoảng dự báo được xây dựng từ phân vị residual trên validation set của mô hình two-stage.
Cách này không giả định residual phân phối chuẩn tuyệt đối, phù hợp hơn với dữ liệu bán lẻ thưa và có nhiều giá trị zero.

{figure(FIGURES_DIR / "owner_two_stage_forecast_interval.png", "Khoảng dự báo 80\\% và 95\\% của mô hình two-stage trên recovered latent demand.", "fig:vi-owner-interval")}

\section{{Diễn giải đặc trưng của mô hình two-stage}}
Với hướng two-stage, feature importance chính được lấy từ mô hình \textbf{{Recovered-demand forecasting}} thay vì LightGBM hourly observed-sales.
Các đặc trưng quan trọng nhất chủ yếu là lag và rolling statistics của recovered demand, vì chúng mô tả mức nền nhu cầu sau khi đã hiệu chỉnh stockout.
Biến \texttt{{stockout\_rate}} vẫn được giữ trong mô hình forecasting để mô hình biết mức độ kiểm duyệt gần đây của chuỗi.

\begin{{table}}[H]
\centering
\caption{{Top đặc trưng quan trọng của recovered-demand forecasting}}
{csv_table(owner_feature_importance, columns=["feature", "importance"], max_rows=15)}
\end{{table}}

{figure(FIGURES_DIR / "owner_recovereddemand_forecasting_feature_importance_top20.png", "Top 20 feature importance của mô hình recovered-demand forecasting.", "fig:vi-owner-feature-importance", r"0.78\linewidth")}


\section{{Phân tích bổ sung trên benchmark observed-sales}}
Các phân tích trong mục này được giữ như sanity check cho pipeline feature engineering ban đầu.
Chúng không phải kết quả chính của đồ án, vì target của chúng vẫn là observed sales.
Tuy nhiên, chúng giúp giải thích vì sao các nhóm đặc trưng thời gian, promotion, weather và stockout-aware vẫn hữu ích trước khi chuyển sang mô hình two-stage.

\subsection{{Ablation study}}
Ablation study kiểm tra giá trị gia tăng của promotion, weather và stockout features trên benchmark LightGBM observed-sales.

\begin{{table}}[H]
\centering
\caption{{Kết quả ablation LightGBM trên observed-sales benchmark}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(ablation, columns=["Horizon", "Feature set", "RMSE", "MAE", "WAPE", "sMAPE", "Train rows used"])}
}}
\end{{table}}

{figure(FIGURES_DIR / "ablation_wape_h1.png", "Ablation theo WAPE cho h=1.", "fig:vi-ablation")}

\subsection{{Phân tích stockout và nhu cầu bị kiểm duyệt}}
Khi hết hàng, doanh số quan sát được có thể thấp hơn nhu cầu thực.
Phân tích lost-sales proxy ban đầu dùng rolling mean 168 giờ để minh họa cơ chế censoring; pipeline chính sau đó thay proxy thô này bằng expanding-window latent demand recovery.

\begin{{table}}[H]
\centering
\caption{{Tóm tắt stockout-censoring và lost-sales proxy}}
{csv_table(stockout_censoring)}
\end{{table}}

{figure(FIGURES_DIR / "stockout_lost_sales_proxy_by_hour.png", "Doanh số quan sát và lost-sales proxy theo giờ.", "fig:vi-lost-sales", r"0.82\linewidth")}

\begin{{table}}[H]
\centering
\caption{{Đánh giá riêng stockout và non-stockout periods}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(stockout_eval, columns=["Horizon", "Model", "Segment", "RMSE", "MAE", "WAPE", "sMAPE", "N"], max_rows=30)}
}}
\end{{table}}

\subsection{{Kiểm tra độ ổn định và mô hình cổ điển}}
Rolling-window evaluation cho thấy hiệu năng LightGBM observed-sales tương đối ổn định giữa hai tuần test.
Ngoài ra, SARIMAX trên chuỗi tổng hợp được dùng như đối chứng time-series cổ điển.
Kết quả cho thấy SARIMAX aggregate không tự động thắng seasonal naive 168h, nhấn mạnh sức mạnh của mùa vụ tuần và sự cần thiết của benchmark.

\begin{{table}}[H]
\centering
\caption{{Rolling-window evaluation trên test set}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(rolling_eval, columns=["Horizon", "Model", "Window start", "Window end", "RMSE", "MAE", "WAPE", "sMAPE", "N"])}
}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{So sánh SARIMAX tổng hợp với seasonal naive}}
{csv_table(sarimax_comparison, columns=["Model", "RMSE", "MAE", "WAPE", "sMAPE", "AIC", "BIC"])}
\end{{table}}

{figure(FIGURES_DIR / "aggregate_sarimax_forecast.png", "Dự báo SARIMAX trên chuỗi tổng hợp observed-sales.", "fig:vi-sarimax")}

\section{{Kết luận và hạn chế}}
Đồ án cuối cùng được tiếp cận theo hướng two-stage: trước hết khôi phục latent demand trong các giờ stockout bằng expanding-window recovery, sau đó forecast trên daily recovered demand.
Khi đánh giá trên recovered latent demand proxy, LightGBM train trên recovered demand đạt WAPE {owner_recovered["WAPE"]:.4f}, tốt hơn mô hình train trên observed sales với WAPE {owner_observed["WAPE"]:.4f}.
Seasonal naive 7 ngày là baseline rất mạnh; vì vậy mô hình kết luận cuối dùng hybrid seasonal-ML chọn trọng số trên validation.
Hybrid đạt WAPE {owner_main["WAPE"]:.4f} và WPE {owner_main["WPE"]:.4f}, là kết quả tốt nhất trong nhóm mô hình/baseline đã thử.
Điều này cho thấy hướng two-stage có ích, nhưng cũng nhấn mạnh rằng seasonality tuần là thành phần nền không thể bỏ qua.

Hạn chế chính là recovered demand vẫn là proxy được ước lượng từ mô hình, không phải ground truth nhu cầu thật.
Ngoài ra, diagnostics chính trên rolling 7-day target vẫn còn tự tương quan nhẹ vì các target liên tiếp overlap mạnh; kiểm tra non-overlapping không bác bỏ tự tương quan nhưng chỉ có ít điểm aggregate.
Jarque--Bera vẫn bác bỏ phân phối chuẩn.
Điều này phù hợp với bản chất dữ liệu bán lẻ thưa, nhiều zero và có spike do khuyến mãi/tồn kho.
Hướng phát triển tiếp theo là thử các mô hình recovery chuyên biệt hơn như SAITS, TimesNet, DLinear hoặc ImputeFormer, đồng thời kiểm tra sensitivity của warmup length và block size.

\end{{document}}
"""
    return content


def main() -> None:
    ensure_directories()
    output = REPORTS_DIR / "fresh50k_report_vi.tex"
    output.write_text(build_report_vi(), encoding="utf-8")
    print(f"Saved Vietnamese LaTeX report: {output}")


if __name__ == "__main__":
    main()
