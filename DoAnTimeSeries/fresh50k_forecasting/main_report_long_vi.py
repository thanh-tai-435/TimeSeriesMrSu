"""Generate a long Vietnamese report package for the Fresh50K project."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from main_report import format_value, latex_escape, read_metric
from src.config import FIGURES_DIR, TABLES_DIR, ensure_directories


REPORT_DIR = Path("report_long")
SECTIONS_DIR = REPORT_DIR / "sections"
REPORT_FIGURES_DIR = REPORT_DIR / "figures"
REPORT_TABLES_DIR = REPORT_DIR / "tables"


FIGURES = [
    "aggregate_sales_over_time.png",
    "stockout_rate_over_time.png",
    "sale_amount_distribution.png",
    "log_sale_amount_distribution.png",
    "sales_by_hour_of_day.png",
    "sales_by_day_of_week.png",
    "representative_series_high_volume.png",
    "representative_series_intermittent.png",
    "representative_series_stockout_heavy.png",
    "acf_aggregate_sales.png",
    "pacf_aggregate_sales.png",
    "acf_high_volume_series.png",
    "pacf_high_volume_series.png",
    "spectrum_hourly_aggregate_sales.png",
    "spectrum_daily_recovered_demand.png",
    "owner_expanding_window_process.png",
    "owner_expanding_window_recovery_detail.png",
    "owner_observed_vs_recovered_demand.png",
    "imputation_daily_uplift.png",
    "imputation_series_lift_distribution.png",
    "imputation_top_series_lift.png",
    "owner_two_stage_forecast_comparison.png",
    "owner_two_stage_diagnostics_wape.png",
    "owner_two_stage_forecast_interval.png",
    "owner_two_stage_residual_acf.png",
    "owner_two_stage_residual_distribution.png",
    "owner_recovereddemand_forecasting_feature_importance_top20.png",
    "owner_two_stage_bias_comparison.png",
    "stockout_lost_sales_proxy_by_hour.png",
    "ablation_wape_h1.png",
    "stockout_segment_wape.png",
    "aggregate_sarimax_forecast.png",
]


TABLES = [
    "data_quality_summary.csv",
    "representative_series.csv",
    "split_summary.csv",
    "owner_daily_split_summary.csv",
    "stationarity_tests.csv",
    "spectrum_hourly_top_peaks.csv",
    "spectrum_daily_recovered_top_peaks.csv",
    "owner_latent_recovery_summary.csv",
    "owner_recovery_blocks.csv",
    "imputation_uplift_summary.csv",
    "imputation_pseudo_stockout_validation.csv",
    "imputation_pseudo_stockout_aggregate_validation.csv",
    "imputation_cap_sensitivity.csv",
    "imputation_series_uplift.csv",
    "owner_two_stage_forecasting_comparison.csv",
    "owner_hybrid_blend_selection.csv",
    "owner_two_stage_diagnostics.csv",
    "owner_two_stage_nonoverlap_diagnostics.csv",
    "owner_recovereddemand_forecasting_feature_importance.csv",
    "model_comparison_h1.csv",
    "model_comparison_h24.csv",
    "ablation_results.csv",
    "stockout_censoring_summary.csv",
    "stockout_segment_evaluation.csv",
    "aggregate_sarimax_comparison.csv",
    "rolling_window_evaluation.csv",
]


def pct(value: float) -> str:
    return f"{value * 100:.2f}\\%"


def metric(path_name: str, name: str) -> str:
    return read_metric(TABLES_DIR / path_name, name)


def tex_table(path_name: str, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    path = REPORT_TABLES_DIR / path_name
    df = pd.read_csv(path)
    if columns:
        df = df[[column for column in columns if column in df.columns]]
    if max_rows:
        df = df.head(max_rows)
    spec = "l" * len(df.columns)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    lines.append(" & ".join(latex_escape(col) for col in df.columns) + r" \\")
    lines.append(r"\midrule")
    for row in df.itertuples(index=False):
        lines.append(" & ".join(latex_escape(format_value(value)) for value in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_block(
    path_name: str,
    caption: str,
    label: str,
    columns: list[str] | None = None,
    max_rows: int | None = None,
    width: str = r"\linewidth",
) -> str:
    return rf"""
\begin{{table}}[H]
\centering
\caption{{{latex_escape(caption)}}}
\label{{{label}}}
\resizebox{{{width}}}{{!}}{{%
{tex_table(path_name, columns=columns, max_rows=max_rows)}
}}
\end{{table}}
"""


def figure_block(path_name: str, caption: str, label: str, width: str = r"0.92\linewidth") -> str:
    return rf"""
\begin{{figure}}[H]
    \centering
    \includegraphics[width={width}]{{figures/{path_name}}}
    \caption{{{latex_escape(caption)}}}
    \label{{{label}}}
\end{{figure}}
"""


def full_page_figure(path_name: str, caption: str, label: str) -> str:
    return rf"""
\clearpage
\begin{{figure}}[p]
    \centering
    \includegraphics[width=0.95\linewidth,height=0.78\textheight,keepaspectratio]{{figures/{path_name}}}
    \caption{{{latex_escape(caption)}}}
    \label{{{label}}}
\end{{figure}}
\clearpage
"""


def copy_assets() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        source = FIGURES_DIR / name
        if source.exists():
            shutil.copy2(source, REPORT_FIGURES_DIR / name)
    for name in TABLES:
        source = TABLES_DIR / name
        if source.exists():
            shutil.copy2(source, REPORT_TABLES_DIR / name)


def write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def build_sections() -> None:
    data_quality = TABLES_DIR / "data_quality_summary.csv"
    owner_cmp = pd.read_csv(TABLES_DIR / "owner_two_stage_forecasting_comparison.csv")
    owner_main = owner_cmp[owner_cmp["Evaluation target"] == "Recovered latent demand proxy"].sort_values("WAPE").iloc[0]
    observed = owner_cmp[
        (owner_cmp["Model"] == "Observed-sales forecasting")
        & (owner_cmp["Evaluation target"] == "Recovered latent demand proxy")
    ].iloc[0]
    recovered = owner_cmp[
        (owner_cmp["Model"] == "Recovered-demand forecasting")
        & (owner_cmp["Evaluation target"] == "Recovered latent demand proxy")
    ].iloc[0]
    seasonal = owner_cmp[owner_cmp["Model"] == "Recovered seasonal naive 7-day"].iloc[0]
    impute_lift = float(metric("imputation_uplift_summary.csv", "Recovered lift over observed"))
    stockout_rate = float(metric("owner_latent_recovery_summary.csv", "Stockout row rate"))
    sample_frac = metric("data_quality_summary.csv", "Sample fraction")
    n_rows = metric("data_quality_summary.csv", "Number of rows")
    n_series = metric("data_quality_summary.csv", "Number of series")
    start_date = metric("data_quality_summary.csv", "Start date")
    end_date = metric("data_quality_summary.csv", "End date")

    write(
        SECTIONS_DIR / "01_introduction.tex",
        rf"""
\section{{Giới thiệu và động lực kinh doanh}}

Bài toán trong đồ án này không chỉ là dự báo doanh số bán được trong tương lai. Trong bán lẻ, doanh số quan sát được có thể bị giới hạn bởi tồn kho: khi một sản phẩm hết hàng, số lượng bán được thấp không phản ánh đầy đủ nhu cầu thật. Nếu doanh nghiệp dùng trực tiếp doanh số quan sát được để dự báo, hệ thống có thể học nhầm rằng nhu cầu thấp, từ đó nhập hàng ít hơn, tiếp tục gây stockout trong các kỳ sau. Đây là vòng lặp vận hành quan trọng:

\begin{{center}}
\texttt{{stockout -> observed sales thấp -> forecast thấp -> nhập ít -> tiếp tục stockout}}
\end{{center}}

Vì vậy, câu hỏi nghiên cứu của đồ án được đặt lại theo hướng gần với nghiệp vụ hơn: làm thế nào để dự báo nhu cầu bán lẻ khi dữ liệu doanh số bị kiểm duyệt bởi trạng thái hết hàng? Thay vì chỉ dự báo \textit{{observed sales}}, báo cáo này xây dựng pipeline dự báo \textit{{recovered demand}}, tức nhu cầu được hiệu chỉnh sau khi xét đến stockout.

\subsection{{Câu hỏi nghiên cứu}}

Báo cáo tập trung vào bốn câu hỏi chính. Thứ nhất, dữ liệu FreshRetailNet-50K có các pattern thời gian nào quan trọng cho dự báo, đặc biệt là seasonality theo giờ, theo ngày và theo tuần? Thứ hai, stockout làm observed sales bị kiểm duyệt ở mức độ nào và có cần latent demand recovery hay không? Thứ ba, recovery demand có làm tổng nhu cầu tăng quá mức hay không, và làm sao kiểm soát rủi ro imputation? Thứ tư, sau khi recover demand, mô hình forecasting nào phù hợp nhất cho bài toán daily next-7-day demand?

\subsection{{Đóng góp chính của đồ án}}

Đóng góp của đồ án được hiểu theo nghĩa thực nghiệm và ứng dụng, không phải đề xuất một thuật toán deep learning hoàn toàn mới. Điểm mới nằm ở cách framing bài toán, cách xử lý stockout và cách biến kết quả forecast thành insight vận hành.

\begin{{table}}[H]
\centering
\caption{{Tóm tắt contribution của đồ án}}
\label{{tab:contribution-summary}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lll}}
\toprule
Nhóm đóng góp & Nội dung thực hiện & Ý nghĩa \\
\midrule
Problem framing & Chuyển từ forecast observed sales sang forecast recovered demand & Gần hơn với quyết định replenishment \\
Data treatment & Recover latent demand ở cấp hourly rồi aggregate lên daily & Giữ đúng nơi stockout xuất hiện nhưng trả forecast ở cấp vận hành \\
Leakage control & Expanding-window recovery theo block tuần & Không dùng ngày tương lai để recover quá khứ \\
Imputation control & Calibration, cap q90, pseudo-stockout validation, sensitivity & Tránh để imputation làm phình tổng demand quá mức \\
Modeling insight & So sánh observed-sales, recovered-demand, seasonal naive và hybrid & Chứng minh seasonality tuần là baseline rất mạnh \\
Business insight & Đo mức demand bị che bởi stockout và tác động lên forecast & Hỗ trợ nhận diện SKU/store có rủi ro thiếu hàng \\
\bottomrule
\end{{tabular}}
}}
\end{{table}}

Nói ngắn gọn, contribution chính không phải là ``dùng LightGBM'' hay ``vẽ thêm nhiều biểu đồ''. Contribution chính là xây dựng một quy trình forecast có xét cơ chế stockout-censoring: observed sales bị kiểm duyệt, demand được recover có kiểm soát, sau đó forecast được đánh giá trên demand proxy phù hợp hơn cho vận hành.

\subsection{{Phạm vi và điều không claim}}

Đồ án không claim rằng recovered demand là ground truth tuyệt đối. Đây là proxy được ước lượng từ dữ liệu non-stockout và được kiểm soát bằng các kiểm tra aggregate/sensitivity. Đồ án cũng không claim đánh bại mọi mô hình deep learning hiện đại. Mục tiêu là một pipeline chặt chẽ, tái lập được, có benchmark rõ ràng và diễn giải được về mặt business.

\subsection{{Kết quả chính ở mức tổng quan}}

Sample thí nghiệm dùng {latex_escape(sample_frac)} theo cấp \texttt{{series\_id}}, gồm {latex_escape(n_rows)} quan sát hourly và {latex_escape(n_series)} chuỗi, từ {latex_escape(start_date)} đến {latex_escape(end_date)}. Tỷ lệ stockout row khoảng {pct(stockout_rate)}. Sau recovery, recovered demand tăng khoảng {pct(impute_lift)} so với observed sales, cho thấy observed sales có khả năng đang understated demand do hết hàng.

Khi đánh giá trên recovered latent demand proxy, observed-sales LightGBM đạt WAPE {pct(float(observed["WAPE"]))}; recovered-demand LightGBM giảm xuống {pct(float(recovered["WAPE"]))}. Seasonal naive 7-day đạt {pct(float(seasonal["WAPE"]))}, cho thấy seasonality tuần rất mạnh. Mô hình kết luận cuối là {latex_escape(owner_main["Model"])}, đạt WAPE {pct(float(owner_main["WAPE"]))}.
""",
    )

    write(
        SECTIONS_DIR / "02_data_preprocessing.tex",
        rf"""
\section{{Dữ liệu và tiền xử lý}}

\subsection{{Nguồn dữ liệu}}

FreshRetailNet-50K là bộ dữ liệu bán lẻ nhiều chuỗi, trong đó mỗi chuỗi tương ứng với một tổ hợp cửa hàng--sản phẩm--thành phố. Dữ liệu gốc gồm các file \texttt{{train.parquet}} và \texttt{{eval.parquet}}. Điểm đặc biệt của bộ dữ liệu là ngoài sales, dữ liệu còn chứa trạng thái tồn kho theo giờ thông qua \texttt{{hours\_stock\_status}}, giúp phân biệt giai đoạn bán thấp vì nhu cầu thấp và giai đoạn bán thấp vì hết hàng.

\subsection{{Tạo định danh chuỗi}}

Biến \texttt{{series\_id}} được tạo từ \texttt{{city\_id}}, \texttt{{store\_id}} và \texttt{{product\_id}}. Việc tạo định danh này giúp mọi phép lag, rolling và split được thực hiện đúng theo từng chuỗi, không trộn thông tin giữa các sản phẩm hoặc cửa hàng khác nhau.

\subsection{{Bung dữ liệu daily thành hourly}}

Dữ liệu sale theo ngày được bung thành hourly bằng \texttt{{hours\_sale}}. Trạng thái stockout theo giờ được lấy từ \texttt{{hours\_stock\_status}}. Việc giữ cấp hourly là cần thiết vì stockout xảy ra theo giờ; nếu aggregate trực tiếp lên daily trước khi recover, ta có thể làm mất thông tin về thời điểm hết hàng trong ngày.

\subsection{{Lấy sample theo series}}

Đồ án không yêu cầu chạy full data. Vì vậy, sample 10\% được lấy ở cấp \texttt{{series\_id}}, giữ nguyên toàn bộ dòng thời gian của các chuỗi được chọn. Cách lấy sample này tốt hơn random row sampling vì không phá vỡ cấu trúc thời gian và không tạo chuỗi bị đứt đoạn.

{table_block("data_quality_summary.csv", "Tóm tắt chất lượng dữ liệu", "tab:long-data-quality", width=r"0.82\linewidth")}

\subsection{{Split theo thời gian}}

Tất cả các bước đánh giá đều dùng split theo thời gian, không dùng random split. Với dữ liệu hourly, validation gồm 7 ngày và test gồm 14 ngày cuối. Với bài toán daily next-7-day forecasting, báo cáo lưu thêm split theo forecast origin để làm rõ ngày dự báo và target window tương ứng.

{table_block("split_summary.csv", "Split hourly theo thời gian", "tab:long-hourly-split", width=r"0.82\linewidth")}

{table_block("owner_daily_split_summary.csv", "Split daily forecasting theo forecast origin", "tab:long-daily-split")}
""",
    )

    write(
        SECTIONS_DIR / "03_eda.tex",
        rf"""
\section{{Phân tích khám phá dữ liệu}}

\subsection{{Aggregate sales và stockout}}

Hai hình đầu tiên cho thấy bức tranh tổng quát của dữ liệu: aggregate observed sales theo thời gian và stockout rate theo thời gian. Đây là hai đồ thị quan trọng nhất để motivate bài toán. Nếu chỉ nhìn aggregate sales, các giai đoạn giảm bán có thể bị hiểu là demand giảm. Nhưng khi đặt cạnh stockout rate, ta thấy cần phân biệt giữa demand thấp và availability thấp.

{figure_block("aggregate_sales_over_time.png", "Tổng observed sales theo giờ.", "fig:long-aggregate-sales")}

{figure_block("stockout_rate_over_time.png", "Tỷ lệ stockout theo giờ.", "fig:long-stockout-rate")}

\subsection{{Phân phối sales}}

Phân phối sale amount cho thấy dữ liệu bán lẻ rất thưa, nhiều giá trị nhỏ và zero, đồng thời có các spike lớn. Đây là lý do không dùng MAPE đơn thuần làm metric chính; WAPE, MAE và RMSE phù hợp hơn vì ổn định hơn khi dữ liệu có nhiều zero.

{figure_block("sale_amount_distribution.png", "Phân phối sale amount hourly.", "fig:long-sale-dist", r"0.78\linewidth")}

{figure_block("log_sale_amount_distribution.png", "Phân phối log1p sale amount hourly.", "fig:long-log-sale-dist", r"0.78\linewidth")}

\subsection{{Seasonality theo giờ và theo tuần}}

Seasonality là insight quan trọng nhất của dữ liệu. Sales thay đổi theo giờ trong ngày và theo thứ trong tuần. Điều này giải thích vì sao seasonal naive theo tuần là baseline rất mạnh ở phần kết quả. Trong bối cảnh bán lẻ, pattern tuần phản ánh hành vi mua sắm lặp lại của khách hàng, lịch hoạt động của cửa hàng và các chu kỳ khuyến mãi.

{figure_block("sales_by_hour_of_day.png", "Sales trung bình theo giờ trong ngày.", "fig:long-hour-seasonality", r"0.78\linewidth")}

{figure_block("sales_by_day_of_week.png", "Sales trung bình theo thứ trong tuần.", "fig:long-dow-seasonality", r"0.78\linewidth")}

\subsection{{Chuỗi đại diện}}

Để không plot hàng nghìn chuỗi, báo cáo chọn ba chuỗi đại diện: high-volume, intermittent và stockout-heavy. Ba nhóm này minh họa ba kiểu khó khác nhau của forecasting: chuỗi lớn có pattern rõ, chuỗi thưa có nhiều zero, và chuỗi stockout-heavy có observed sales bị censor nặng.

{table_block("representative_series.csv", "Ba chuỗi đại diện", "tab:long-representative", width=r"0.86\linewidth")}

{figure_block("representative_series_high_volume.png", "Chuỗi đại diện high-volume.", "fig:long-rep-high")}

{figure_block("representative_series_intermittent.png", "Chuỗi đại diện intermittent.", "fig:long-rep-intermittent")}

{figure_block("representative_series_stockout_heavy.png", "Chuỗi đại diện stockout-heavy.", "fig:long-rep-stockout")}
""",
    )

    write(
        SECTIONS_DIR / "04_time_series_diagnostics.tex",
        rf"""
\section{{Kiểm tra time-series: stationarity, ACF/PACF và spectrum}}

\subsection{{Stationarity testing}}

Theo yêu cầu môn học, báo cáo vẫn thực hiện ADF và KPSS trên aggregate series và các chuỗi đại diện. Tuy nhiên, với bài toán nhiều chuỗi bán lẻ, mục đích của stationarity test không phải để ép toàn bộ pipeline thành ARIMA cổ điển. Thay vào đó, ADF/KPSS giúp nhận diện mức độ trend/seasonality và củng cố quyết định dùng lag, rolling và seasonal baseline.

{table_block("stationarity_tests.csv", "Kết quả ADF/KPSS", "tab:long-stationarity", columns=["Series", "Transformation", "ADF p-value", "KPSS p-value", "Conclusion", "N"])}

\subsection{{ACF/PACF}}

ACF/PACF cho thấy autocorrelation và seasonal dependence ở các lag ngắn và lag theo ngày/tuần. Với dữ liệu hourly, lag 24 và 168 có ý nghĩa tự nhiên. Trong pipeline cuối, recovery được làm ở cấp hourly, còn forecasting được aggregate lên daily và dùng lag 7/14/28 ngày.

{figure_block("acf_aggregate_sales.png", "ACF của aggregate hourly sales.", "fig:long-acf-agg", r"0.84\linewidth")}

{figure_block("pacf_aggregate_sales.png", "PACF của aggregate hourly sales.", "fig:long-pacf-agg", r"0.84\linewidth")}

{figure_block("acf_high_volume_series.png", "ACF của chuỗi high-volume.", "fig:long-acf-high", r"0.84\linewidth")}

{figure_block("pacf_high_volume_series.png", "PACF của chuỗi high-volume.", "fig:long-pacf-high", r"0.84\linewidth")}

\subsection{{Spectrum analysis}}

Periodogram được dùng như một cách kiểm tra độc lập trong miền tần số. Hourly aggregate sales có peak mạnh quanh 24 giờ, còn daily recovered demand có peak gần 7 ngày. Điều này giải thích vì sao seasonal naive 7 ngày không phải baseline yếu mà là benchmark rất khó đánh bại.

{figure_block("spectrum_hourly_aggregate_sales.png", "Spectrum của hourly aggregate observed sales.", "fig:long-spectrum-hourly", r"0.86\linewidth")}

{table_block("spectrum_hourly_top_peaks.csv", "Các peak spectrum hourly", "tab:long-spectrum-hourly")}

{figure_block("spectrum_daily_recovered_demand.png", "Spectrum của daily recovered demand.", "fig:long-spectrum-daily", r"0.86\linewidth")}

{table_block("spectrum_daily_recovered_top_peaks.csv", "Các peak spectrum daily", "tab:long-spectrum-daily")}
""",
    )

    write(
        SECTIONS_DIR / "05_methodology.tex",
        rf"""
\section{{Phương pháp}}

\subsection{{Vì sao không forecast observed sales trực tiếp?}}

Observed sales là biến có thể bị censor bởi tồn kho. Trong giờ stockout, doanh số quan sát được là phần bán được trước khi hết hàng hoặc bằng không, không nhất thiết bằng demand thật. Nếu train model trực tiếp trên observed sales, mô hình có thể học rằng các chuỗi thường hết hàng có demand thấp. Điều này dẫn đến under-forecast và quyết định replenishment thiếu hàng.

\subsection{{Two-stage framework}}

Pipeline chính gồm hai stage. Stage 1 recover latent demand ở cấp hourly cho các giờ stockout. Stage 2 aggregate hourly recovered demand lên daily và forecast tổng nhu cầu 7 ngày tiếp theo. Cách này giữ thông tin stockout chi tiết ở nơi nó xuất hiện, nhưng trả forecast cuối ở cấp daily phù hợp hơn cho quyết định vận hành.

{figure_block("owner_expanding_window_process.png", "Quy trình tổng quát two-stage.", "fig:long-process", r"0.88\linewidth")}

\subsection{{Expanding-window recovery}}

Điểm dễ gây leakage nhất là latent demand recovery. Nếu dùng dữ liệu tương lai để impute quá khứ, forecast sau đó sẽ quá lạc quan. Vì vậy, trong train period, recovery được thực hiện theo block 7 ngày: mỗi block chỉ được recover bằng model học từ non-stockout rows nằm trước block đó. Validation/test được recover bằng final recovery model train trên non-stockout rows của train period.

{figure_block("owner_expanding_window_recovery_detail.png", "Chi tiết expanding-window recovery.", "fig:long-recovery-detail", r"0.90\linewidth")}

\subsection{{Calibration và cap}}

Raw imputed demand có thể overpredict. Do đó, dự đoán raw được hiệu chỉnh bằng calibration factor học từ validation non-stockout rows. Sau đó lost demand dương được cap ở q90 để tránh một số giờ stockout cực trị làm tổng demand tăng quá mức. Công thức triển khai là:

\[
\text{{lost}}_t = \max\left(\frac{{\hat{{y}}^{{raw}}_t}}{{c}} - y^{{obs}}_t, 0\right),
\quad
\text{{lost}}^{{cap}}_t = \min(\text{{lost}}_t, q90),
\quad
y^{{rec}}_t = y^{{obs}}_t + \text{{lost}}^{{cap}}_t.
\]

\subsection{{Daily forecasting target}}

Sau recovery hourly, dữ liệu được aggregate lên daily. Target chính là \texttt{{target\_next7\_recovered\_daily}}, tức tổng recovered demand trong 7 ngày sau forecast origin. Target dạng tổng 7 ngày phù hợp với bài toán replenishment hơn forecast từng giờ riêng lẻ vì quyết định nhập hàng thường dựa trên nhu cầu vài ngày tới.

\subsection{{Modeling choice}}

LightGBM được chọn vì đây là bài toán tabular time-series quy mô lớn với nhiều feature lag, rolling, calendar, stockout, promotion và weather. Classical ARIMA/SARIMA không được chọn làm model chính vì khó scale tới hàng nghìn chuỗi store-product và khó xử lý chuỗi thưa. Deep learning không phải trọng tâm vì yêu cầu tuning/tài nguyên cao hơn, trong khi mục tiêu đồ án là một pipeline có thể giải thích và tái lập.

\subsection{{Baselines và hybrid}}

Báo cáo không chỉ so sánh hai mô hình ML. Các baseline daily gồm naive x7, seasonal naive 7-day và rolling mean 14-day. Seasonal naive 7-day rất mạnh do dữ liệu có seasonality tuần rõ. Vì vậy, mô hình cuối là hybrid seasonal-ML: blend giữa seasonal naive 7-day và recovered-demand LightGBM, với trọng số chọn trên validation set.
""",
    )

    write(
        SECTIONS_DIR / "06_recovery_imputation.tex",
        rf"""
\section{{Latent demand recovery và kiểm soát imputation}}

\subsection{{Kết quả recovery tổng quan}}

Bảng sau tóm tắt kết quả recovery. Recovered demand tăng khoảng {pct(impute_lift)} so với observed sales. Về business, con số này có thể hiểu là phần demand tiềm ẩn mà observed sales không phản ánh đầy đủ do stockout.

{table_block("owner_latent_recovery_summary.csv", "Tóm tắt latent demand recovery", "tab:long-recovery-summary")}

{figure_block("owner_observed_vs_recovered_demand.png", "Observed sales và recovered demand theo thời gian.", "fig:long-observed-recovered")}

\subsection{{Uplift do imputation}}

Vì recovered demand không phải ground truth thật, báo cáo không dùng imputation một cách mù quáng. Uplift được kiểm tra ở nhiều cấp: tổng dataset, từng ngày và từng series. Nếu uplift quá lớn hoặc tập trung ở một số series, kết quả forecast sẽ phụ thuộc quá mạnh vào imputation.

{table_block("imputation_uplift_summary.csv", "Tổng mức uplift do imputation", "tab:long-uplift", width=r"0.86\linewidth")}

{figure_block("imputation_daily_uplift.png", "Observed sales, recovered demand và recovered lost demand theo ngày.", "fig:long-daily-uplift")}

\subsection{{Pseudo-stockout validation}}

Pseudo-stockout validation lấy các dòng non-stockout trong validation period, giả sử chúng cần impute, rồi so sánh prediction của recovery model với observed sales thật. Row-level hourly validation có WAPE cao vì hourly sales rất thưa và nhiều giá trị nhỏ. Do đó, báo cáo bổ sung aggregate validation, phù hợp hơn với mục tiêu forecast daily demand.

{table_block("imputation_pseudo_stockout_validation.csv", "Pseudo-stockout validation ở cấp row hourly", "tab:long-pseudo-row", width=r"0.86\linewidth")}

{table_block("imputation_pseudo_stockout_aggregate_validation.csv", "Pseudo-stockout validation ở các cấp aggregate", "tab:long-pseudo-agg")}

\subsection{{Sensitivity theo cap}}

Cap q90 được chọn như một điểm cân bằng. Cap q75 bảo thủ hơn nhưng có thể under-recover; q100 giữ toàn bộ raw lost demand nhưng dễ phụ thuộc mạnh vào outlier. Sensitivity giúp chứng minh kết luận không chỉ dựa trên một lựa chọn cap tùy tiện.

{table_block("imputation_cap_sensitivity.csv", "Sensitivity theo mức cap lost demand", "tab:long-cap-sensitivity")}

{figure_block("imputation_series_lift_distribution.png", "Phân phối recovered lift ở cấp series.", "fig:long-lift-dist", r"0.78\linewidth")}

{figure_block("imputation_top_series_lift.png", "Top series có lift cao nhất.", "fig:long-top-lift", r"0.78\linewidth")}
""",
    )

    write(
        SECTIONS_DIR / "07_forecasting_results.tex",
        rf"""
\section{{Forecasting results}}

\subsection{{So sánh không impute và có impute}}

So sánh quan trọng nhất là giữa observed-sales forecasting và recovered-demand forecasting. Observed-sales LightGBM đại diện cho hướng không xử lý stockout trong target. Recovered-demand LightGBM đại diện cho hướng có recovery/imputation. Khi đánh giá trên recovered latent demand proxy, WAPE giảm từ {pct(float(observed["WAPE"]))} xuống {pct(float(recovered["WAPE"]))}. Điều này cho thấy stockout-aware recovery giúp giảm under-forecast.

{table_block("owner_two_stage_forecasting_comparison.csv", "So sánh mô hình/baseline trên recovered latent demand proxy", "tab:long-main-comparison", columns=["Model", "Training target", "Evaluation target", "RMSE", "MAE", "WAPE", "sMAPE", "WPE", "N"])}

{figure_block("owner_two_stage_forecast_comparison.png", "Forecast tổng nhu cầu 7 ngày.", "fig:long-forecast-comparison")}

{figure_block("owner_two_stage_diagnostics_wape.png", "WAPE của các mô hình/baseline trên recovered latent demand proxy.", "fig:long-wape")}

\subsection{{Seasonal naive là baseline mạnh}}

Seasonal naive 7-day đạt WAPE {pct(float(seasonal["WAPE"]))}, tốt hơn recovered-demand LightGBM thuần. Đây không phải thất bại của ML mà là insight quan trọng: pattern tuần là thành phần nền rất mạnh. Một mô hình tốt trong bài toán này cần tôn trọng seasonality thay vì cố học lại hoàn toàn từ đầu.

\subsection{{Hybrid seasonal-ML}}

Hybrid seasonal-ML blend seasonal naive 7-day với recovered-demand LightGBM. Trọng số LightGBM được chọn trên validation set, không chọn bằng test. Trọng số tốt nhất hiện tại là 0.60 cho LightGBM và 0.40 cho seasonal naive. Kết quả test đạt WAPE {pct(float(owner_main["WAPE"]))}, tốt nhất trong nhóm đã thử.

{table_block("owner_hybrid_blend_selection.csv", "Chọn trọng số hybrid trên validation", "tab:long-hybrid-blend", max_rows=10, width=r"0.80\linewidth")}

\subsection{{Business interpretation}}

Ý nghĩa của kết quả không chỉ là giảm WAPE. Nếu doanh nghiệp forecast observed sales, model có thể tiếp tục đánh giá thấp nhu cầu ở các SKU hay hết hàng. Recovery làm rõ phần demand bị ẩn, còn seasonal baseline cho thấy lịch tuần là thông tin vận hành rất quan trọng. Hybrid cuối cùng kết hợp cả hai: demand correction và weekly seasonality.

\subsection{{Feature importance}}

Feature importance của recovered-demand LightGBM cho thấy các lag và rolling features của recovered demand, cùng với stockout rate, là các tín hiệu quan trọng. Điều này phù hợp với trực giác: demand tương lai phụ thuộc mạnh vào mức nền gần đây, pattern tuần và mức độ kiểm duyệt bởi tồn kho.

{table_block("owner_recovereddemand_forecasting_feature_importance.csv", "Feature importance của recovered-demand LightGBM", "tab:long-feature-importance", max_rows=15, width=r"0.75\linewidth")}

{figure_block("owner_recovereddemand_forecasting_feature_importance_top20.png", "Top 20 feature importance.", "fig:long-feature-importance", r"0.78\linewidth")}
""",
    )

    write(
        SECTIONS_DIR / "08_diagnostics.tex",
        rf"""
\section{{Diagnostics và forecast uncertainty}}

\subsection{{Residual diagnostics}}

Diagnostics được thực hiện trên recovered latent demand proxy. Với target là tổng 7 ngày tiếp theo, các forecast origin liên tiếp có target window overlap mạnh: forecast tại ngày $t$ và ngày $t+1$ chia sẻ 6/7 ngày trong target. Vì vậy, residual rolling evaluation có thể còn autocorrelation một phần do thiết kế target, không nhất thiết chỉ do model thiếu pattern.

{table_block("owner_two_stage_diagnostics.csv", "Diagnostics trên recovered latent demand proxy", "tab:long-diagnostics", columns=["Model", "RMSE", "MAE", "WAPE", "WPE", "Residual mean", "Residual std", "Ljung-Box p-value", "Jarque-Bera p-value", "N"])}

{figure_block("owner_two_stage_residual_distribution.png", "Phân phối residual của mô hình chính.", "fig:long-residual-dist", r"0.78\linewidth")}

{figure_block("owner_two_stage_residual_acf.png", "Residual ACF của mô hình chính.", "fig:long-residual-acf", r"0.82\linewidth")}

\subsection{{Non-overlap diagnostics}}

Báo cáo bổ sung non-overlap diagnostics bằng cách lấy forecast origin cách nhau 7 ngày. Kiểm tra này công bằng hơn với target next-7-day, nhưng test window chỉ có 14 ngày nên chỉ có rất ít aggregate points. Vì vậy, đây là sanity check phụ, không phải bằng chứng tuyệt đối rằng residual độc lập hoàn toàn.

{table_block("owner_two_stage_nonoverlap_diagnostics.csv", "Diagnostics trên non-overlapping 7-day targets", "tab:long-nonoverlap", columns=["Model", "Evaluation dates", "RMSE", "MAE", "WAPE", "WPE", "Ljung-Box p-value", "N", "Aggregate residual points"])}

\subsection{{Prediction intervals}}

Khoảng dự báo được xây dựng từ phân vị residual validation. Cách này không giả định residual chuẩn tuyệt đối, phù hợp hơn với dữ liệu bán lẻ thưa và nhiều spike.

{figure_block("owner_two_stage_forecast_interval.png", "Khoảng dự báo 80\\% và 95\\% cho mô hình chính.", "fig:long-forecast-interval")}
""",
    )

    write(
        SECTIONS_DIR / "09_business_discussion.tex",
        rf"""
\section{{Business insights}}

\subsection{{Insight 1: Stockout làm sai lệch tín hiệu demand}}

Tỷ lệ stockout row khoảng {pct(stockout_rate)} cho thấy đây không phải hiện tượng hiếm. Nếu doanh nghiệp chỉ quan sát sales, một phần demand bị ẩn sẽ bị hiểu nhầm là demand thấp. Trong vận hành tồn kho, đây là vấn đề nghiêm trọng vì forecast thấp dẫn tới reorder thấp.

\subsection{{Insight 2: Latent demand recovery cho biết mức demand bị bỏ lỡ}}

Recovered demand tăng khoảng {pct(impute_lift)} so với observed sales. Con số này không nên hiểu là doanh nghiệp chắc chắn mất đúng từng đó doanh số, vì recovered demand là proxy. Tuy nhiên, nó là chỉ báo hữu ích: observed sales có khả năng đang understated demand ở mức đáng kể.

\subsection{{Insight 3: Weekly seasonality là baseline nghiệp vụ mạnh}}

Seasonal naive 7-day rất mạnh. Về business, điều này nghĩa là nhu cầu trong bán lẻ có lịch tuần rõ ràng. Khi triển khai hệ thống forecast, không nên bỏ qua các quy tắc đơn giản như lấy demand cùng kỳ tuần trước. ML chỉ có ý nghĩa khi nó cải thiện trên nền seasonality mạnh đó hoặc kết hợp tốt với nó.

\subsection{{Insight 4: Model cuối nên hỗ trợ replenishment}}

Mục tiêu cuối không phải tạo model có metric đẹp nhất trong phòng thí nghiệm, mà là hỗ trợ quyết định nhập hàng. Forecast recovered demand phù hợp hơn observed sales vì nó hướng tới nhu cầu vận hành. Hybrid seasonal-ML có thể xem là một forecast thực dụng: dùng seasonality làm nền, dùng ML để hiệu chỉnh theo stockout và pattern gần đây.

\subsection{{Khuyến nghị triển khai}}

Nếu triển khai thực tế, hệ thống nên báo cáo song song observed sales forecast và recovered demand forecast. Khi hai giá trị lệch nhau lớn ở các SKU có stockout cao, đây là tín hiệu cần kiểm tra replenishment. Doanh nghiệp cũng nên theo dõi uplift do imputation theo SKU/store để phát hiện nhóm sản phẩm có rủi ro mất doanh số do hết hàng.
""",
    )

    write(
        SECTIONS_DIR / "10_conclusion.tex",
        rf"""
\section{{Kết luận, hạn chế và hướng phát triển}}

\subsection{{Kết luận}}

Báo cáo cho thấy bài toán forecasting trên FreshRetailNet-50K nên được framing như bài toán stockout-aware demand forecasting thay vì raw sales forecasting. Khi không xử lý stockout, observed-sales LightGBM đạt WAPE {pct(float(observed["WAPE"]))}. Sau khi recover latent demand và train trên recovered demand, WAPE giảm xuống {pct(float(recovered["WAPE"]))}. Seasonal naive 7-day là baseline mạnh với WAPE {pct(float(seasonal["WAPE"]))}. Mô hình cuối hybrid seasonal-ML đạt WAPE {pct(float(owner_main["WAPE"]))}.

Kết quả này có hai thông điệp chính. Thứ nhất, xử lý stockout là cần thiết vì observed sales có thể understates demand. Thứ hai, seasonality tuần là pattern nền cực mạnh; một mô hình forecasting tốt nên kết hợp seasonality này thay vì bỏ qua.

\subsection{{Hạn chế}}

Recovered demand vẫn là proxy được ước lượng, không phải ground truth demand thật. Imputation ở cấp hourly row-level nhiễu vì dữ liệu thưa và nhiều zero. Test window chỉ kéo dài 14 ngày nên diagnostics non-overlap còn ít điểm. Ngoài ra, báo cáo chưa trực tiếp tối ưu inventory policy như safety stock, reorder point, service level hoặc lost-sales cost.

\subsection{{Hướng phát triển}}

Nếu có thêm thời gian, hướng phát triển đầu tiên là kiểm tra pipeline trên nhiều sample hoặc full data. Hướng thứ hai là thử các mô hình imputation chuyên biệt như SAITS, TimesNet hoặc DLinear ở quy mô nhỏ để so sánh với LightGBM recovery. Hướng thứ ba là chuyển từ forecast accuracy sang inventory impact: service level, fill rate, lost sales và holding cost. Đây là bước quan trọng để biến forecast thành quyết định vận hành.
""",
    )

    write(
        SECTIONS_DIR / "appendix.tex",
        rf"""
\appendix
\section{{Phụ lục A: Bảng hỗ trợ trực tiếp}}

Phụ lục chỉ giữ các bảng hỗ trợ trực tiếp cho contribution chính: recovery không leakage, kiểm soát imputation, diagnostics phụ và khả năng tái lập. Các benchmark observed-sales hourly, ablation cũ, SARIMAX aggregate và các bảng thử nghiệm phụ được giữ trong thư mục \texttt{{archive\_auxiliary/}} thay vì đưa vào báo cáo chính, để tránh làm loãng câu chuyện.

{table_block("owner_recovery_blocks.csv", "Các block expanding-window dùng cho recovery", "tab:app-recovery-blocks", columns=["Block start", "Block end", "Training end", "Training rows", "Predicted rows"])}

{table_block("imputation_series_uplift.csv", "Top series theo recovered lift", "tab:app-series-uplift", columns=["series_id", "observed_sales", "recovered_demand", "recovered_lost_demand", "stockout_rate", "Recovered lift over observed", "Recovered share from imputation"], max_rows=25)}

{table_block("owner_two_stage_nonoverlap_diagnostics.csv", "Diagnostics phụ trên non-overlapping target", "tab:app-nonoverlap", columns=["Model", "Evaluation dates", "RMSE", "MAE", "WAPE", "WPE", "Ljung-Box p-value", "N", "Aggregate residual points"])}

\section{{Phụ lục B: Reproducibility}}

Pipeline chính có thể chạy bằng:

\begin{{verbatim}}
pip install -r requirements.txt
python main_pipeline.py
\end{{verbatim}}

Các output chính nằm trong \texttt{{deliverables/}} và \texttt{{outputs/}}. Dữ liệu raw cần đặt trong \texttt{{data/raw/}}. Random seed được cố định là 42 trong các mô hình LightGBM và các bước sampling.
""",
    )


def build_main_tex() -> str:
    return r"""
\documentclass[12pt,a4paper]{report}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{longtable}
\usepackage{array}
\usepackage{caption}
\usepackage{hyperref}
\usepackage{setspace}
\usepackage{amsmath}
\usepackage{tocloft}
\onehalfspacing
\setlength{\parskip}{0.45em}
\setlength{\parindent}{1.2em}

\title{Dự báo nhu cầu bán lẻ có xét đến stockout trên FreshRetailNet-50K}
\author{}
\date{}

\begin{document}
\pagenumbering{roman}
\maketitle

\begin{abstract}
Báo cáo này trình bày một pipeline stockout-aware demand forecasting cho FreshRetailNet-50K. Thay vì dự báo trực tiếp observed sales, đồ án recover latent demand trong các giờ stockout, aggregate lên daily demand, rồi forecast tổng nhu cầu 7 ngày tiếp theo. Trọng tâm của báo cáo là tính đúng đắn theo thời gian, kiểm soát imputation, so sánh với benchmark và diễn giải business.
\end{abstract}

\tableofcontents
\listoffigures
\listoftables
\clearpage
\pagenumbering{arabic}

\input{sections/01_introduction}
\input{sections/02_data_preprocessing}
\input{sections/03_eda}
\input{sections/04_time_series_diagnostics}
\input{sections/05_methodology}
\input{sections/06_recovery_imputation}
\input{sections/07_forecasting_results}
\input{sections/08_diagnostics}
\input{sections/09_business_discussion}
\input{sections/10_conclusion}
\input{sections/appendix}

\end{document}
""".strip()


def main() -> None:
    ensure_directories()
    copy_assets()
    build_sections()
    write(REPORT_DIR / "main.tex", build_main_tex())
    print(f"Saved long report package: {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
