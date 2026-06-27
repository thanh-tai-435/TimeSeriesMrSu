"""Generate Vietnamese Beamer slides for the Fresh50K project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from main_report import latex_escape, read_metric
from src.config import REPORTS_DIR, TABLES_DIR, ensure_directories


def pct(value: float) -> str:
    return f"{value * 100:.2f}\\%"


def num(value: float) -> str:
    return f"{value:,.2f}"


def build_slides() -> str:
    data_quality = TABLES_DIR / "data_quality_summary.csv"
    owner_recovery = TABLES_DIR / "owner_latent_recovery_summary.csv"
    imputation_uplift = TABLES_DIR / "imputation_uplift_summary.csv"
    imputation_sensitivity = TABLES_DIR / "imputation_cap_sensitivity.csv"
    owner_forecast = TABLES_DIR / "owner_two_stage_forecasting_comparison.csv"
    owner_diagnostics = TABLES_DIR / "owner_two_stage_diagnostics.csv"
    nonoverlap = TABLES_DIR / "owner_two_stage_nonoverlap_diagnostics.csv"

    sample_frac = read_metric(data_quality, "Sample fraction")
    n_rows = read_metric(data_quality, "Number of rows")
    n_series = read_metric(data_quality, "Number of series")
    start_date = read_metric(data_quality, "Start date")
    end_date = read_metric(data_quality, "End date")

    observed_sales = float(read_metric(owner_recovery, "Observed sales"))
    recovered_demand = float(read_metric(owner_recovery, "Recovered latent demand"))
    recovered_lost = float(read_metric(owner_recovery, "Recovered lost demand"))
    stockout_rate = float(read_metric(owner_recovery, "Stockout row rate"))
    lift = float(read_metric(imputation_uplift, "Recovered lift over observed"))
    raw_lost = float(read_metric(imputation_uplift, "Raw calibrated lost demand before cap"))
    cap_share = float(read_metric(imputation_uplift, "Cap retained share of raw lost demand"))
    calibration_factor = float(read_metric(owner_recovery, "Imputation calibration factor"))
    lost_cap = float(read_metric(owner_recovery, "Lost demand cap value"))

    sens = pd.read_csv(imputation_sensitivity)
    sens_q75 = sens[sens["Scenario"].str.contains("q75", regex=False)].iloc[0]
    sens_q90 = sens[sens["Scenario"].str.contains("q90", regex=False)].iloc[0]
    sens_q100 = sens[sens["Scenario"].str.contains("q100", regex=False)].iloc[0]

    cmp = pd.read_csv(owner_forecast)
    recovered_target_cmp = cmp[cmp["Evaluation target"] == "Recovered latent demand proxy"]
    main_model = recovered_target_cmp.sort_values("WAPE").iloc[0]
    seasonal_model = recovered_target_cmp[recovered_target_cmp["Model"] == "Recovered seasonal naive 7-day"].iloc[0]
    obs_on_rec = cmp[
        (cmp["Model"] == "Observed-sales forecasting")
        & (cmp["Evaluation target"] == "Recovered latent demand proxy")
    ].iloc[0]
    rec_on_rec = cmp[
        (cmp["Model"] == "Recovered-demand forecasting")
        & (cmp["Evaluation target"] == "Recovered latent demand proxy")
    ].iloc[0]
    rel_improve = (obs_on_rec["WAPE"] - main_model["WAPE"]) / obs_on_rec["WAPE"]

    diag = pd.read_csv(owner_diagnostics)
    rec_diag = diag[
        (diag["Model"] == main_model["Model"])
        & (diag["Evaluation target"] == "Recovered latent demand proxy")
    ].iloc[0]
    nonoverlap_rec = pd.read_csv(nonoverlap)
    nonoverlap_rec = nonoverlap_rec[nonoverlap_rec["Model"] == main_model["Model"]].iloc[0]

    return rf"""\documentclass[aspectratio=169,10pt]{{beamer}}
\usetheme{{Madrid}}
\usecolortheme{{default}}
\usepackage{{fontspec}}
\setmainfont{{Times New Roman}}
\setsansfont{{Arial}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{array}}
\graphicspath{{{{../figures/}}}}
\setbeamertemplate{{navigation symbols}}{{}}
\setbeamertemplate{{caption}}[numbered]
\setbeamertemplate{{itemize items}}[circle]
\definecolor{{FreshGreen}}{{RGB}}{{31,121,74}}
\setbeamercolor{{structure}}{{fg=FreshGreen}}
\setbeamercolor{{frametitle}}{{fg=white,bg=FreshGreen}}
\setbeamercolor{{title}}{{fg=white,bg=FreshGreen}}

\title[FreshRetailNet-50K]{{Dự báo nhu cầu bán lẻ có xét stockout}}
\subtitle{{Two-stage latent demand recovery + daily forecasting trên FreshRetailNet-50K}}
\author{{}}
\date{{}}

\begin{{document}}

\begin{{frame}}
    \titlepage
\end{{frame}}

\begin{{frame}}{{Framing vấn đề}}
    \begin{{block}}{{Không chỉ là bài toán dự báo sales}}
    Observed sales trong bán lẻ có thể bị \textbf{{censor}} bởi stockout: bán thấp chưa chắc là nhu cầu thấp, mà có thể là không còn hàng để bán.
    \end{{block}}
    \vspace{{0.15cm}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Nếu forecast trực tiếp observed sales}}
            \begin{{itemize}}
                \item Model học cả nhu cầu thật lẫn giới hạn tồn kho.
                \item Dễ under-forecast ở nhóm hay hết hàng.
                \item Sai với mục tiêu vận hành: cần dự báo demand để chuẩn bị hàng.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Hướng của đồ án}}
            \begin{{itemize}}
                \item Recover latent demand ở giờ stockout.
                \item Aggregate hourly recovered demand lên daily.
                \item Forecast tổng nhu cầu 7 ngày tiếp theo.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Câu hỏi nghiên cứu}}
    \begin{{enumerate}}
        \item Có thể khôi phục demand bị ẩn trong các giờ stockout mà không leakage theo thời gian không?
        \item Imputation có làm tổng demand tăng quá mức hay không?
        \item Forecast trên recovered demand có tốt hơn forecast trên observed sales không?
    \end{{enumerate}}
    \vspace{{0.25cm}}
    \begin{{block}}{{Metric chính}}
    WAPE = $\sum |y-\hat{{y}}| / \sum |y|$. Với bài toán này, WAPE cho biết tổng sai số tuyệt đối chiếm bao nhiêu phần trăm tổng nhu cầu thật.
    \end{{block}}
\end{{frame}}

\begin{{frame}}{{Dữ liệu và thách thức}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.48\linewidth}}
            \begin{{tabular}}{{lr}}
                \toprule
                Mẫu sử dụng & {latex_escape(sample_frac)} \\
                Số quan sát hourly & {latex_escape(n_rows)} \\
                Số chuỗi & {latex_escape(n_series)} \\
                Bắt đầu & {latex_escape(start_date)} \\
                Kết thúc & {latex_escape(end_date)} \\
                Stockout row rate & {pct(stockout_rate)} \\
                \bottomrule
            \end{{tabular}}
            \vspace{{0.25cm}}
            \begin{{itemize}}
                \item Nhiều chuỗi store--product.
                \item Demand thưa, nhiều zero và spike.
                \item Có promotion, weather, calendar và stockout.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.50\linewidth}}
            \includegraphics[width=\linewidth]{{stockout_rate_over_time.png}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Vì sao không dùng mô hình cổ điển làm chính?}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.5\linewidth}}
            \textbf{{ARIMA/SARIMA riêng từng chuỗi}}
            \begin{{itemize}}
                \item Khó scale cho nhiều store--product.
                \item Không ổn với chuỗi thưa và nhiều zero.
                \item Khó đưa đầy đủ promotion, weather, stockout.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.5\linewidth}}
            \textbf{{Deep learning}}
            \begin{{itemize}}
                \item Có thể dùng, nhưng tốn tuning và tài nguyên.
                \item Khó bảo vệ nếu dữ liệu/validation chưa thật chặt.
                \item Đồ án ưu tiên pipeline rõ, tái lập, giải thích được.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
    \vspace{{0.2cm}}
    \begin{{block}}{{Lựa chọn chính}}
    LightGBM toàn cục: học chung pattern giữa nhiều chuỗi, xử lý phi tuyến tốt, chạy nhẹ, và dễ kết hợp feature engineering.
    \end{{block}}
\end{{frame}}

\begin{{frame}}{{Pipeline tổng quát}}
    \centering
    \includegraphics[width=0.92\linewidth,height=0.78\textheight,keepaspectratio]{{owner_expanding_window_process.png}}
\end{{frame}}

\begin{{frame}}{{Latent demand recovery không leakage}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.52\linewidth}}
            \begin{{itemize}}
                \item Hourly stockout được recover trước.
                \item Train period dùng expanding window theo block 7 ngày.
                \item Mỗi block chỉ học từ non-stockout rows trong quá khứ.
                \item Warmup 14 ngày để có đủ hai chu kỳ tuần.
                \item Validation/test chỉ dùng model fit từ train-period non-stockout rows.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.46\linewidth}}
            \includegraphics[width=\linewidth,height=0.66\textheight,keepaspectratio]{{owner_expanding_window_recovery_detail.png}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Imputation được kiểm soát như thế nào?}}
    \begin{{block}}{{Công thức thực tế}}
    Raw impute $\rightarrow$ chia calibration factor $\rightarrow$ lấy lost demand dương $\rightarrow$ cap q90 $\rightarrow$ cộng vào observed sales.
    \end{{block}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.48\linewidth}}
            \begin{{tabular}}{{lr}}
                \toprule
                Calibration factor & {calibration_factor:.4f} \\
                Lost demand cap q90 & {lost_cap:.4f} \\
                Observed sales & {num(observed_sales)} \\
                Recovered demand & {num(recovered_demand)} \\
                Recovered lost demand & {num(recovered_lost)} \\
                Lift & {pct(lift)} \\
                \bottomrule
            \end{{tabular}}
        \end{{column}}
        \begin{{column}}{{0.48\linewidth}}
            \begin{{itemize}}
                \item Raw calibrated lost demand trước cap: {num(raw_lost)}.
                \item Cap q90 giữ lại {pct(cap_share)} phần lost demand raw.
                \item Mục tiêu: không để impute quyết định toàn bộ kết quả.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Sensitivity của imputation}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.45\linewidth}}
            \begin{{tabular}}{{lrr}}
                \toprule
                Kịch bản & Lift & Demand \\
                \midrule
                Cap q75 & {pct(float(sens_q75["Recovered lift over observed"]))} & {num(float(sens_q75["Recovered demand"]))} \\
                Cap q90 & {pct(float(sens_q90["Recovered lift over observed"]))} & {num(float(sens_q90["Recovered demand"]))} \\
                Cap q100 & {pct(float(sens_q100["Recovered lift over observed"]))} & {num(float(sens_q100["Recovered demand"]))} \\
                \bottomrule
            \end{{tabular}}
            \vspace{{0.25cm}}
            \begin{{itemize}}
                \item q90 là lựa chọn cân bằng.
                \item q100 làm lift cao hơn nhưng rủi ro phụ thuộc impute.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.52\linewidth}}
            \includegraphics[width=\linewidth]{{imputation_daily_uplift.png}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Forecasting setup}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Target chính}}
            \begin{{itemize}}
                \item Hourly recovery.
                \item Aggregate lên daily demand.
                \item Forecast tổng demand 7 ngày tiếp theo.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{So sánh hai hướng}}
            \begin{{itemize}}
                \item Train trên observed sales.
                \item Train trên recovered demand.
                \item Seasonal naive 7 ngày và hybrid seasonal-ML.
                \item Cùng đánh giá trên recovered latent demand proxy.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
    \vspace{{0.25cm}}
    \begin{{block}}{{Baseline phù hợp}}
    Seasonal naive là baseline chính vì dữ liệu có chu kỳ tuần rõ; naive/drift/average dùng như đối chứng phụ.
    \end{{block}}
\end{{frame}}

\begin{{frame}}{{Kết quả chính}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.48\linewidth}}
            \begin{{tabular}}{{lrr}}
                \toprule
                Mô hình & WAPE & WPE \\
                \midrule
                Observed-sales & {pct(float(obs_on_rec["WAPE"]))} & {pct(float(obs_on_rec["WPE"]))} \\
                Seasonal naive 7-day & {pct(float(seasonal_model["WAPE"]))} & {pct(float(seasonal_model["WPE"]))} \\
                Recovered LightGBM & {pct(float(rec_on_rec["WAPE"]))} & {pct(float(rec_on_rec["WPE"]))} \\
                Seasonal-ML hybrid & {pct(float(main_model["WAPE"]))} & {pct(float(main_model["WPE"]))} \\
                \bottomrule
            \end{{tabular}}
            \vspace{{0.25cm}}
            \begin{{itemize}}
                \item WAPE giảm tương đối {pct(float(rel_improve))} so với observed-sales.
                \item Seasonal naive rất mạnh; hybrid giữ mùa vụ tuần và thêm hiệu chỉnh ML.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.50\linewidth}}
            \includegraphics[width=\linewidth]{{owner_two_stage_forecast_comparison.png}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Diagnostics và diễn giải}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.50\linewidth}}
            \begin{{itemize}}
                \item Rolling 7-day targets bị overlap mạnh.
                \item Model diagnostics: {latex_escape(str(main_model["Model"]))}.
                \item Ljung--Box p-value = {float(rec_diag["Ljung-Box p-value"]):.4f}, còn tự tương quan nhẹ.
                \item Non-overlap check: p-value = {float(nonoverlap_rec["Ljung-Box p-value"]):.4f}, không bác bỏ tự tương quan.
                \item Nhưng non-overlap chỉ có {int(nonoverlap_rec["Aggregate residual points"])} điểm aggregate, nên chỉ là kiểm tra phụ.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.48\linewidth}}
            \includegraphics[width=\linewidth]{{owner_two_stage_residual_acf.png}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Thông điệp bảo vệ}}
    \begin{{enumerate}}
        \item Framing đúng hơn: forecast demand, không chỉ forecast sales bị giới hạn bởi tồn kho.
        \item Recovery hourly trước, forecast daily sau: phù hợp với bản chất stockout theo giờ và mục tiêu dự báo vận hành.
        \item Imputation có kiểm soát: calibration + cap + sensitivity, lift cuối {pct(lift)}.
        \item Two-stage có lợi: WAPE giảm từ {pct(float(obs_on_rec["WAPE"]))} xuống {pct(float(main_model["WAPE"]))}.
        \item Hạn chế được nói rõ: recovered demand là proxy, residual còn nhiễu, deep imputation có thể là hướng mở rộng.
    \end{{enumerate}}
\end{{frame}}

\begin{{frame}}{{Kết luận}}
    \begin{{block}}{{Kết luận chính}}
    Với dữ liệu bán lẻ có stockout, forecast trực tiếp observed sales dễ học sai nhu cầu. Pipeline two-stage giúp tách bài toán thành recovery demand bị censor và forecast trên demand đã hiệu chỉnh.
    \end{{block}}
    \vspace{{0.2cm}}
    \begin{{itemize}}
        \item Kết quả không chỉ tốt hơn về metric mà còn tốt hơn về lập luận.
        \item WAPE {pct(float(main_model["WAPE"]))} là mức chấp nhận được cho multi-series retail thưa, có stockout.
        \item Điểm mạnh của đồ án là quy trình tránh leakage và kiểm tra imputation rõ ràng.
    \end{{itemize}}
\end{{frame}}

\end{{document}}
"""


def main() -> None:
    ensure_directories()
    tex_path = REPORTS_DIR / "fresh50k_slides_vi.tex"
    tex_path.write_text(build_slides(), encoding="utf-8")
    print(f"Saved Vietnamese Beamer slides: {tex_path}")


if __name__ == "__main__":
    main()
