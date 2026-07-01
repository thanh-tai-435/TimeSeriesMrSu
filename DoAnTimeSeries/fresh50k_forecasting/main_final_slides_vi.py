from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TABLE_DIR = BASE_DIR / "outputs" / "tables"
REPORT_DIR = BASE_DIR / "deliverables" / "reports"
TEX_PATH = REPORT_DIR / "fresh50k_final_presentation_vi.tex"
PDF_PATH = REPORT_DIR / "fresh50k_final_presentation_vi.pdf"


def read_metric(filename: str, metric: str) -> str:
    df = pd.read_csv(TABLE_DIR / filename)
    key = "Metric" if "Metric" in df.columns else "metric"
    value = "Value" if "Value" in df.columns else "value"
    return str(df.loc[df[key] == metric, value].iloc[0])


def pct(value: float | str, digits: int = 2) -> str:
    return f"{float(value) * 100:.{digits}f}\\%"


def num(value: float | str, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def integer(value: float | str) -> str:
    return f"{int(float(value)):,}"


def tex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def metric_bundle() -> dict[str, object]:
    quality = "data_quality_summary.csv"
    recovery = "owner_latent_recovery_summary.csv"
    forecast = pd.read_csv(TABLE_DIR / "owner_two_stage_forecasting_comparison.csv")
    forecast = forecast[forecast["Evaluation target"] == "Recovered latent demand proxy"].copy()
    diag = pd.read_csv(TABLE_DIR / "owner_two_stage_diagnostics.csv")
    intervals = pd.read_csv(TABLE_DIR / "owner_two_stage_interval_coverage_summary.csv")
    pseudo = pd.read_csv(TABLE_DIR / "imputation_pseudo_stockout_aggregate_validation.csv")
    sensitivity = pd.read_csv(TABLE_DIR / "imputation_cap_sensitivity.csv")
    blend = pd.read_csv(TABLE_DIR / "owner_hybrid_blend_selection.csv").iloc[0]
    spectrum = pd.read_csv(TABLE_DIR / "spectrum_daily_recovered_top_peaks.csv").iloc[0]

    def model_row(name: str) -> pd.Series:
        return forecast[forecast["Model"] == name].iloc[0]

    hybrid = model_row("Recovered seasonal-ML hybrid")
    observed = model_row("Observed-sales forecasting")
    seasonal = model_row("Recovered seasonal naive 7-day")
    recovered_ml = model_row("Recovered-demand forecasting")
    naive = model_row("Recovered naive x7")
    rolling = model_row("Recovered rolling mean 14-day")
    hybrid_diag = diag[diag["Model"] == "Recovered seasonal-ML hybrid"].iloc[0]
    hybrid_interval = intervals[intervals["Model"] == "Recovered seasonal-ML hybrid"].iloc[0]
    pseudo_daily = pseudo[pseudo["Validation level"] == "Daily aggregate"].iloc[0]
    q90 = sensitivity[sensitivity["Scenario"].str.contains("q90", regex=False)].iloc[0]
    q100 = sensitivity[sensitivity["Scenario"].str.contains("q100", regex=False)].iloc[0]

    return {
        "sample_frac": read_metric(quality, "Sample fraction"),
        "n_rows": read_metric(quality, "Number of rows"),
        "n_series": read_metric(quality, "Number of series"),
        "n_stores": read_metric(quality, "Number of stores"),
        "n_products": read_metric(quality, "Number of products"),
        "start_date": read_metric(quality, "Start date"),
        "end_date": read_metric(quality, "End date"),
        "stockout_rate_quality": read_metric(quality, "Stockout rate"),
        "observed_sales": read_metric(recovery, "Observed sales"),
        "recovered_demand": read_metric(recovery, "Recovered latent demand"),
        "recovered_lost": read_metric(recovery, "Recovered lost demand"),
        "recovery_lift": read_metric(recovery, "Recovered lift over observed"),
        "stockout_rate": read_metric(recovery, "Stockout row rate"),
        "calibration_factor": read_metric(recovery, "Imputation calibration factor"),
        "lost_cap": read_metric(recovery, "Lost demand cap value"),
        "hybrid": hybrid,
        "observed": observed,
        "seasonal": seasonal,
        "recovered_ml": recovered_ml,
        "naive": naive,
        "rolling": rolling,
        "hybrid_diag": hybrid_diag,
        "hybrid_interval": hybrid_interval,
        "pseudo_daily": pseudo_daily,
        "q90": q90,
        "q100": q100,
        "blend": blend,
        "spectrum": spectrum,
        "rel_gain": (float(observed["WAPE"]) - float(hybrid["WAPE"])) / float(observed["WAPE"]),
    }


def build_tex() -> str:
    m = metric_bundle()
    return rf"""\documentclass[aspectratio=169,10pt]{{beamer}}
\usetheme{{Madrid}}
\usecolortheme{{default}}
\usepackage{{fontspec}}
\setmainfont{{Times New Roman}}
\setsansfont{{Arial}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{tikz}}
\graphicspath{{{{../../outputs/figures/}}{{../figures/}}}}
\setbeamertemplate{{navigation symbols}}{{}}
\setbeamertemplate{{caption}}[numbered]
\setbeamertemplate{{itemize items}}[circle]
\definecolor{{FreshGreen}}{{RGB}}{{31,121,74}}
\definecolor{{FreshDark}}{{RGB}}{{22,78,99}}
\setbeamercolor{{structure}}{{fg=FreshGreen}}
\setbeamercolor{{frametitle}}{{fg=white,bg=FreshGreen}}
\setbeamercolor{{title}}{{fg=white,bg=FreshGreen}}
\setbeamercolor{{block title}}{{fg=white,bg=FreshDark}}
\setbeamercolor{{block body}}{{bg=gray!6}}

\title[FreshRetailNet-50K]{{Dự báo chuỗi thời gian bán lẻ có xét đến tình trạng hết hàng}}
\subtitle{{Two-stage latent demand recovery + daily demand forecasting trên FreshRetailNet-50K}}
\author{{Nhóm 3}}
\date{{}}

\begin{{document}}

\begin{{frame}}
    \titlepage
\end{{frame}}

\begin{{frame}}{{Mục lục trình bày}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.48\linewidth}}
            \begin{{enumerate}}
                \item Introduction
                \item Dataset \& EDA
                \item Methodology
                \item Forecasting Results
            \end{{enumerate}}
        \end{{column}}
        \begin{{column}}{{0.48\linewidth}}
            \begin{{enumerate}}
                \setcounter{{enumi}}{{4}}
                \item Model Evaluation
                \item Business Insights
                \item Conclusion
                \item Future Work
            \end{{enumerate}}
        \end{{column}}
    \end{{columns}}
    \vspace{{0.25cm}}
    \begin{{block}}{{Thông điệp chính}}
    Với dữ liệu bán lẻ có stockout, forecast trực tiếp observed sales dễ học sai nhu cầu. Đồ án xử lý bằng pipeline two-stage: recover latent demand theo giờ, rồi forecast daily next-7-day demand.
    \end{{block}}
\end{{frame}}

\section{{Introduction}}
\begin{{frame}}{{1. Introduction: bài toán không chỉ là forecast sales}}
    \begin{{block}}{{Vấn đề cốt lõi}}
    Observed sales có thể bị \textbf{{censor}} bởi stockout: bán thấp không chắc là nhu cầu thấp, mà có thể là cửa hàng không còn hàng để bán.
    \end{{block}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Nếu forecast trực tiếp observed sales}}
            \begin{{itemize}}
                \item Model học cả nhu cầu thật lẫn giới hạn tồn kho.
                \item Dễ under-forecast ở nhóm hay hết hàng.
                \item Vòng lặp xấu: dự báo thấp $\rightarrow$ nhập ít $\rightarrow$ tiếp tục stockout.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Hướng của đồ án}}
            \begin{{itemize}}
                \item Recover latent demand ở các giờ stockout.
                \item Aggregate hourly recovered demand lên daily.
                \item Forecast tổng demand 7 ngày tiếp theo.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Câu hỏi nghiên cứu và contribution}}
    \begin{{enumerate}}
        \item Có thể recover demand bị che trong stockout mà không leakage thời gian không?
        \item Imputation có làm tổng demand tăng quá mức không?
        \item Forecast trên recovered demand có tốt hơn forecast trên observed sales không?
        \item Các kết quả này chuyển thành insight vận hành như thế nào?
    \end{{enumerate}}
    \vspace{{0.2cm}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Đóng góp kỹ thuật}}
            \begin{{itemize}}
                \item Expanding-window recovery.
                \item Calibration + cap q90.
                \item Pseudo-stockout validation.
                \item Hybrid seasonal-ML.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Đóng góp business}}
            \begin{{itemize}}
                \item Ước lượng lost sales.
                \item Giữ weekly seasonality làm baseline vận hành.
                \item Đề xuất workflow replenishment có uncertainty.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\section{{Dataset \& EDA}}
\begin{{frame}}{{2. Dataset: FreshRetailNet-50K sample}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.48\linewidth}}
            \begin{{tabular}}{{lr}}
                \toprule
                Sample fraction & {pct(m["sample_frac"])} \\
                Hourly rows & {integer(m["n_rows"])} \\
                Series & {integer(m["n_series"])} \\
                Stores & {integer(m["n_stores"])} \\
                Products & {integer(m["n_products"])} \\
                Stockout row rate & {pct(m["stockout_rate"])} \\
                \bottomrule
            \end{{tabular}}
            \vspace{{0.25cm}}
            \begin{{itemize}}
                \item Daily rows chứa \texttt{{hours\_sale}} và \texttt{{hours\_stock\_status}}.
                \item Disaggregate daily $\rightarrow$ hourly để xử lý stockout đúng cấp.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.50\linewidth}}
            \includegraphics[width=\linewidth]{{data_preprocessing_pipeline.png}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{EDA: stockout là vấn đề trung tâm}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.50\linewidth}}
            \includegraphics[width=\linewidth]{{stockout_rate_over_time.png}}
        \end{{column}}
        \begin{{column}}{{0.47\linewidth}}
            \textbf{{Hàm ý}}
            \begin{{itemize}}
                \item Stockout rate khoảng {pct(m["stockout_rate"])}.
                \item Đây không phải nhiễu nhỏ để bỏ qua.
                \item Observed sales cần được đọc là dữ liệu có thể bị kiểm duyệt.
            \end{{itemize}}
            \vspace{{0.15cm}}
            \begin{{block}}{{Framing}}
            Target cuối không phải raw sales, mà là recovered demand proxy có kiểm soát.
            \end{{block}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{EDA: dữ liệu thưa, lệch và có mùa vụ}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.49\linewidth}}
            \includegraphics[width=\linewidth]{{log_sale_amount_distribution.png}}
            \begin{{itemize}}
                \item Nhiều zero, spike và phân phối lệch.
                \item WAPE phù hợp hơn MAPE.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.49\linewidth}}
            \includegraphics[width=\linewidth]{{sales_by_day_of_week.png}}
            \begin{{itemize}}
                \item Weekly seasonality rõ.
                \item Seasonal naive 7-day là benchmark bắt buộc.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{ACF/PACF và spectrum dùng để làm gì?}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.50\linewidth}}
            \includegraphics[width=\linewidth]{{acf_aggregate_sales.png}}
        \end{{column}}
        \begin{{column}}{{0.47\linewidth}}
            \includegraphics[width=\linewidth]{{spectrum_daily_recovered_demand.png}}
        \end{{column}}
    \end{{columns}}
    \vspace{{0.1cm}}
    \begin{{block}}{{Cách dùng trong bài}}
    Không dùng ACF/PACF để ép chọn ARIMA. Dùng chúng để biện minh lag/rolling/seasonal features và baseline mùa vụ.
    \end{{block}}
\end{{frame}}

\begin{{frame}}{{Tại sao không chọn classical time series làm model chính?}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.50\linewidth}}
            \textbf{{ARIMA/SARIMA từng chuỗi}}
            \begin{{itemize}}
                \item Khó scale cho 5.000 series sample, 50.000 series full.
                \item Không ổn với chuỗi thưa, zero-inflated.
                \item Khó đưa stockout, promotion, weather, category.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.50\linewidth}}
            \textbf{{Deep learning}}
            \begin{{itemize}}
                \item Có thể dùng nhưng cần tuning/tài nguyên lớn.
                \item Rủi ro khó bảo vệ nếu validation chưa cực chặt.
                \item Đồ án ưu tiên pipeline rõ, nhẹ, tái lập.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
    \vspace{{0.2cm}}
    \begin{{block}}{{Lựa chọn chính}}
    LightGBM global model + seasonal baseline: cân bằng giữa hiệu suất, tốc độ, giải thích và tính tái lập.
    \end{{block}}
\end{{frame}}

\section{{Methodology}}
\begin{{frame}}{{3. Methodology: two-stage framework}}
    \centering
    \includegraphics[width=0.92\linewidth,height=0.78\textheight,keepaspectratio]{{owner_expanding_window_process.png}}
\end{{frame}}

\begin{{frame}}{{Expanding-window recovery chống leakage}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.52\linewidth}}
            \begin{{itemize}}
                \item Warm-up 14 ngày để có đủ hai chu kỳ tuần.
                \item Mỗi block 7 ngày chỉ học từ non-stockout rows trong quá khứ.
                \item Validation/test recovery chỉ dùng model fit từ train-period.
                \item Calibration cũng lấy từ train-period out-of-fold rows.
            \end{{itemize}}
            \begin{{block}}{{Nguyên tắc}}
            Không dùng ngày tương lai để recover nhu cầu của ngày quá khứ.
            \end{{block}}
        \end{{column}}
        \begin{{column}}{{0.46\linewidth}}
            \includegraphics[width=\linewidth,height=0.66\textheight,keepaspectratio]{{owner_expanding_window_recovery_detail.png}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Feature engineering: không chỉ lag của chính series}}
    \small
    \begin{{columns}}[T]
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Nhóm feature chính}}
            \begin{{itemize}}
                \item Calendar: hour, day-of-week, month.
                \item Lag/rolling: 1, 7, 14, 28 ngày; 24/168 giờ.
                \item Stockout: flag, rate, rolling stockout.
                \item Promotion/weather/category/store/product.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Hai điểm mù được bổ sung}}
            \begin{{itemize}}
                \item Velocity/momentum: bán nhanh trước stockout.
                \item Peer/substitution: sản phẩm cùng store/category.
                \item Mục tiêu: cho recovery nhìn thêm bối cảnh xung quanh stockout.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
    \vspace{{0.05cm}}
    \centering
    \includegraphics[width=0.62\linewidth,height=0.28\textheight,keepaspectratio]{{stockout_substitution_velocity_diagnostics.png}}
\end{{frame}}

\begin{{frame}}{{Imputation control: không để recovery phóng đại demand}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.47\linewidth}}
            \begin{{tabular}}{{lr}}
                \toprule
                Observed sales & {num(m["observed_sales"])} \\
                Recovered demand & {num(m["recovered_demand"])} \\
                Recovered lost demand & {num(m["recovered_lost"])} \\
                Lift & {pct(m["recovery_lift"])} \\
                Calibration factor & {num(m["calibration_factor"], 4)} \\
                Cap q90 & {num(m["lost_cap"], 4)} \\
                \bottomrule
            \end{{tabular}}
        \end{{column}}
        \begin{{column}}{{0.50\linewidth}}
            \begin{{block}}{{Công thức}}
            Raw impute $\rightarrow$ calibration $\rightarrow$ positive lost demand $\rightarrow$ cap q90 $\rightarrow$ recovered demand.
            \end{{block}}
            \begin{{itemize}}
                \item q90 giữ khoảng {pct(m["q90"]["Share of original recovered lost demand"])} raw lost demand.
                \item q100 lift lên {pct(m["q100"]["Recovered lift over observed"])} nhưng rủi ro over-imputation.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Pseudo-stockout validation và sensitivity}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.48\linewidth}}
            \textbf{{Pseudo-stockout validation}}
            \begin{{itemize}}
                \item Dùng non-stockout rows trong validation làm ``đáp án giả''.
                \item Daily aggregate WAPE: {pct(m["pseudo_daily"]["WAPE"])}.
                \item Prediction/actual ratio: {num(m["pseudo_daily"]["Prediction / actual ratio"], 4)}.
                \item Recovery còn hơi over-predict, nên cần cap.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.50\linewidth}}
            \includegraphics[width=\linewidth]{{imputation_daily_uplift.png}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\section{{Forecasting Results}}
\begin{{frame}}{{4. Forecasting setup}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.50\linewidth}}
            \textbf{{Target}}
            \begin{{itemize}}
                \item Hourly recovered demand $\rightarrow$ daily demand.
                \item Forecast origin ngày $t$.
                \item Target: tổng demand từ $t+1$ đến $t+7$.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.48\linewidth}}
            \textbf{{Models}}
            \begin{{itemize}}
                \item Naive x7.
                \item Seasonal naive 7-day.
                \item Observed-sales LightGBM.
                \item Recovered-demand LightGBM.
                \item Hybrid seasonal-ML.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
    \vspace{{0.2cm}}
    \begin{{block}}{{Evaluation}}
    So sánh trên recovered latent demand proxy để trả lời đúng câu hỏi: forecast demand đã giảm bias do stockout.
    \end{{block}}
\end{{frame}}

\begin{{frame}}{{Forecasting results: hybrid tốt nhất nhưng seasonal naive rất mạnh}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.47\linewidth}}
            \begin{{tabular}}{{lrr}}
                \toprule
                Model & WAPE & WPE \\
                \midrule
                Naive x7 & {pct(m["naive"]["WAPE"])} & {pct(m["naive"]["WPE"])} \\
                Seasonal naive & {pct(m["seasonal"]["WAPE"])} & {pct(m["seasonal"]["WPE"])} \\
                Observed LGBM & {pct(m["observed"]["WAPE"])} & {pct(m["observed"]["WPE"])} \\
                Recovered LGBM & {pct(m["recovered_ml"]["WAPE"])} & {pct(m["recovered_ml"]["WPE"])} \\
                Hybrid & {pct(m["hybrid"]["WAPE"])} & {pct(m["hybrid"]["WPE"])} \\
                \bottomrule
            \end{{tabular}}
            \vspace{{0.15cm}}
            \begin{{itemize}}
                \item Hybrid giảm WAPE {pct(m["rel_gain"])} so với observed-sales LGBM.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.51\linewidth}}
            \includegraphics[width=\linewidth]{{owner_two_stage_forecast_comparison.png}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Hybrid seasonal-ML là gì?}}
    \begin{{block}}{{Công thức kết hợp}}
    $\hat{{y}}_{{hybrid}} = \alpha \hat{{y}}_{{LightGBM}} + (1-\alpha)\hat{{y}}_{{SeasonalNaive}}$
    \end{{block}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Trọng số chọn bằng validation}}
            \begin{{itemize}}
                \item LightGBM weight: {pct(m["blend"]["LightGBM weight"], 0)}.
                \item Seasonal naive weight: {pct(m["blend"]["Seasonal naive weight"], 0)}.
                \item Không chọn bằng test set.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Tại sao hợp lý?}}
            \begin{{itemize}}
                \item Seasonal naive giữ pattern tuần.
                \item LightGBM hiệu chỉnh theo stockout, lag, product/store.
                \item Giảm rủi ro ML học quá phức tạp nhưng thua baseline.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Feature importance: mô hình học gì?}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.47\linewidth}}
            \begin{{itemize}}
                \item \texttt{{stockout\_rate}} là feature quan trọng nhất.
                \item Lag 7/14/28 ngày xác nhận weekly pattern.
                \item Rolling mean/sum giúp ổn định chuỗi thưa.
            \end{{itemize}}
            \begin{{block}}{{Diễn giải}}
            Model không chỉ dựa vào trend gần nhất; nó kết hợp stockout, seasonality và lịch sử demand đã recover.
            \end{{block}}
        \end{{column}}
        \begin{{column}}{{0.51\linewidth}}
            \includegraphics[width=\linewidth]{{owner_recovereddemand_forecasting_feature_importance_top20.png}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\section{{Model Evaluation}}
\begin{{frame}}{{5. Model evaluation: residual diagnostics}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.49\linewidth}}
            \includegraphics[width=\linewidth]{{owner_two_stage_residual_acf.png}}
        \end{{column}}
        \begin{{column}}{{0.49\linewidth}}
            \begin{{itemize}}
                \item Hybrid Ljung--Box p-value: {num(m["hybrid_diag"]["Ljung-Box p-value"], 4)}.
                \item Residual chưa hoàn toàn white noise.
                \item Một phần do target next-7-day bị overlap.
                \item Kết luận trung thực: model cải thiện forecast nhưng chưa học hết pattern.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Prediction intervals: forecast không phải một con số chắc chắn}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.48\linewidth}}
            \includegraphics[width=\linewidth]{{owner_two_stage_forecast_interval.png}}
        \end{{column}}
        \begin{{column}}{{0.49\linewidth}}
            \begin{{tabular}}{{lr}}
                \toprule
                Hybrid 80\% coverage & {pct(m["hybrid_interval"]["80% coverage"])} \\
                Hybrid 95\% coverage & {pct(m["hybrid_interval"]["95% coverage"])} \\
                Mean 80\% width & {num(m["hybrid_interval"]["Mean 80% width"], 2)} \\
                Mean 95\% width & {num(m["hybrid_interval"]["Mean 95% width"], 2)} \\
                \bottomrule
            \end{{tabular}}
            \vspace{{0.2cm}}
            \begin{{itemize}}
                \item 80\% interval hơi hẹp.
                \item 95\% interval phù hợp hơn nhưng rộng hơn.
                \item Dùng để chọn safety stock theo rủi ro.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\section{{Business Insights}}
\begin{{frame}}{{6. Business insights: lost sales và ưu tiên vận hành}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.49\linewidth}}
            \includegraphics[width=\linewidth]{{imputation_top_series_lift.png}}
        \end{{column}}
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Insight}}
            \begin{{itemize}}
                \item Recovered lost demand: {num(m["recovered_lost"])}.
                \item Lift over observed: {pct(m["recovery_lift"])}.
                \item Không phải mọi stockout quan trọng như nhau.
                \item Ưu tiên series có uplift/lost demand cao.
            \end{{itemize}}
            \begin{{block}}{{Hành động}}
            Kiểm tra safety stock, lead time, tần suất bổ sung và chất lượng ghi nhận stock status ở nhóm uplift cao.
            \end{{block}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\begin{{frame}}{{Ứng dụng vào replenishment}}
    \begin{{center}}
    \begin{{tabular}}{{p{{0.23\linewidth}}p{{0.30\linewidth}}p{{0.34\linewidth}}}}
        \toprule
        Bước & Đầu vào & Đầu ra vận hành \\
        \midrule
        Recovery & Hourly sales + stockout & Recovered demand proxy \\
        Forecast & Daily recovered demand & Next-7-day demand \\
        Quy đổi nhập hàng & Forecast + tồn kho + lead time & Đề xuất lượng nhập \\
        Theo dõi & Actual + stockout + error & Cảnh báo bias/drift \\
        \bottomrule
    \end{{tabular}}
    \end{{center}}
    \vspace{{0.15cm}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Nếu chi phí stockout cao}}
            \begin{{itemize}}
                \item Dùng forecast gần upper interval.
                \item Tăng safety stock cho nhóm uplift cao.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.49\linewidth}}
            \textbf{{Nếu hàng dễ hư hỏng}}
            \begin{{itemize}}
                \item Dùng point forecast hoặc interval thấp hơn.
                \item Tăng tần suất cập nhật forecast.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
\end{{frame}}

\section{{Conclusion}}
\begin{{frame}}{{7. Conclusion}}
    \begin{{enumerate}}
        \item Stockout làm observed sales đánh giá thấp demand, nên cần recovery trước forecast.
        \item Expanding-window recovery giúp tránh leakage theo thời gian.
        \item Recovered demand lift khoảng {pct(m["recovery_lift"])} so với observed sales.
        \item Forecast trên recovered demand tốt hơn forecast trực tiếp observed sales.
        \item Hybrid seasonal-ML đạt WAPE {pct(m["hybrid"]["WAPE"])} và giữ được weekly pattern mạnh.
    \end{{enumerate}}
    \vspace{{0.2cm}}
    \begin{{block}}{{Câu chốt}}
    Điểm mạnh của đồ án là pipeline có kiểm soát: đúng framing, có benchmark mạnh, có diagnostics, có uncertainty và có business interpretation.
    \end{{block}}
\end{{frame}}

\begin{{frame}}{{Limitations: nói rõ để bảo vệ tốt hơn}}
    \begin{{itemize}}
        \item Recovered demand là proxy, không phải ground truth tuyệt đối.
        \item Pseudo-stockout validation còn over-predict nhẹ.
        \item Residual chưa white noise hoàn toàn.
        \item Peer/substitution mới ở mức feature đơn giản.
        \item Sample 10\% có thể chưa phản ánh đầy đủ toàn bộ network sản phẩm.
        \item Chưa triển khai deep imputation do phạm vi và tài nguyên đồ án.
    \end{{itemize}}
    \vspace{{0.2cm}}
    \begin{{block}}{{Tinh thần trình bày}}
    Không giấu vấn đề; dùng diagnostics để biết mô hình còn sai ở đâu và vì sao.
    \end{{block}}
\end{{frame}}

\section{{Future Work}}
\begin{{frame}}{{8. Future work}}
    \begin{{columns}}[T]
        \begin{{column}}{{0.50\linewidth}}
            \textbf{{Kỹ thuật}}
            \begin{{itemize}}
                \item Deep imputation: SAITS, CSDI, TimesNet, ImputeFormer.
                \item Product substitution graph.
                \item Quantile/conformal forecasting.
                \item Rolling retraining và drift monitoring.
            \end{{itemize}}
        \end{{column}}
        \begin{{column}}{{0.48\linewidth}}
            \textbf{{Business evaluation}}
            \begin{{itemize}}
                \item Giảm stockout rate.
                \item Giảm lost sales proxy.
                \item Kiểm soát hàng hủy/tồn kho dư.
                \item A/B test theo store/category.
            \end{{itemize}}
        \end{{column}}
    \end{{columns}}
    \vspace{{0.2cm}}
    \begin{{block}}{{Nguyên tắc mở rộng}}
    Model phức tạp hơn chỉ đáng dùng nếu vượt seasonal baseline và giữ được validation chống leakage.
    \end{{block}}
\end{{frame}}

\begin{{frame}}{{Cảm ơn}}
    \centering
    \vspace{{1cm}}
    {{\Large Cảm ơn thầy và các bạn đã lắng nghe!}}\\[0.5cm]
    {{\large Q\&A}}\\[0.8cm]
    \begin{{block}}{{Một câu nhớ nhất}}
    Forecast tốt trong bài này không bắt đầu từ model phức tạp, mà bắt đầu từ việc hiểu đúng observed sales bị stockout censor.
    \end{{block}}
\end{{frame}}

\end{{document}}
"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(build_tex(), encoding="utf-8")
    print(f"Saved: {TEX_PATH}")
    try:
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", TEX_PATH.name],
            cwd=REPORT_DIR,
            check=True,
            timeout=120,
        )
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", TEX_PATH.name],
            cwd=REPORT_DIR,
            check=True,
            timeout=120,
        )
        print(f"Compiled: {PDF_PATH}")
    except Exception as exc:
        print(f"LaTeX compile failed: {exc}")


if __name__ == "__main__":
    main()
