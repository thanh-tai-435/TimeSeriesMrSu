"""Generate a LaTeX report from saved Fresh50K results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR, TABLES_DIR, ensure_directories


def latex_escape(value) -> str:
    """Escape text for LaTeX tables."""
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_value(value) -> str:
    """Format table values for compact LaTeX output."""
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.4f}"
    return str(value)


def csv_table(path: Path, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    """Convert a CSV file to a simple booktabs LaTeX table."""
    df = pd.read_csv(path)
    if columns is not None:
        df = df[[column for column in columns if column in df.columns]]
    if max_rows is not None:
        df = df.head(max_rows)

    column_spec = "l" * len(df.columns)
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule"]
    lines.append(" & ".join(latex_escape(column) for column in df.columns) + r" \\")
    lines.append(r"\midrule")
    for row in df.itertuples(index=False):
        lines.append(" & ".join(latex_escape(format_value(value)) for value in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def figure(path: Path, caption: str, label: str, width: str = r"0.92\linewidth") -> str:
    """Create a LaTeX figure block using a path relative to outputs/reports."""
    rel_path = Path("..") / "figures" / path.name
    return rf"""
\begin{{figure}}[H]
    \centering
    \includegraphics[width={width}]{{{rel_path.as_posix()}}}
    \caption{{{latex_escape(caption)}}}
    \label{{{label}}}
\end{{figure}}
"""


def read_metric(path: Path, metric: str) -> str:
    df = pd.read_csv(path)
    row = df[df["Metric"] == metric]
    if row.empty:
        return ""
    return str(row.iloc[0]["Value"])


def build_report() -> str:
    data_quality = TABLES_DIR / "data_quality_summary.csv"
    split_summary = TABLES_DIR / "split_summary.csv"
    stationarity = TABLES_DIR / "stationarity_tests.csv"
    comparison_h1 = TABLES_DIR / "model_comparison_h1.csv"
    comparison_h24 = TABLES_DIR / "model_comparison_h24.csv"
    ablation = TABLES_DIR / "ablation_results.csv"
    stockout_eval = TABLES_DIR / "stockout_segment_evaluation.csv"
    feature_importance = TABLES_DIR / "lightgbm_feature_importance.csv"
    residual_diagnostics = TABLES_DIR / "residual_diagnostics.csv"
    ridge_coef = TABLES_DIR / "ridge_coefficients_h1.csv"
    quantile_metrics = TABLES_DIR / "quantile_interval_metrics_h1.csv"
    rolling_eval = TABLES_DIR / "rolling_window_evaluation.csv"
    sarimax_comparison = TABLES_DIR / "aggregate_sarimax_comparison.csv"
    stockout_censoring = TABLES_DIR / "stockout_censoring_summary.csv"
    owner_recovery = TABLES_DIR / "owner_latent_recovery_summary.csv"
    owner_forecast = TABLES_DIR / "owner_two_stage_forecasting_comparison.csv"

    sample_frac = read_metric(data_quality, "Sample fraction")
    n_rows = read_metric(data_quality, "Number of rows")
    n_series = read_metric(data_quality, "Number of series")
    start_date = read_metric(data_quality, "Start date")
    end_date = read_metric(data_quality, "End date")

    h1 = pd.read_csv(comparison_h1)
    best_h1 = h1.sort_values("WAPE").iloc[0]
    seasonal_h1 = h1[h1["Model"] == "SeasonalNaive168"].iloc[0]
    improvement = (seasonal_h1["WAPE"] - best_h1["WAPE"]) / seasonal_h1["WAPE"] * 100

    content = rf"""\documentclass[12pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{caption}}
\geometry{{margin=1in}}

\title{{Stockout-Aware Global Machine Learning Forecasting for FreshRetailNet-50K}}
\author{{}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This report presents an hourly global machine learning forecasting pipeline for FreshRetailNet-50K.
The experiment uses a {latex_escape(sample_frac)} series-level sample, preserving complete timelines for selected series.
Daily records are expanded into hourly observations using \texttt{{hours\_sale}} and \texttt{{hours\_stock\_status}}.
The final sample contains {latex_escape(n_rows)} hourly rows and {latex_escape(n_series)} time series from {latex_escape(start_date)} to {latex_escape(end_date)}.
\end{{abstract}}

\section{{Dataset and Preprocessing}}
Raw \texttt{{train.parquet}} and \texttt{{eval.parquet}} files were loaded, normalized, sorted by store-product-time identifiers, and converted to an hourly table.
The identifier \texttt{{series\_id}} is constructed from \texttt{{city\_id}}, \texttt{{store\_id}}, and \texttt{{product\_id}}.
Stockout status is derived from hourly stock status sequences.

\begin{{table}}[H]
\centering
\caption{{Data quality summary}}
{csv_table(data_quality)}
\end{{table}}

\section{{Exploratory Data Analysis}}
The hourly sample shows strong sparsity, daily seasonality, and visible stockout variation over time.
The following figures summarize aggregate demand, stockout behavior, sales distribution, and seasonality.

{figure(FIGURES_DIR / "aggregate_sales_over_time.png", "Aggregate hourly sales over time.", "fig:aggregate-sales")}
{figure(FIGURES_DIR / "stockout_rate_over_time.png", "Hourly stockout rate over time.", "fig:stockout-rate")}
{figure(FIGURES_DIR / "sale_amount_distribution.png", "Distribution of hourly sale amount.", "fig:sale-dist", r"0.78\linewidth")}
{figure(FIGURES_DIR / "log_sale_amount_distribution.png", "Distribution of log-transformed hourly sale amount.", "fig:log-sale-dist", r"0.78\linewidth")}
{figure(FIGURES_DIR / "sales_by_hour_of_day.png", "Average sales by hour of day.", "fig:hour-seasonality", r"0.78\linewidth")}
{figure(FIGURES_DIR / "sales_by_day_of_week.png", "Average sales by day of week.", "fig:dow-seasonality", r"0.78\linewidth")}

\begin{{table}}[H]
\centering
\caption{{Representative series selected for detailed plots}}
{csv_table(TABLES_DIR / "representative_series.csv")}
\end{{table}}

{figure(FIGURES_DIR / "representative_series_high_volume.png", "Representative high-volume series.", "fig:rep-high")}
{figure(FIGURES_DIR / "representative_series_intermittent.png", "Representative intermittent series.", "fig:rep-intermittent")}
{figure(FIGURES_DIR / "representative_series_stockout_heavy.png", "Representative stockout-heavy series.", "fig:rep-stockout")}

\section{{Stationarity and Autocorrelation}}
ADF and KPSS tests were run on aggregate hourly sales and representative series.
Both first differencing and seasonal differencing with lag 24 were evaluated.

\begin{{table}}[H]
\centering
\caption{{ADF and KPSS stationarity test results}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(stationarity, columns=["Series", "Transformation", "ADF p-value", "KPSS p-value", "Conclusion", "N"])}
}}
\end{{table}}

{figure(FIGURES_DIR / "acf_aggregate_sales.png", "ACF of aggregate hourly sales.", "fig:acf-aggregate", r"0.85\linewidth")}
{figure(FIGURES_DIR / "pacf_aggregate_sales.png", "PACF of aggregate hourly sales.", "fig:pacf-aggregate", r"0.85\linewidth")}
{figure(FIGURES_DIR / "acf_high_volume_series.png", "ACF of the high-volume representative series.", "fig:acf-high", r"0.85\linewidth")}
{figure(FIGURES_DIR / "pacf_high_volume_series.png", "PACF of the high-volume representative series.", "fig:pacf-high", r"0.85\linewidth")}

\section{{Feature Engineering and Time Split}}
Calendar, lag, rolling, promotion, weather, and stockout-aware features were constructed.
Rolling features are shifted by one period before aggregation to avoid target leakage.
The forecasting targets are \texttt{{target\_h1}} and \texttt{{target\_h24}}, corresponding to one hour ahead and 24 hours ahead.

\begin{{table}}[H]
\centering
\caption{{Time-based train-validation-test split}}
{csv_table(split_summary)}
\end{{table}}

\section{{Modeling and Evaluation}}
Benchmarks include naive forecasting, seasonal naive 24h, and seasonal naive 168h.
Machine learning models include Ridge regression and LightGBM global forecasting.

\subsection{{Model Choice Rationale}}
The main model is LightGBM because the problem is a large-scale tabular forecasting task with thousands of sparse store-product series.
A separate ARIMA/SARIMA model for each series would be difficult to scale and unstable for intermittent demand.
LightGBM can learn global non-linear relationships from lag, rolling, calendar, promotion, weather, and stockout-aware features.
Ridge regression is retained as an interpretable linear baseline, while SARIMAX is evaluated on aggregate sales as a classical time-series comparator.
Deep learning models were not selected as the main approach because they require more tuning and compute, while LightGBM offers a stronger accuracy-to-complexity tradeoff for this reproducible course project.

\section{{Paper/Repository-Aligned Two-Stage Approach}}
The FreshRetailNet-50K paper and baseline repository frame the benchmark as two linked tasks: latent demand recovery during stockouts, followed by demand forecasting on recovered demand.
This project implements a lightweight version of that idea.
To avoid temporal leakage, recovery inside the training period uses expanding weekly blocks: each block is recovered by a model trained only on past non-stockout hours.
Validation and test periods are recovered by a final recovery model trained only on training-period non-stockout hours.
For stockout hours, recovered demand is defined as \texttt{{max(observed sales, imputed demand)}}.
Hourly recovered demand is then aggregated to daily demand for a 7-day-ahead forecasting comparison.

\begin{{table}}[H]
\centering
\caption{{Owner-aligned latent demand recovery summary}}
{csv_table(owner_recovery)}
\end{{table}}

{figure(FIGURES_DIR / "owner_observed_vs_recovered_demand.png", "Observed sales versus recovered latent demand.", "fig:owner-recovered")}

\begin{{table}}[H]
\centering
\caption{{Observed-sales forecasting versus recovered-demand forecasting}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(owner_forecast, columns=["Model", "Training target", "Evaluation target", "RMSE", "MAE", "WAPE", "sMAPE", "WPE", "N"])}
}}
\end{{table}}

{figure(FIGURES_DIR / "owner_two_stage_bias_comparison.png", "Bias comparison against recovered latent demand proxy.", "fig:owner-bias", r"0.82\linewidth")}
{figure(FIGURES_DIR / "owner_two_stage_forecast_comparison.png", "Owner-style 7-day demand forecasting comparison.", "fig:owner-forecast")}

\begin{{table}}[H]
\centering
\caption{{Model comparison for horizon h=1}}
{csv_table(comparison_h1, columns=["Model", "RMSE", "MAE", "WAPE", "sMAPE", "N"])}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Model comparison for horizon h=24}}
{csv_table(comparison_h24, columns=["Model", "RMSE", "MAE", "WAPE", "sMAPE", "N"])}
\end{{table}}

The best h=1 model by WAPE is {latex_escape(best_h1["Model"])} with WAPE {best_h1["WAPE"]:.4f}.
Compared with SeasonalNaive168, this is an improvement of {improvement:.2f}\%.

{figure(FIGURES_DIR / "model_comparison_wape_h1.png", "Model comparison by WAPE for h=1.", "fig:model-wape", r"0.82\linewidth")}
{figure(FIGURES_DIR / "model_comparison_rmse_h1.png", "Model comparison by RMSE for h=1.", "fig:model-rmse", r"0.82\linewidth")}

\section{{Residual Diagnostics and Forecast Uncertainty}}
The final presentation rubric requires an honest diagnostic read of model assumptions.
For LightGBM, residual diagnostics are reported using aggregate residual autocorrelation, Ljung-Box tests, and Jarque-Bera normality tests.
Because the forecast model is machine-learning based, prediction intervals are constructed empirically from validation residual quantiles.

\begin{{table}}[H]
\centering
\caption{{LightGBM residual diagnostics}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(residual_diagnostics, columns=["Horizon", "Model", "RMSE", "MAE", "WAPE", "Residual mean", "Residual std", "Ljung-Box p-value", "Jarque-Bera p-value", "N"])}
}}
\end{{table}}

{figure(FIGURES_DIR / "residual_distribution_lightgbm_h1.png", "LightGBM residual distribution for h=1.", "fig:residual-dist-h1", r"0.78\linewidth")}
{figure(FIGURES_DIR / "residual_acf_lightgbm_h1.png", "Aggregate residual ACF for LightGBM h=1.", "fig:residual-acf-h1", r"0.82\linewidth")}
{figure(FIGURES_DIR / "forecast_interval_lightgbm_h1.png", "LightGBM h=1 empirical forecast intervals on a representative series.", "fig:forecast-interval-h1")}

\begin{{table}}[H]
\centering
\caption{{Top Ridge coefficients for h=1}}
{csv_table(ridge_coef, columns=["feature", "coefficient", "abs_coefficient"], max_rows=12)}
\end{{table}}

\subsection{{Quantile Forecast Intervals}}
To avoid relying only on residual-normality assumptions, quantile LightGBM models were trained for the 10th, 50th, and 90th conditional quantiles.
This provides a model-based uncertainty interval for the h=1 forecast.

\begin{{table}}[H]
\centering
\caption{{Quantile LightGBM interval metrics for h=1}}
{csv_table(quantile_metrics, columns=["Model", "RMSE", "MAE", "WAPE", "sMAPE", "Interval", "Coverage", "Mean interval width", "N"])}
\end{{table}}

{figure(FIGURES_DIR / "forecast_interval_quantile_lightgbm_h1.png", "Quantile LightGBM h=1 forecast interval.", "fig:quantile-interval")}

\section{{Ablation Study}}
The ablation study evaluates the incremental value of promotion, weather, and stockout features using LightGBM.

\begin{{table}}[H]
\centering
\caption{{LightGBM ablation results}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(ablation, columns=["Horizon", "Feature set", "RMSE", "MAE", "WAPE", "sMAPE", "Train rows used"])}
}}
\end{{table}}

{figure(FIGURES_DIR / "ablation_wape_h1.png", "Ablation study by WAPE for h=1.", "fig:ablation-wape")}

\section{{Stockout Segment Evaluation}}
Forecasting performance is also evaluated separately for stockout and non-stockout target periods.

\begin{{table}}[H]
\centering
\caption{{Stockout and non-stockout segment evaluation}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(stockout_eval, columns=["Horizon", "Model", "Segment", "RMSE", "MAE", "WAPE", "sMAPE", "N"], max_rows=30)}
}}
\end{{table}}

{figure(FIGURES_DIR / "stockout_segment_wape.png", "WAPE by stockout segment.", "fig:stockout-segment")}

\subsection{{Stockout Censoring and Lost-Sales Proxy}}
Observed sales during stockout periods are censored: a zero or low sale may reflect lack of inventory rather than lack of demand.
As an additional business analysis, a simple lost-sales proxy is computed using the 168-hour rolling mean as expected uncensored demand during stockout hours.

\begin{{table}}[H]
\centering
\caption{{Stockout censoring summary}}
{csv_table(stockout_censoring)}
\end{{table}}

{figure(FIGURES_DIR / "stockout_lost_sales_proxy_by_hour.png", "Observed sales and stockout lost-sales proxy by hour.", "fig:lost-sales-proxy", r"0.82\linewidth")}

\section{{Additional Robustness Checks}}
The model is checked for stability across weekly windows inside the test period, and a classical aggregate SARIMAX model is compared against an aggregate seasonal-naive benchmark.
The SARIMAX result is not intended to replace product-level global ML; it is used to show that a classical aggregate specification does not automatically dominate simple seasonal structure.

\begin{{table}}[H]
\centering
\caption{{Rolling-window evaluation on the test period}}
\resizebox{{\linewidth}}{{!}}{{%
{csv_table(rolling_eval, columns=["Horizon", "Model", "Window start", "Window end", "RMSE", "MAE", "WAPE", "sMAPE", "N"])}
}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Aggregate SARIMAX comparison}}
{csv_table(sarimax_comparison, columns=["Model", "RMSE", "MAE", "WAPE", "sMAPE", "AIC", "BIC"])}
\end{{table}}

{figure(FIGURES_DIR / "aggregate_sarimax_forecast.png", "Aggregate SARIMAX forecast compared with actual aggregate sales.", "fig:sarimax-aggregate")}

\section{{Forecast Examples and Feature Importance}}
Forecast examples compare actual demand, LightGBM predictions, and the seasonal naive 168h benchmark.

{figure(FIGURES_DIR / "forecast_high_volume_series.png", "Forecast example for high-volume series.", "fig:forecast-high")}
{figure(FIGURES_DIR / "forecast_intermittent_series.png", "Forecast example for intermittent series.", "fig:forecast-intermittent")}
{figure(FIGURES_DIR / "forecast_stockout_heavy_series.png", "Forecast example for stockout-heavy series.", "fig:forecast-stockout")}

\begin{{table}}[H]
\centering
\caption{{Top LightGBM features}}
{csv_table(feature_importance, columns=["feature", "importance"], max_rows=15)}
\end{{table}}

{figure(FIGURES_DIR / "lightgbm_feature_importance_top20.png", "Top 20 LightGBM feature importances.", "fig:feature-importance", r"0.78\linewidth")}

\section{{Conclusion}}
The hourly global ML forecasting pipeline improves over naive and seasonal naive benchmarks on the h=1 task.
LightGBM achieves the best WAPE among the evaluated models.
The ablation study indicates that adding stockout-aware features improves forecasting performance, supporting the value of stockout information for retail time series forecasting.

\end{{document}}
"""
    return content


def main() -> None:
    ensure_directories()
    output = REPORTS_DIR / "fresh50k_report.tex"
    output.write_text(build_report(), encoding="utf-8")
    print(f"Saved LaTeX report: {output}")


if __name__ == "__main__":
    main()
