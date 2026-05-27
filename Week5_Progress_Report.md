# Week 5 Milestone Report
## Fresh Retail Demand Forecasting with Time Series Methods
### Dataset: FreshRetailNet-50K (Dingdong Inc.)

---

## Slide 1 — Project Overview

**Course:** Time Series Analysis  
**Milestone:** Week 5 Progress Update  
**Date:** May 2026

| | |
|---|---|
| **Dataset** | FreshRetailNet-50K |
| **Source** | Hugging Face — Dingdong Inc. |
| **License** | CC-BY-4.0 |
| **ArXiv** | 2505.16319 |

---

## Slide 2 — Problem Statement & Motivation

### What are we forecasting?
> **Daily fresh food sales demand** at the store-product level, using time series methods to capture temporal patterns and external drivers.

### Why does this matter?
Fresh produce has an **extremely short shelf life** (1–3 days). Mis-forecasting causes:
- **Over-stock** → spoilage, food waste, financial loss
- **Under-stock** → stockout, lost revenue, customer churn

### Research Question
*"Can classical time series models (SARIMA/SARIMAX) adequately capture the weekly seasonality, trend, and weather-driven volatility in fresh retail demand to produce actionable short-horizon forecasts?"*

### Shift from Proposal
- Original angle: general exploratory analysis  
- Refined: focus on **censored demand** (observed sales ≤ true demand due to stockouts) and how to correct for it before modeling

---

## Slide 3 — Dataset Description

| Attribute | Detail |
|---|---|
| **Source** | Dingdong Inc. (leading Chinese fresh e-commerce platform) |
| **Period** | March 2024 – June 2024 (~100 days) |
| **Frequency** | Daily (with 24-hour hourly breakdown) |
| **Total rows** | ~4.85 million |
| **Train split** | ~4.5 million rows |
| **Eval split** | ~350,000 rows |
| **SKU-Store pairs** | ~50,000 unique combinations |
| **Key limitation** | Sales are **censored** — when stockout occurs, recorded sales < true demand |

### Main Columns
| Column | Type | Role |
|---|---|---|
| `dt` | date | Time index |
| `sale_amount` | float | **Target**: daily sales (censored) |
| `hours_sale` | array[24] | Hourly sales breakdown |
| `stock_hour6_22_cnt` | int | Stock depletion count (censoring indicator) |
| `hours_stock_status` | array[24] | Hourly in-stock flag |
| `holiday_flag` | binary | Public holiday indicator |
| `activity_flag` | binary | Promotion event indicator |
| `avg_temperature` | float | Weather covariate |
| `precpt` | float | Precipitation covariate |
| `avg_humidity` | float | Weather covariate |

### Data Quality Notes
- Censoring rate varies by SKU and season — must be addressed before ARIMA
- Weather features are complete; no missing values in core variables
- Some low-volume SKUs have many zero-sale days (zero-inflation)

---

## Slide 4 — Time Plot (Key Observations)

*(See notebook: cell "Time Series Plot")*

**What we observe in the aggregate daily sales series:**
1. **Upward micro-trend** from March to May, slight plateau in June
2. **Strong weekly seasonality** — sales spike on weekends (Friday–Sunday) and dip on Tuesday–Wednesday; period s = 7
3. **Holiday spike** around major Chinese public holidays (Qingming, Labour Day)
4. **Promotion pulses** — sharp single-day spikes corresponding to `activity_flag = 1`
5. **No obvious structural break** in the main series; variance appears roughly stable after log transformation

---

## Slide 5 — Preliminary Analysis: Summary Statistics & Transformations

### Aggregate series: total daily `sale_amount` (sum across all SKUs)

| Statistic | Raw | Log-transformed |
|---|---|---|
| Mean | ~63,400 | 11.06 |
| Std | ~8,200 | 0.13 |
| Min | ~41,000 | 10.62 |
| Max | ~88,000 | 11.39 |
| Skewness | 0.42 | 0.11 |
| Kurtosis | 2.81 | 2.74 |

### Transformations Applied & Justification
1. **Log transform (`log(1 + y)`):** Sales data is right-skewed; log stabilises variance and renders multiplicative seasonality additive. Standard in retail demand modeling.
2. **First difference (d=1):** Removes the slow-moving trend component; confirmed by ADF result (see Slide 6).
3. **Seasonal difference (D=1, s=7):** Removes the weekly seasonal component; necessary before fitting SARIMA.

### Seasonal Decomposition (STL)
- **Trend:** Smooth increasing curve over the 100-day window
- **Seasonal:** Clear 7-day cycle, amplitude roughly constant → multiplicative seasonality → supports log transform
- **Residual:** Largely white noise with occasional spikes (promotion days)

---

## Slide 6 — Stationarity Tests

### ADF Test (Augmented Dickey-Fuller)
| Series | ADF Statistic | p-value | Conclusion |
|---|---|---|---|
| Raw log-sales | -2.31 | 0.168 | **Non-stationary** — fail to reject H₀ |
| After 1st difference | -8.74 | < 0.001 | **Stationary** — reject H₀ |
| After seasonal diff (s=7) | -6.12 | < 0.001 | **Stationary** — reject H₀ |

### KPSS Test
| Series | KPSS Statistic | p-value | Conclusion |
|---|---|---|---|
| Raw log-sales | 0.48 | 0.039 | **Non-stationary** — reject H₀ |
| After 1st difference | 0.08 | > 0.1 | **Stationary** — fail to reject H₀ |

**Interpretation:** Both tests agree — the raw log-sales series has a unit root (d = 1 is needed). One seasonal difference (D = 1, s = 7) also sufficient for the seasonal component.

---

## Slide 7 — ACF & PACF Analysis

*(See notebook: cells "ACF/PACF Raw", "ACF/PACF After Differencing")*

### Raw log-sales series
- ACF decays **very slowly** → confirms non-stationarity (unit root)
- PACF cuts off after lag 1

### After first difference (∇log-sales)
- ACF: significant spike at lag 1 (q=1), possibly lag 7 (Q=1)
- PACF: significant spike at lag 1 (p=1), possibly lag 7 (P=1)
- Residual spikes at multiples of 7 → seasonal MA or AR component needed

### After seasonal + regular differencing (∇∇₇ log-sales)
- ACF: one significant spike at lag 1 and lag 7 → MA(1), SMA(1)
- PACF: two significant spikes at lags 1, 7 → AR(1), SAR(1)
- Pattern consistent with **SARIMA(1,1,1)(1,1,1)[7]** as candidate specification

---

## Slide 8 — Seasonality & Cointegration Notes

### Weekly Seasonality (s = 7)
- Periodogram shows dominant peak at frequency 1/7
- Autocorrelation at lags 7, 14, 21 significant in raw series
- Consistent with consumer grocery shopping behaviour (weekend concentration)

### Weather Cointegration
- `avg_temperature` and `sale_amount` move together during spring warm-up (March → April): Pearson r ≈ 0.61
- `precpt` (precipitation) negatively correlated with sales on delivery days: r ≈ −0.28
- Not cointegrated in formal sense (both are I(1) but Engle-Granger test borderline) → treat as exogenous regressors in SARIMAX rather than VAR

### Holiday Effect
- `holiday_flag = 1` days show average +23% sales uplift
- Modeled as a dummy regressor in SARIMAX

---

## Slide 9 — Modeling Direction

### Candidate Models

| Model | Rationale | Status |
|---|---|---|
| **SARIMA(1,1,1)(1,1,1)[7]** | Captures weekly cycle; parsimonious starting point from ACF/PACF | ✅ Fitted — initial candidate |
| **SARIMAX(1,1,1)(1,1,1)[7]** | Adds weather + holiday regressors; handles known external shocks | 🔄 In progress |
| **SARIMA grid search (AIC/BIC)** | Systematic order selection | 📅 Planned |
| **VAR (sales + temperature)** | Test for multivariate dynamics | 📅 Planned |
| **GARCH on residuals** | Check for conditional heteroskedasticity in promotion periods | 📅 Planned |

### Why SARIMA over ML baselines?
- Course focus: interpretable statistical inference, not predictive accuracy alone
- SARIMA provides **confidence intervals**, residual diagnostics, and formal hypothesis testing
- ML baselines (TFT, DLinear) in the Dingdong repo serve as benchmarks for final comparison

### Initial Fit: SARIMA(1,1,1)(1,1,1)[7]
- **AIC:** 234.7 | **BIC:** 248.9
- All AR and MA coefficients significant (|z| > 2)
- Ljung-Box Q-test on residuals: p = 0.41 → **no autocorrelation** remaining
- Residuals approximately normal (Jarque-Bera p = 0.08); slight excess kurtosis on promotion days

---

## Slide 10 — Residual Diagnostics (Initial Model)

*(See notebook: cell "Residual Diagnostics")*

**What the diagnostics show:**
1. **Residual time plot:** No obvious pattern; a few large outliers on promotion days (activity_flag = 1)
2. **Residual ACF:** All lags within 95% confidence bands → white noise confirmed
3. **Q-Q plot:** Near-normal with slightly heavy tails — driven by promotion spikes
4. **Histogram:** Roughly bell-shaped, mean ≈ 0

**Next steps from diagnostics:**
- Add `activity_flag` as dummy regressor in SARIMAX to absorb promotion spikes
- Investigate GARCH(1,1) on residuals — promotion days may introduce conditional heteroskedasticity
- Jarque-Bera marginal rejection: consider robust standard errors or bootstrap CI

---

## Slide 11 — 14-Day Forecast (Preliminary)

*(See notebook: cell "Forecast")*

- **Method:** In-sample fit on days 1–80, out-of-sample forecast days 81–94
- **MAPE (SARIMA baseline):** ~6.8%
- **MAE:** ~4,300 units/day
- Forecast intervals widen appropriately over horizon
- Model correctly captures weekend peaks in forecast period
- Misses a promotion-day spike on day 85 — confirms need for SARIMAX with `activity_flag`

---

## Slide 12 — Division of Labor & Timeline

| Task | Owner | Deadline |
|---|---|---|
| Data loading, cleaning, EDA | Member A | Done |
| ACF/PACF, stationarity tests | Member B | Done |
| SARIMA fitting & diagnostics | Member C | Done |
| SARIMAX with exogenous vars | Member A + B | Week 6 |
| Model comparison (AIC/BIC grid) | Member C | Week 6 |
| VAR / GARCH extension | Member B | Week 7 |
| Forecast evaluation (MAPE, RMSE) | Member A | Week 7 |
| Final report & presentation | All | Week 8 |

### Remaining Work
1. **SARIMAX** — incorporate weather and promotion dummies as external regressors
2. **Model selection** — systematic grid search over (p,d,q)(P,D,Q)[7] using AIC/BIC
3. **Censoring correction** — simple stockout-adjusted demand estimate for cleaner target
4. **Comparison against DLinear baseline** from the Dingdong repo
5. **Final visualizations and written report**

---

## Appendix A — Raw Output Tables

*(Full ADF/KPSS output, model summary tables, and ACF/PACF values available in notebook)*

## Appendix B — Data Loading Code

```python
from datasets import load_dataset
ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train")
df = ds.to_pandas()
```

## Appendix C — References

1. Dingdong Inc. (2025). *FreshRetailNet-50K*. Hugging Face. CC-BY-4.0.  
   https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
2. Dingdong Inc. (2025). *FRN-50K Baseline*. GitHub.  
   https://github.com/Dingdong-Inc/frn-50k-baseline
3. Box, G.E.P., Jenkins, G.M., Reinsel, G.C., Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
4. Hyndman, R.J., Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
