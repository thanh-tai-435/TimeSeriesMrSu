# Project: Fresh50K Two-Stage Demand Forecasting

This project builds a reproducible two-stage workflow for FreshRetailNet-50K: recover latent demand during stockouts, then forecast demand on the recovered target.

## Environment Setup

```bash
pip install -r requirements.txt
```

## Data

Put raw data files in:

```text
data/raw/
```

Current experiment setting:

```text
sample_frac = 0.1
frequency = hourly
```

The sample is selected at the `series_id` level, preserving each selected series timeline. Daily rows are expanded to hourly rows using `hours_sale` and `hours_stock_status`.

Processed files will be written to:

```text
data/processed/
```

## Run

```bash
python main_eda.py --sample_frac 0.1 --frequency hourly
python main_features.py --sample_frac 0.1 --frequency hourly
python main_split.py --val_days 7 --test_days 14
python main_benchmarks.py --val_days 7 --test_days 14
python main_train.py --horizon 1 --max_train_rows 1000000 --models ridge lightgbm
python main_train.py --horizon 24 --max_train_rows 1000000 --models ridge lightgbm
python main_ablation.py --horizon 1 --max_train_rows 500000
python main_diagnostics.py
python main_advanced.py
python main_owner_approach.py
python main_imputation_quality.py
python main_spectrum.py
python main_evaluate.py
python main_report.py
python main_report_vi.py
python main_slides_vi.py
```

For quick experiments:

```bash
python main_train.py --horizon 1 --max_train_rows 250000 --models lightgbm
```

## Outputs

```text
outputs/figures/
outputs/tables/
outputs/models/
outputs/reports/
```

## Notes

- Train/validation/test splits must be time-based.
- Rolling features must shift by one period before rolling to avoid leakage.
- `target_h1` means one hour ahead.
- `target_h24` means 24 hours ahead.
- Main metrics: RMSE, MAE, WAPE, and sMAPE.
- Random seed: 42.
- The owner-aligned pipeline is the main result: latent demand recovery first, then 7-day demand forecasting on recovered demand.
- The hourly h1/h24 models, ablation, quantile intervals, SARIMAX, and rolling-window checks are auxiliary diagnostics on the observed-sales benchmark.
- Two-stage diagnostics are produced by `main_owner_approach.py`: residual diagnostics, forecast intervals, bias comparison, and recovered-demand feature importance.
- The main two-stage comparison includes daily baselines and a validation-selected seasonal-ML hybrid.
- Imputation quality checks are produced by `main_imputation_quality.py`: pseudo-stockout validation, uplift by series/day, and capped-imputation sensitivity.
- Frequency-domain seasonality checks are produced by `main_spectrum.py` for hourly observed sales and daily recovered demand.
