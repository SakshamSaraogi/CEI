# Week 2 — Supervised Learning & Time Series

**Notebook:** `week2_saksham_saraogi_jecrc.ipynb`  
**Dataset:** `tesla_deliveries_dataset_2015_2025.csv`  
**Intern:** Saksham Saraogi · JECRC University · Celebal Technologies CEI

---

## Overview

An end-to-end machine learning pipeline built on Tesla EV delivery and production data (2015–2025). Covers the complete ML workflow from raw data cleaning through feature engineering, regression modelling, hyperparameter tuning, and SARIMA time series forecasting — with a strong focus on preventing data leakage and using time-aware evaluation strategies.

---

## Dataset

| Property | Value |
|----------|-------|
| File | `tesla_deliveries_dataset_2015_2025.csv` |
| Rows | ~1,320 |
| Columns | 12 |
| Target | `Estimated_Deliveries` |
| Time span | January 2015 – December 2025 |
| Granularity | Monthly, by model and region |

**Key columns:** `Year`, `Month`, `Model`, `Region`, `Source_Type`, `Estimated_Deliveries`, `Production_Units`, `Avg_Price_USD`, `Battery_Capacity_kWh`, `Range_km`, `CO2_Saved_tons`, `Charging_Stations`

---

## Sections Covered

| Section | Topic |
|---------|-------|
| 1 | Data Cleaning — nulls, duplicates, outliers (IQR + Z-score) |
| 2 | EDA — univariate distributions, correlation heatmap, temporal trends |
| 3 | Encoding — LabelEncoder, One-Hot Encoding |
| 4 | Feature Scaling — StandardScaler vs RobustScaler vs MinMaxScaler |
| 5 | Feature Engineering — date features, cyclical Month encoding (sin/cos), lag features, rolling means |
| 6 | Data Leakage — identifying and eliminating target-derived features |
| 7 | Sklearn Pipelines — chaining scaler + model, leakage-free CV |
| 8 | Chronological Split — 80/20 index-based split (no `train_test_split`) |
| 9 | Regression Models — Linear Regression, Ridge (L2), Lasso (L1) |
| 10 | Bias-Variance Tradeoff — learning curves, overfitting vs underfitting |
| 11 | Evaluation Metrics — MAE, RMSE, MAPE, R² |
| 12 | Cross Validation — `TimeSeriesSplit` (5-fold), per-fold R² |
| 13 | Hyperparameter Tuning — GridSearchCV (Ridge), RandomizedSearchCV (Lasso) |
| 14 | Time Series Components — trend, seasonality, residuals (multiplicative decomposition) |
| 15 | Stationarity — ADF + KPSS tests, differencing, ACF/PACF diagnostics |
| 16 | SARIMA — grid-searched orders, evaluation vs naive baselines, case study on white-noise data |

---

## Key Results

| Model | R² | RMSE | MAPE |
|-------|----|------|------|
| Linear Regression | ~0.998 | ~150 | ~1.7% |
| Ridge (tuned, α=10) | ~0.998 | ~150 | ~1.7% |
| Lasso (tuned) | ~0.998 | ~150 | ~1.7% |
| SARIMA (grid-searched) | — | — | ~5–6% |

**Feature importance:** `Del_Lag_1` and `Roll_3_mean` are the strongest predictors, confirming that lag-based features dominate regression performance on time-ordered data.

---

## Notable Design Choices

- **Chronological split** used throughout — random splitting on time series data inflates R² by letting the model see future patterns during training
- **RobustScaler** chosen over StandardScaler due to heavy-tailed delivery distributions
- **Cyclical Month encoding** (sin/cos) preserves the circular nature of the calendar — January and December are treated as adjacent
- **SARIMA case study**: demonstrates that fitting SARIMA to white-noise synthetic data (ADF-stationary, near-zero autocorrelations) produces a model that cannot beat a naive mean forecast — an important diagnostic lesson

---

## Libraries Used

```python
pandas, numpy, matplotlib, seaborn, scipy
sklearn (LinearRegression, Ridge, Lasso, Pipeline, GridSearchCV, TimeSeriesSplit, cross_val_score)
statsmodels (SARIMAX, adfuller, kpss, seasonal_decompose, plot_acf, plot_pacf)
```

---

## How to Run

1. Upload `tesla_deliveries_dataset_2015_2025.csv` to your Colab session
2. Update `DATA_PATH` in Section 0 to `/content/tesla_deliveries_dataset_2015_2025.csv`
3. Run all cells top to bottom (`Runtime → Run all`)

---

*Week 2 · Celebal Excellence Internship · Data Science · 2025*
