# Validation of the supplied Kaggle dataset

Validated on 2026-10-09 with Python 3.12 and the versions in `requirements.lock`. The downloaded source CSV SHA-256 is `4b5af1e737d74ec307f807a2c4c8b5f37b36df6815532e91e1f69afaef05db77`.

## Input inspection

* 2,822,004 hourly rows across 31 countries.
* UTC timestamp range: 2015-01-01 00:00 through 2026-05-11 21:00; country coverage differs.
* 1,095 missing price rows, excluded and counted per country without imputation.
* No duplicate country/UTC timestamp keys.
* Raw price range: −500 to 4,000 EUR/MWh; negative prices retained.
* Germany: 4,149 daily averages, no missing daily dates, 32 negative-price days, three UTC days with fewer than 24 priced observations. Partial days remain in these research results.

The data is historical and does not cover the validation date. It supplies no demand, generation, or weather measurements. Fundamental-input support was tested with synthetic fixtures; its predictive benefit was not evaluated on real observations.

## German daily forecasting

Command:

```bash
power-market --data data/raw/european_wholesale_electricity_prices.csv \
  --country Germany --output artifacts/germany
```

Defaults: 20% chronological test fraction, 28-day lag/rolling warmup, expanding-window refits every 30 forecast observations, fixed hyperparameters. There are 3,296 initial training observations and 825 held-out observations, from **2024-02-07 through 2026-05-11**. Each forecast is issued at end of the prior day. Earlier test actuals can enter later one-day features/refits.

| Model | MAE (EUR/MWh) | RMSE (EUR/MWh) | Direction accuracy |
| --- | ---: | ---: | ---: |
| Persistence | 21.4472 | 29.0817 | 0.0000 |
| Weekly seasonal naive | 27.4254 | 38.8916 | 0.6642 |
| AR(1,2,7) | 19.3496 | 26.1098 | 0.6752 |
| Ridge | 17.5225 | 24.2134 | 0.7103 |
| XGBoost | 17.0465 | 23.7387 | 0.7018 |

Direction accuracy compares the sign of forecast-minus-yesterday-price with the sign of actual-minus-yesterday-price, excluding exactly zero actual movements. Persistence predicts zero movement, so it receives no directional credit when prices change. These results concern daily averages and this country/period, not individual hourly auction products or every market. They do not establish accuracy under a noon auction forecast origin.

## Hypothetical simulation and risk

The XGBoost simulation ran on the same 825-day holdout with a 5 EUR/MWh signal threshold, 10 MWh fixed daily exposure, 0.50 EUR/MWh per leg, and 100,000 EUR initial capital. The daily P&L, equity, risk tails, and stress scenarios were generated successfully. These are sensitivity calculations for an **assumed cash-settled price-change contract**. There is no demonstrated venue at which yesterday's daily average is an executable entry price, so apparent profitability and high Sharpe values cannot be interpreted as attainable trading performance.

Exact-cost, loss-tail, bankruptcy, negative-price, and initial-equity drawdown tests verify the simulation arithmetic. Future-data perturbation tests verify that changing prices from a future cutoff does not change earlier forecasts or the forecast at that cutoff. Training audit files expose all model cutoffs.

## Application and environment

All 24 tests passed. The suite exercises ingestion, SQL persistence, all five forecasting models, next-dataset-date prediction, optional lagged fundamental inputs, backtesting/risk calculations, CLI artifact generation, and Streamlit source/country interactions. A running Streamlit server also returned HTTP 200 with `ok` from its health endpoint. The actual local Kaggle data was loaded and rendered during the dashboard interaction check. No public hosting or preview deployment is implied.

The dependency/setup script is repeatable and preserves the existing CSV and source files. CI runs the fixture-based tests and a synthetic CLI run on Python 3.12; Kaggle availability is not required for CI. Full raw data, local SQLite databases, and run artifacts are excluded from the Git repository.
