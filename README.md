# European Power Market Quantitative Analysis

Python tools for European electricity price forecasting, hypothetical trading backtests, risk analysis, and an interactive Streamlit dashboard.

**Data:** [European Wholesale Electricity Prices on Kaggle](https://www.kaggle.com/datasets/lakshmi2305/european-wholesale-electricity-prices). Hourly prices are aggregated into daily UTC averages in EUR/MWh.

## Features

- Forecasting with persistence and weekly baselines, autoregression, Ridge, and XGBoost.
- Chronological evaluation, expanding-window refits, and next-dataset-date forecasts.
- Hypothetical trading signals, transaction costs, P&L, VaR, expected shortfall, Sharpe, drawdown, and stress tests.
- Streamlit charts, country filters, CSV exports, SQLite storage, and optional supply/demand/weather inputs.

## Quick start

With Conda installed, run from the repository directory:

```bash
conda create --name power-market python=3.12 pip --yes
conda activate power-market
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python scripts/download_dataset.py
power-market --data data/raw/european_wholesale_electricity_prices.csv --country Germany --output artifacts/germany
streamlit run app.py --server.headless true
```

If the download fails, download the CSV from Kaggle into `data/raw/` or upload it in the dashboard. Dataset licensing is separate from the code license. Raw data and generated outputs are excluded from Git.

For a clearly labeled synthetic demo without downloading data:

```bash
power-market --demo --country Germany --output artifacts/demo
```

Choose **Synthetic demo** in the dashboard to use generated data. Each CLI run requires a fresh output directory. Run `power-market --help` for options.

## Analysis assumptions

Forecasts target daily averages at the end of the previous day, rather than the noon auction gate closure. Backtests simulate an assumed cash-settled price-change contract; they do not establish executable trading profits. The price dataset contains no supply, demand, or weather observations; those require additional inputs.

See [methodology and usage](docs/METHODOLOGY.md) for assumptions, optional input schemas, outputs, and cloud setup, and [validation results](docs/VALIDATION.md) for the real-data evaluation.

## Tests

```bash
python -m pytest -q
```

CI runs the tests and a synthetic CLI workflow on Python 3.12.
