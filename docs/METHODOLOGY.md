# Methodology and usage reference

## Forecast and simulation assumptions

Each forecast is issued **at the end of day D−1** for day D's daily average. All price/observed-weather features end at D−1; calendar values for D are known in advance. Earlier test-period observations may enter subsequent one-day forecasts and periodic refits, as in a rolling research evaluation. This is neither a fixed-origin multi-day forecast nor an emulation of the previous-day noon auction gate closure. Hyperparameters are fixed in code; no model is automatically selected on test scores. Choosing a model or threshold after viewing results is exploratory and would require another untouched evaluation period.

Missing source prices are excluded, with counts recorded per country; malformed nonmissing prices are rejected. Missing calendar days are reindexed before creating lags. No prices or fundamentals are forward-filled. Forecast rows with incomplete history are excluded; the trading series treats absent forecasts between its first and last date as zero-position, zero-P&L days. For the Kaggle hourly schema, days with fewer than 24 priced observations are counted in the quality report. Partial hourly days are averaged from the available rows, so incomplete hourly coverage and UTC/local market-day differences must be considered before operational use. A daily date-only CSV is accepted as supplied; no currency conversion or market-zone equivalence is assumed.

**Trading is a hypothetical daily cash-settled price-change contract** referenced to yesterday's average, not an executable physical power strategy. A signal `forecast − yesterday_price` selects +volume, −volume, or zero. Daily gross P&L is `position_mwh × (today_price − yesterday_price)`, with `2 × per_leg_cost × abs(position_mwh)` deducted because each contract opens and settles daily. Prior-day averages are not verified entry quotes. The simulation excludes real venues, liquidity, bid/ask spreads beyond the chosen cost, margin, funding, imbalance, capacity constraints, and execution slippage. It cannot demonstrate realized or attainable profits.

Returns equal daily net P&L divided by **initial** capital, with fixed exposure, zero risk-free rate, and 365 days/year for Sharpe/volatility. VaR and ES are positive empirical EUR loss amounts floored at zero; they are descriptive estimates from this finite test sample. A zero-volatility Sharpe is undefined. Equity can become nonpositive, which is flagged; the path continues mechanically for research and is not a feasible funded strategy. Stress scenarios apply price-spread shocks on every day without changing positions, rather than assigning scenario probabilities.

## Optional supply, demand, and weather data

The Kaggle price dataset contains no supply, demand, or weather measurements. Do not claim weather-driven forecasting was validated unless independently sourced inputs were supplied. Provide a UTF-8 CSV with daily UTC dates and country names matching the price data:

```text
date,country,demand_mw,generation_mw,renewables_mw,net_imports_mw,temperature_c
```

At least one numeric field is required; omit unavailable columns entirely. Each included field must be finite and each country/date unique. Demand, generation, and renewable output must be nonnegative. Imports may be signed. All energy fields here are **daily average MW**, not MWh totals. Example usage:

```bash
power-market --data data/raw/european_wholesale_electricity_prices.csv --country Germany \
  --exogenous /path/to/observed_fundamentals.csv --output artifacts/germany-fundamentals
```

Observed exogenous values enter at lags 1 and 7 only. Verify that these measurements were actually available at the stated forecast origin; publication delays can require larger lags. Contemporaneous dashboard scatter plots show associations, not causation. Supply minus demand excludes cross-border flows, storage, and losses unless separately accounted for.

## Outputs and validation

Every CLI run uses a fresh output directory and writes `forecasts.csv`, `forecast_metrics.csv`, `training_audit.csv`, `backtest.csv`, `risk_metrics.json`, `stress_tests.csv`, `data_quality.csv`, `prices.sqlite`, and `run_metadata.json` (input hashes, settings, versions, and assumptions). If recent history is complete, it also writes `next_date_forecast.csv`; otherwise the reason is recorded in metadata. Use `power-market --help` for configuration.

For the prepared Codex cloud workspace, `bash scripts/setup_cloud.sh` uses an environment outside the checkout at `/workspace/shared/power-market/.venv`. Use that environment's executables or activate it. Each cloud task is already isolated; reuse the checkout without creating a Git worktree unless explicitly requested. Streamlit processes need to be started again in new tasks; dependency files and downloaded data can be retained in the environment snapshot.

```bash
python -m pytest -q
```

Tests exercise parsing of the real Kaggle schema, negative prices, gaps, SQL round-trips, chronological forecast cutoffs, future-data invariance, optional fundamental lags, exact simulation costs, drawdown, risk tails, and Streamlit interactions. CI repeats the suite on Python 3.12 with synthetic fixtures, avoiding a dependency on Kaggle availability. See [validation results](VALIDATION.md) for results from the supplied dataset.

The repository is organized as `src/power_market/` for research logic, `app.py` for the dashboard, `scripts/` for dataset download, and `tests/` for verification. The analysis does not contact trading venues or place orders.
