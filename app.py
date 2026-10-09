"""Streamlit research dashboard. Run: streamlit run app.py"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from power_market.backtest import simulate
from power_market.data import EXOGENOUS_COLUMNS, demo_prices, load_exogenous, load_prices, quality_report
from power_market.models import evaluate_forecasts, forecast_next_day
from power_market.risk import risk_metrics, stress_tests

st.set_page_config(page_title="European Power Market Research", page_icon="⚡", layout="wide")
st.title("European Power Market Research")
st.caption("Daily electricity price forecasts · hypothetical spread simulation · risk analysis")


@st.cache_data
def parse_prices(content: bytes, date_column: str | None, price_column: str | None,
                 country_column: str | None, wide: bool, dayfirst: bool) -> pd.DataFrame:
    return load_prices(BytesIO(content), date_column=date_column, price_column=price_column,
                       country_column=country_column, wide=wide, dayfirst=dayfirst)


@st.cache_data
def forecast(prices: pd.DataFrame, country: str, fraction: float, refit: int,
             exogenous: pd.DataFrame | None):
    return evaluate_forecasts(prices, country, test_fraction=fraction, refit_every=refit, exogenous=exogenous)


@st.cache_data
def next_forecast(prices: pd.DataFrame, country: str, exogenous: pd.DataFrame | None):
    return forecast_next_day(prices, country, exogenous)


with st.sidebar:
    st.header("Research inputs")
    source = st.radio("Data source", ["Local Kaggle CSV", "Upload CSV", "Synthetic demo"])
    uploaded = st.file_uploader("Price CSV", type=["csv"]) if source == "Upload CSV" else None
    with st.expander("CSV format"):
        st.caption("Prices must already be EUR/MWh. Intraday rows become equal-weight UTC daily averages.")
        date_column = st.text_input("Date column override") or None
        price_column = st.text_input("Price column override") or None
        country_column = st.text_input("Country column override") or None
        wide = st.checkbox("Wide format: date plus country price columns")
        dayfirst = st.checkbox("Day-first dates (DD/MM/YYYY)")
    exogenous_upload = st.file_uploader("Optional observed supply/demand/weather CSV", type=["csv"])

try:
    if source == "Synthetic demo":
        prices = demo_prices()
        st.warning("SYNTHETIC DEMO — generated test data. These results are not evidence about European markets.")
        source_label = "Synthetic demo (seed 42)"
    else:
        if source == "Local Kaggle CSV":
            files = sorted((Path(__file__).parent / "data" / "raw").rglob("*.csv"))
            if not files:
                st.info("Download the Kaggle data with scripts/download_dataset.py, upload a CSV, or select Synthetic demo.")
                st.stop()
            chosen = st.sidebar.selectbox("Local file", files, format_func=lambda path: path.name)
            content, source_label = chosen.read_bytes(), chosen.name
        else:
            if uploaded is None:
                st.info("Upload the Kaggle price CSV to begin.")
                st.stop()
            content, source_label = uploaded.getvalue(), uploaded.name
        prices = parse_prices(content, date_column, price_column, country_column, wide, dayfirst)
        st.info(f"Source: {source_label}. Confirm EUR/MWh units and country definitions before interpreting results.")
    exogenous = load_exogenous(BytesIO(exogenous_upload.getvalue())) if exogenous_upload else None
    country = st.sidebar.selectbox("Country / bidding zone", sorted(prices.country.unique()))
    fraction = st.sidebar.slider("Chronological test fraction", 0.1, 0.4, 0.2, 0.05)
    refit = st.sidebar.selectbox("Refit every N forecast days", [30, 14, 7])
    model = st.sidebar.selectbox("Simulation forecast", ["xgboost", "ridge", "autoregression", "seasonal_naive", "persistence"])
    threshold = st.sidebar.number_input("Signal threshold (EUR/MWh)", min_value=0.0, value=5.0)
    volume = st.sidebar.number_input("Exposure per daily contract (MWh)", min_value=0.0, value=10.0)
    cost = st.sidebar.number_input("Cost per leg (EUR/MWh)", min_value=0.0, value=0.5)
    capital = st.sidebar.number_input("Initial capital (EUR)", min_value=1.0, value=100_000.0)
    confidence = st.sidebar.selectbox("Historical VaR confidence", [0.95, 0.99])
    shock = st.sidebar.number_input("Spread stress (EUR/MWh)", min_value=0.0, value=50.0)
    selected = prices.loc[prices.country.eq(country)].copy()
    trends, forecasts, trading, risk, context = st.tabs(["Price trends", "Forecasts", "Simulation", "Risk", "Supply, demand & weather"])
    with trends:
        st.plotly_chart(px.line(selected, x="date", y="price", title=f"{country} daily average price (EUR/MWh)"), width="stretch")
        selected["rolling_price_std"] = selected.set_index("date").price.reindex(pd.date_range(selected.date.min(), selected.date.max())).rolling(30).std().reindex(selected.date).to_numpy()
        st.plotly_chart(px.line(selected, x="date", y="rolling_price_std", title="30-day price volatility (standard deviation, EUR/MWh)"), width="stretch")
        st.dataframe(quality_report(prices), hide_index=True)
        st.caption("Missing days are retained as gaps. Negative electricity prices are valid and preserved.")
    with st.spinner("Evaluating chronological rolling one-day forecasts…"):
        predictions, scores, audit = forecast(prices, country, fraction, refit, exogenous)
    with forecasts:
        st.write("Forecast origin: end of the previous day. Earlier test observations enter later one-day forecasts. This does not emulate the noon day-ahead auction gate closure.")
        st.dataframe(scores, hide_index=True)
        st.plotly_chart(px.line(predictions.reset_index(), x="date", y=["actual", "seasonal_naive", model], title="Held-out price forecasts (EUR/MWh)"), width="stretch")
        st.download_button("Download forecasts", predictions.to_csv().encode(), "forecasts.csv", "text/csv")
        st.write("Next date after the dataset")
        st.caption("Uses all available history. Check the date: historical data produces a historical-date forecast, not a current market forecast. Incomplete final UTC days can affect the forecast.")
        try:
            st.dataframe(next_forecast(prices, country, exogenous))
        except ValueError as exc:
            st.info(str(exc))
        with st.expander("Training cutoffs"):
            st.dataframe(audit, hide_index=True)
    backtest = simulate(predictions, model, threshold=threshold, volume_mwh=volume, cost_per_mwh=cost, initial_capital=capital)
    metrics = risk_metrics(backtest, confidence=confidence)
    with trading:
        st.warning("Hypothetical cash-settled daily price-change contract. Yesterday's average is an assumed reference, not an executable entry quote. There is no claimed physical trading or realized profitability.")
        st.caption("Signal = forecast − yesterday's price. Long/short when absolute signal exceeds the threshold. P&L = position × today's price change − two-leg costs. Model and threshold changes after inspecting results are exploratory.")
        st.metric("Simulated net P&L", f"€{metrics['net_pnl_eur']:,.2f}")
        st.plotly_chart(px.line(backtest.reset_index(), x="date", y="equity_eur", title="Simulated equity (EUR)"), width="stretch")
        st.plotly_chart(px.bar(backtest.reset_index(), x="date", y="position_mwh", title="Daily contract exposure (MWh)"), width="stretch")
        st.download_button("Download simulation", backtest.to_csv().encode(), "simulation.csv", "text/csv")
    with risk:
        cols = st.columns(4)
        cols[0].metric("Daily historical VaR", f"€{metrics['daily_historical_var_eur']:,.2f}")
        cols[1].metric("Daily expected shortfall", f"€{metrics['daily_historical_expected_shortfall_eur']:,.2f}")
        cols[2].metric("Maximum drawdown", f"{metrics['max_drawdown']:.2%}")
        sharpe = metrics["annualized_sharpe"]
        cols[3].metric("Annualized Sharpe", "Undefined" if sharpe is None else f"{sharpe:.2f}")
        if metrics["capital_exhausted"]:
            st.error("Simulated equity reaches zero or below. The fixed-exposure path continues mechanically; it is not a feasible funded strategy.")
        st.caption(f"VaR/ES use {confidence:.0%} empirical one-day EUR losses. Sharpe assumes zero risk-free rate and 365 days/year, with P&L normalized by initial capital. Finite-sample risk estimates and stress scenarios are not guarantees.")
        st.plotly_chart(px.area(backtest.reset_index(), x="date", y="drawdown", title="Drawdown from peak equity (fraction)"), width="stretch")
        st.dataframe(stress_tests(backtest, shock_eur_mwh=shock), hide_index=True)
        st.caption("Spread-up/down shocks apply on every simulated day; adverse_each_position moves against each position. Positions stay fixed; these are sensitivity tests, not probabilistic forecasts.")
    with context:
        if exogenous is None:
            st.info("This price dataset supplies no demand, generation, or weather observations. Upload date,country and demand_mw, generation_mw, renewables_mw, net_imports_mw and/or temperature_c. Only prior-day and prior-week observations enter forecasts.")
        else:
            merged = selected.merge(exogenous, on=["date", "country"], how="inner")
            st.caption("Contemporaneous exploratory relationships below are associations, not causal effects. Forecasts use lagged observations only.")
            if {"demand_mw", "renewables_mw"}.issubset(merged.columns):
                merged["residual_load_mw"] = merged.demand_mw - merged.renewables_mw
            if {"demand_mw", "generation_mw"}.issubset(merged.columns):
                merged["generation_minus_demand_mw"] = merged.generation_mw - merged.demand_mw
            for column in [c for c in [*EXOGENOUS_COLUMNS, "residual_load_mw", "generation_minus_demand_mw"] if c in merged]:
                st.plotly_chart(px.scatter(merged, x=column, y="price", title=f"Price versus {column}"), width="stretch")
            st.dataframe(merged.describe(), width="stretch")
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()
