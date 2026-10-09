"""Hypothetical cash-settled daily price-change contracts; not physical trading."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(predictions: pd.DataFrame, model: str = "xgboost", *, threshold: float = 5,
             volume_mwh: float = 10, cost_per_mwh: float = 0.5,
             initial_capital: float = 100_000) -> pd.DataFrame:
    if model not in predictions or model in {"actual", "previous_price"}:
        raise ValueError("Select a forecast model column.")
    if predictions.empty or not isinstance(predictions.index, pd.DatetimeIndex):
        raise ValueError("Need nonempty predictions with a DatetimeIndex.")
    parameters = [threshold, volume_mwh, cost_per_mwh, initial_capital]
    if not np.isfinite(parameters).all() or min(parameters[:3]) < 0 or initial_capital <= 0:
        raise ValueError("Use finite nonnegative threshold, volume, costs, and positive capital.")
    if not predictions.index.is_monotonic_increasing or predictions.index.has_duplicates:
        raise ValueError("Predictions must have a unique chronological index.")
    if not np.isfinite(predictions[[model, "actual", "previous_price"]]).all().all():
        raise ValueError("Forecasts and prices must be finite.")
    frame = predictions[["actual", "previous_price", model]].copy()
    frame["signal_spread"] = frame[model] - frame.previous_price
    frame["position_mwh"] = np.where(frame.signal_spread.abs() > threshold, np.sign(frame.signal_spread) * volume_mwh, 0.0)
    frame["realized_spread"] = frame.actual - frame.previous_price
    frame["gross_pnl_eur"] = frame.position_mwh * frame.realized_spread
    # Each daily contract is opened and cash-settled: charge both legs every day.
    frame["cost_eur"] = 2 * cost_per_mwh * frame.position_mwh.abs()
    frame["net_pnl_eur"] = frame.gross_pnl_eur - frame.cost_eur
    frame["forecast_available"] = True
    # Skip trading on missing forecasts, but retain calendar days in risk estimates.
    frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="D", name="date"))
    frame["forecast_available"] = frame.forecast_available.eq(True)
    for column in ["position_mwh", "gross_pnl_eur", "cost_eur", "net_pnl_eur"]:
        frame[column] = frame[column].fillna(0.0)
    frame["equity_eur"] = initial_capital + frame.net_pnl_eur.cumsum()
    # Fixed exposure simulation uses returns on INITIAL capital, not compounding.
    frame["capital_return"] = frame.net_pnl_eur / initial_capital
    frame["drawdown"] = frame.equity_eur / frame.equity_eur.cummax().clip(lower=initial_capital) - 1
    return frame
