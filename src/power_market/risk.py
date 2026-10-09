"""Empirical loss risk and transparent price-spread/cost stress scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_metrics(backtest: pd.DataFrame, *, confidence: float = 0.95) -> dict:
    if backtest.empty or not 0.5 < confidence < 1:
        raise ValueError("Need nonempty backtest and confidence between 0.5 and 1.")
    pnl = backtest.net_pnl_eur
    returns = backtest.capital_return
    sigma = returns.std(ddof=1)
    losses = -pnl
    quantile = float(losses.quantile(confidence))
    tail = losses[losses >= quantile]
    active = backtest.position_mwh.ne(0)
    return {
        "net_pnl_eur": float(pnl.sum()),
        "annualized_sharpe": float(returns.mean() / sigma * np.sqrt(365)) if sigma > 0 else None,
        "annualized_volatility": float(sigma * np.sqrt(365)) if np.isfinite(sigma) else None,
        "max_drawdown": float(backtest.drawdown.min()),
        "daily_historical_var_eur": max(0.0, quantile),
        "daily_historical_expected_shortfall_eur": max(0.0, float(tail.mean())),
        "confidence": confidence,
        "active_days": int(active.sum()),
        "win_rate_active_days": float(pnl[active].gt(0).mean()) if active.any() else None,
        "ending_equity_eur": float(backtest.equity_eur.iloc[-1]),
        "capital_exhausted": bool(backtest.equity_eur.le(0).any()),
        "observations": len(backtest),
    }


def stress_tests(backtest: pd.DataFrame, *, shock_eur_mwh: float = 50) -> pd.DataFrame:
    if not np.isfinite(shock_eur_mwh) or shock_eur_mwh < 0:
        raise ValueError("Shock must be finite and nonnegative.")
    position = backtest.position_mwh
    base_gross = backtest.gross_pnl_eur
    costs = backtest.cost_eur
    scenarios = {
        "observed": base_gross - costs,
        "double_costs": base_gross - 2 * costs,
        "spread_up": base_gross + position * shock_eur_mwh - costs,
        "spread_down": base_gross - position * shock_eur_mwh - costs,
        "adverse_each_position": base_gross - position.abs() * shock_eur_mwh - costs,
    }
    return pd.DataFrame([{"scenario": name, "total_pnl_eur": float(pnl.sum()),
                          "worst_day_eur": float(pnl.min()), "shock_eur_mwh": shock_eur_mwh}
                         for name, pnl in scenarios.items()])
