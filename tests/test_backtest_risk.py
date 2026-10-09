import numpy as np
import pandas as pd
import pytest

from power_market.backtest import simulate
from power_market.risk import risk_metrics, stress_tests


def predictions():
    return pd.DataFrame({"actual": [90.0, 110.0], "previous_price": [100.0, 90.0], "xgboost": [105.0, 85.0]},
                        index=pd.date_range("2023-01-01", periods=2, name="date"))


def test_exact_positions_costs_equity_drawdown_and_loss_tails():
    result = simulate(predictions(), threshold=0, volume_mwh=2, cost_per_mwh=0.5, initial_capital=100)
    np.testing.assert_allclose(result.position_mwh, [2, -2])
    np.testing.assert_allclose(result.cost_eur, [2, 2])
    np.testing.assert_allclose(result.net_pnl_eur, [-22, -42])
    np.testing.assert_allclose(result.equity_eur, [78, 36])
    np.testing.assert_allclose(result.drawdown, [-0.22, -0.64])
    metrics = risk_metrics(result, confidence=0.75)
    assert metrics["daily_historical_var_eur"] == 37
    assert metrics["daily_historical_expected_shortfall_eur"] == 42
    assert metrics["max_drawdown"] == pytest.approx(-0.64)
    stress = stress_tests(result, shock_eur_mwh=50).set_index("scenario")
    assert stress.loc["adverse_each_position", "total_pnl_eur"] == -264
    assert stress.loc["double_costs", "total_pnl_eur"] == -68


def test_no_trades_zero_losses_and_undefined_sharpe():
    result = simulate(predictions(), threshold=100, initial_capital=100)
    metrics = risk_metrics(result)
    assert metrics["active_days"] == 0
    assert metrics["annualized_sharpe"] is None
    assert metrics["daily_historical_var_eur"] == 0
    assert metrics["max_drawdown"] == 0


def test_calendar_gaps_are_zero_position_days():
    frame = predictions()
    frame.index = pd.to_datetime(["2023-01-01", "2023-01-03"])
    result = simulate(frame, initial_capital=100)
    assert len(result) == 3
    assert result.loc["2023-01-02", "net_pnl_eur"] == 0
    assert not result.loc["2023-01-02", "forecast_available"]


def test_bankruptcy_is_flagged_and_negative_prices_work():
    frame = predictions()
    frame["actual"] = [-1000, -2000]
    metrics = risk_metrics(simulate(frame, threshold=0, volume_mwh=10, initial_capital=100))
    assert metrics["capital_exhausted"]


@pytest.mark.parametrize("settings", [{"volume_mwh": -1}, {"initial_capital": 0}, {"threshold": float("nan")}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        simulate(predictions(), **settings)
