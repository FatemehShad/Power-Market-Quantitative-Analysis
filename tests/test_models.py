import numpy as np
import pandas as pd
import pytest

from power_market.data import demo_prices
from power_market.models import evaluate_forecasts, feature_frame, forecast_next_day


def test_features_do_not_use_target_or_future_and_lags_are_calendar_days():
    data = demo_prices(days=220)
    original, columns = feature_frame(data, "Germany")
    cutoff = pd.Timestamp("2022-06-01")
    changed = data.copy()
    changed.loc[changed.date >= cutoff, "price"] += 10_000
    modified, _ = feature_frame(changed, "Germany")
    pd.testing.assert_frame_equal(original.loc[:cutoff, columns], modified.loc[:cutoff, columns])
    gap = data.loc[~data.date.eq(pd.Timestamp("2022-02-10"))]
    gap_features, _ = feature_frame(gap, "Germany")
    assert np.isnan(gap_features.loc["2022-02-17", "lag_7"])
    assert "price" not in columns


def test_rolling_forecasts_and_training_are_invariant_to_later_observations():
    data = demo_prices(days=220)
    before, scores, audit = evaluate_forecasts(data, "Germany", refit_every=14)
    changed = data.copy()
    cutoff = before.index[-10]
    changed.loc[changed.date >= cutoff, "price"] += 10_000
    after, _, _ = evaluate_forecasts(changed, "Germany", refit_every=14)
    columns = ["persistence", "seasonal_naive", "autoregression", "ridge", "xgboost"]
    pd.testing.assert_frame_equal(before.loc[before.index <= cutoff, columns], after.loc[after.index <= cutoff, columns])
    assert (audit.train_end < audit.forecast_start).all()
    assert len(scores) == 5 and scores.test_days.eq(len(before)).all()
    assert np.isfinite(before).all().all()
    np.testing.assert_allclose(before.persistence, before.previous_price)


def test_exogenous_features_are_lagged_observations():
    data = demo_prices(days=220)
    extra = data[["date", "country"]].copy()
    extra["temperature_c"] = np.arange(len(extra), dtype=float)
    extra["demand_mw"] = 1000.0
    frame, columns = feature_frame(data, "Germany", extra)
    date = pd.Timestamp("2022-02-01")
    prior_date = pd.Timestamp("2022-01-31")
    expected = extra.loc[extra.country.eq("Germany") & extra.date.eq(prior_date), "temperature_c"].iloc[0]
    assert frame.loc[date, "temperature_c_lag_1"] == expected
    assert "temperature_c" not in columns
    predictions, scores, _ = evaluate_forecasts(data, "Germany", exogenous=extra)
    assert np.isfinite(predictions).all().all()
    assert len(scores) == 5


def test_small_input_fails_with_actionable_message():
    with pytest.raises(ValueError, match="90 complete training"):
        evaluate_forecasts(demo_prices(days=60), "Germany")


def test_next_date_forecast_uses_only_available_labels():
    data = demo_prices(days=220)
    result = forecast_next_day(data, "Germany")
    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2022-08-09")
    assert result.training_end.iloc[0] < result.index[0]
    assert np.isfinite(result.drop(columns="training_end")).all().all()
    expected = data.loc[data.country.eq("Germany") & data.date.eq(data.date.max()), "price"].iloc[0]
    assert result.persistence.iloc[0] == expected
