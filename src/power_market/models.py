"""Expanding-window, rolling one-day forecasts with auditable training cutoffs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from power_market.data import EXOGENOUS_COLUMNS

AR_FEATURES = ["lag_1", "lag_2", "lag_7"]


def feature_frame(prices: pd.DataFrame, country: str,
                  exogenous: pd.DataFrame | None = None, *, include_next_day: bool = False) -> tuple[pd.DataFrame, list[str]]:
    series = prices.loc[prices.country.eq(country), ["date", "price"]].set_index("date").sort_index()
    if series.empty:
        raise ValueError(f"No observations for {country!r}.")
    # Reindex before shifting: lag_7 always means seven calendar days, including gaps.
    end = series.index.max()
    if include_next_day:
        end = pd.date_range(end, periods=2, freq="D")[-1]
    frame = series.reindex(pd.date_range(series.index.min(), end, freq="D"))
    frame.index.name = "date"
    features = []
    for lag in [1, 2, 3, 7, 14, 28]:
        name = f"lag_{lag}"
        frame[name] = frame.price.shift(lag)
        features.append(name)
    for window in [7, 28]:
        for statistic in ["mean", "std"]:
            name = f"rolling_{statistic}_{window}"
            frame[name] = getattr(frame.price.shift(1).rolling(window), statistic)()
            features.append(name)
    frame["weekday"] = frame.index.dayofweek
    frame["year_sin"] = np.sin(2 * np.pi * frame.index.dayofyear / 365.25)
    frame["year_cos"] = np.cos(2 * np.pi * frame.index.dayofyear / 365.25)
    features.extend(["weekday", "year_sin", "year_cos"])
    if exogenous is not None:
        extra = exogenous.loc[exogenous.country.eq(country)].set_index("date")
        if extra.empty:
            raise ValueError(f"No exogenous observations for {country!r}.")
        for column in [c for c in EXOGENOUS_COLUMNS if c in extra]:
            for lag in [1, 7]:
                name = f"{column}_lag_{lag}"
                frame[name] = extra[column].reindex(frame.index).shift(lag)
                features.append(name)
    return frame, features


def _fit_predict(training: pd.DataFrame, testing: pd.DataFrame, features: list[str]) -> dict:
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    xgb = XGBRegressor(n_estimators=180, max_depth=3, learning_rate=0.05,
                       subsample=1.0, colsample_bytree=1.0, n_jobs=2,
                       random_state=42, objective="reg:squarederror")
    result = {}
    for name, model in [("ridge", ridge), ("xgboost", xgb)]:
        model.fit(training[features], training.price)
        result[name] = model.predict(testing[features])
    ar = sm.OLS(training.price, sm.add_constant(training[AR_FEATURES], has_constant="add")).fit()
    result["autoregression"] = ar.predict(sm.add_constant(testing[AR_FEATURES], has_constant="add")).to_numpy()
    return result


def forecast_next_day(prices: pd.DataFrame, country: str,
                      exogenous: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predict the next dataset date, refitting on available historical labels.

    A stale input produces a historical-date forecast, not a live market signal.
    """
    frame, features = feature_frame(prices, country, exogenous, include_next_day=True)
    training = frame.iloc[:-1].dropna(subset=["price", *features])
    testing = frame.iloc[-1:]
    if len(training) < 90 or testing[features].isna().any().any():
        raise ValueError("Next-date forecast needs 90 complete training days and complete recent price/fundamental history.")
    result = pd.DataFrame({"persistence": testing.lag_1, "seasonal_naive": testing.lag_7})
    for name, values in _fit_predict(training, testing, features).items():
        result[name] = values
    result["training_end"] = training.index.max()
    return result


def evaluate_forecasts(prices: pd.DataFrame, country: str, *, test_fraction: float = 0.2,
                       refit_every: int = 30, exogenous: pd.DataFrame | None = None,
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Forecast at end of D-1; earlier test actuals may enter later forecasts.

    Fixed hyperparameters; no tuning/selection on the test period. Refit models
    periodically on expanding prior observations. Each feature uses only history
    or target-day calendar data. This is not a multi-day forecast from one origin.
    """
    if not 0.05 <= test_fraction <= 0.5 or refit_every < 1:
        raise ValueError("test_fraction must be 0.05–0.5; refit_every must be positive.")
    frame, features = feature_frame(prices, country, exogenous)
    clean = frame.dropna(subset=["price", *features])
    split = int(len(clean) * (1 - test_fraction))
    if split < 90 or len(clean) - split < 20:
        raise ValueError("Need at least 90 complete training days and 20 test days after 28-day feature warmup/gaps.")
    predictions = clean.iloc[split:][["price", "lag_1", "lag_7"]].copy()
    predictions.columns = ["actual", "previous_price", "seasonal_naive"]
    predictions["persistence"] = predictions.previous_price
    audits = []
    for start in range(split, len(clean), refit_every):
        training = clean.iloc[:start]
        testing = clean.iloc[start:start + refit_every]
        for name, values in _fit_predict(training, testing, features).items():
            predictions.loc[testing.index, name] = values
        audits.append({"train_start": training.index.min(), "train_end": training.index.max(),
                       "forecast_start": testing.index.min(), "forecast_end": testing.index.max(),
                       "training_rows": len(training)})
    prediction_columns = ["persistence", "seasonal_naive", "autoregression", "ridge", "xgboost"]
    scores = []
    for model in prediction_columns:
        error = predictions[model] - predictions.actual
        movement = predictions.actual - predictions.previous_price
        signal = predictions[model] - predictions.previous_price
        nonzero = movement.ne(0)
        scores.append({"model": model, "MAE_EUR_MWh": float(error.abs().mean()),
                       "RMSE_EUR_MWh": float(np.sqrt((error ** 2).mean())),
                       "direction_accuracy": float((np.sign(signal[nonzero]) == np.sign(movement[nonzero])).mean()) if nonzero.any() else None,
                       "test_days": len(predictions)})
    return predictions, pd.DataFrame(scores), pd.DataFrame(audits)
