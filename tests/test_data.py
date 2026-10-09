from io import StringIO

import numpy as np
import pandas as pd
import pytest

from power_market.data import load_exogenous, load_prices, query_prices, store_prices


def test_kaggle_schema_uses_utc_and_keeps_negative_prices():
    csv = StringIO("Country,ISO3 Code,Datetime (UTC),Datetime (Local),Price (EUR/MWhe),Year,Month,Day,Hour\n"
                   "Germany,DEU,2023-01-01 00:00:00,2023-01-01 01:00:00,-10,2023,1,1,0\n"
                   "Germany,DEU,2023-01-01 23:00:00,2023-01-02 00:00:00,30,2023,1,1,23\n")
    result = load_prices(csv)
    assert len(result) == 1
    assert result.iloc[0].date == pd.Timestamp("2023-01-01")
    assert result.iloc[0].price == 10


@pytest.mark.parametrize("csv", [
    "date,country,price\n2023-01-01,Germany,NaN\n",
    "date,country,price\n2023-01-01,Germany,inf\n",
    "date,country,price\nbad-date,Germany,10\n",
    "date,country,price\n2023-01-01,Germany,10\n2023-01-01,Germany,11\n",
    "date,country,price\n2023-01-01,,10\n",
])
def test_invalid_or_duplicate_data_rejected(csv):
    with pytest.raises(ValueError):
        load_prices(StringIO(csv))


def test_wide_data_missing_cells_and_semicolon_long_input():
    wide = load_prices(StringIO("date,Germany,France\n2023-01-01,-5,10\n2023-01-02,,-7\n"), wide=True)
    assert len(wide) == 3
    assert wide.price.lt(0).sum() == 2
    result = load_prices(StringIO("date;country;price\n2023-01-01;Germany;-5\n"))
    assert result.iloc[0].price == -5


def test_missing_hourly_prices_are_excluded_and_reported_without_imputation():
    result = load_prices(StringIO("Country,Datetime (UTC),Price (EUR/MWhe),Hour\n"
                                  "Germany,2023-01-01 00:00:00,-10,0\n"
                                  "Germany,2023-01-01 01:00:00,,1\n"))
    assert result.iloc[0].price == -10
    report = result.attrs["import_report"]
    assert report["missing_price_rows_excluded"] == 1
    assert report["partial_utc_days_by_country"]["Germany"] == 1


def test_sql_roundtrip_uses_bound_country_parameter(tmp_path):
    country = "test'; DROP TABLE prices;--"
    frame = pd.DataFrame({"date": pd.to_datetime(["2023-01-01"]), "country": [country], "price": [-12.5]})
    path = tmp_path / "prices.sqlite"
    store_prices(frame, path)
    found = query_prices(path, country)
    assert len(found) == 1 and found.iloc[0].price == -12.5
    store_prices(frame, path)  # refreshing is repeatable without duplicate rows
    assert len(query_prices(path, country)) == 1


def test_exogenous_validation():
    good = load_exogenous(StringIO("date,country,demand_mw,net_imports_mw,temperature_c\n2023-01-01,Germany,100,-20,-5\n"))
    assert good.iloc[0].temperature_c == -5
    with pytest.raises(ValueError, match="nonnegative"):
        load_exogenous(StringIO("date,country,demand_mw\n2023-01-01,Germany,-1\n"))
    with pytest.raises(ValueError, match="unique"):
        load_exogenous(StringIO("date,country,demand_mw\n2023-01-01,Germany,1\n2023-01-01,Germany,2\n"))
