"""Normalize CSVs without silently filling gaps or discarding negative prices."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

ALIASES = {
    "date": {"date", "datetime", "datetimeutc", "timestamp", "time", "deliverydate"},
    "country": {"country", "countryname", "area", "market", "biddingzone"},
    "price": {"price", "priceeurmwh", "priceeurmwhe", "electricityprice", "wholesaleprice", "dayaheadprice", "value"},
}
EXOGENOUS_COLUMNS = ["demand_mw", "generation_mw", "renewables_mw", "net_imports_mw", "temperature_c"]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_prices(source: str | Path | IO, *, date_column: str | None = None,
                price_column: str | None = None, country_column: str | None = None,
                wide: bool = False, dayfirst: bool = False) -> pd.DataFrame:
    """Read long data, or an explicitly selected wide date/country matrix.

    Prices must be EUR/MWh. Intraday observations become equal-weight UTC daily
    averages. No currency/unit conversion or imputation is inferred.
    """
    raw = pd.read_csv(source)
    if len(raw.columns) == 1:
        if hasattr(source, "seek"):
            source.seek(0)
        raw = pd.read_csv(source, sep=None, engine="python")
    mapping = {"date": date_column, "price": price_column, "country": country_column}
    for canonical, explicit in mapping.items():
        if explicit:
            if explicit not in raw.columns:
                raise ValueError(f"Column {explicit!r} does not exist. Available: {list(raw.columns)}")
            continue
        matches = [c for c in raw.columns if _key(c) in ALIASES[canonical]]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {canonical} columns: {matches}; choose one explicitly.")
        mapping[canonical] = matches[0] if matches else None
    if not mapping["date"]:
        raise ValueError(f"No date column found. Available: {list(raw.columns)}")
    if wide:
        if mapping["price"] or mapping["country"]:
            raise ValueError("Wide mode expects only a date column and country price columns.")
        raw = raw.melt(id_vars=[mapping["date"]], var_name="country", value_name="price")
        mapping.update(country="country", price="price")
        raw = raw.dropna(subset=["price"])  # empty wide cells denote absent observations
    if not mapping["price"]:
        raise ValueError("No price column found. Set price_column, or use wide=True for a country matrix.")
    frame = pd.DataFrame({
        "date": pd.to_datetime(raw[mapping["date"]], errors="coerce", utc=True,
                               format="mixed", dayfirst=dayfirst),
        "price": pd.to_numeric(raw[mapping["price"]], errors="coerce"),
        "country": raw[mapping["country"]] if mapping["country"] else "Unspecified",
    })
    if frame.empty:
        raise ValueError("The input contains no price observations.")
    missing_price = raw[mapping["price"]].isna()
    invalid_price = ~missing_price & ~np.isfinite(frame["price"])
    if frame["date"].isna().any() or invalid_price.any():
        raise ValueError("Dates must be valid; nonmissing prices must be finite numeric EUR/MWh.")
    if frame["country"].isna().any():
        raise ValueError("Country names must be present.")
    frame["country"] = frame["country"].astype(str).str.strip()
    if frame["country"].eq("").any():
        raise ValueError("Country names must be nonempty.")
    if frame.duplicated(["country", "date"]).any():
        raise ValueError("Duplicate country/timestamp observations; resolve them before importing.")
    missing_by_country = frame.loc[missing_price].groupby("country").size().to_dict()
    original_rows = len(frame)
    frame = frame.loc[~missing_price].copy()
    if frame.empty:
        raise ValueError("No nonmissing price observations remain.")
    frame["date"] = frame["date"].dt.tz_convert(None).dt.normalize()
    hourly = {"Datetime (UTC)", "Hour"}.issubset(raw.columns)
    partial_by_country = {}
    if hourly:
        coverage = frame.groupby(["country", "date"]).size()
        partial_by_country = coverage.lt(24).groupby(level="country").sum().astype(int).to_dict()
    result = frame.groupby(["country", "date"], as_index=False)["price"].mean().sort_values(["country", "date"]).reset_index(drop=True)
    result.attrs["import_report"] = {
        "input_rows": original_rows, "missing_price_rows_excluded": int(missing_price.sum()),
        "missing_price_rows_by_country": {str(k): int(v) for k, v in missing_by_country.items()},
        "partial_utc_days_by_country": partial_by_country,
        "hourly_coverage_check": hourly,
    }
    return result


def load_exogenous(source: str | Path | IO) -> pd.DataFrame:
    """Optional observed daily demand/weather. Only lagged values enter models."""
    frame = pd.read_csv(source)
    required = {"date", "country"}
    columns = [c for c in EXOGENOUS_COLUMNS if c in frame]
    if not required.issubset(frame.columns) or not columns:
        raise ValueError("Exogenous CSV needs date, country, and at least one demand, generation, renewables, imports, or temperature column; see README.")
    frame = frame[["date", "country", *columns]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce", format="mixed").dt.tz_convert(None).dt.normalize()
    if frame["date"].isna().any() or frame["country"].isna().any():
        raise ValueError("Exogenous dates and countries must be valid.")
    frame["country"] = frame["country"].astype(str).str.strip()
    if frame["country"].eq("").any() or frame.duplicated(["country", "date"]).any():
        raise ValueError("Exogenous countries must be nonempty, with unique country/dates.")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} must contain finite observations.")
    for column in ["demand_mw", "generation_mw", "renewables_mw"]:
        if column in frame and frame[column].lt(0).any():
            raise ValueError(f"{column} must be nonnegative MW.")
    return frame


def demo_prices(days: int = 900, seed: int = 42) -> pd.DataFrame:
    """Synthetic data for exercising the app; never Kaggle-derived evidence."""
    if days < 1:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    t = np.arange(days)
    frames = []
    for country, offset in [("Germany", 0), ("France", 12), ("Spain", -8)]:
        noise = rng.normal(0, 7, days)
        residual = np.zeros(days)
        for i in range(1, days):
            residual[i] = 0.65 * residual[i - 1] + noise[i]
        prices = 75 + offset + 24 * np.cos(2 * np.pi * t / 365.25) + 9 * np.sin(2 * np.pi * t / 7) + residual
        prices[t % 113 == 0] -= 120
        frames.append(pd.DataFrame({"country": country, "date": dates, "price": prices}))
    return pd.concat(frames, ignore_index=True)


def store_prices(frame: pd.DataFrame, path: str | Path) -> None:
    """Persist normalized daily observations, transactionally, with a SQL key."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(str(r.country), r.date.strftime("%Y-%m-%d"), float(r.price)) for r in frame.itertuples()]
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS prices (country TEXT NOT NULL, date TEXT NOT NULL, price REAL NOT NULL, PRIMARY KEY(country, date))")
        connection.execute("DELETE FROM prices")
        connection.executemany("INSERT INTO prices VALUES (?, ?, ?)", rows)


def query_prices(path: str | Path, country: str) -> pd.DataFrame:
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query("SELECT date, country, price FROM prices WHERE country = ? ORDER BY date", connection, params=(country,), parse_dates=["date"])


def quality_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    report = frame.attrs.get("import_report", {})
    for country, group in frame.groupby("country"):
        expected = (group.date.max() - group.date.min()).days + 1
        rows.append({"country": country, "start": group.date.min(), "end": group.date.max(),
                     "observations": len(group), "missing_days": expected - len(group),
                     "negative_price_days": int(group.price.lt(0).sum()),
                     "excluded_missing_price_rows": report.get("missing_price_rows_by_country", {}).get(country, 0),
                     "partial_utc_days": report.get("partial_utc_days_by_country", {}).get(country)})
    return pd.DataFrame(rows)
