"""Reproducible command-line research run with CSV/JSON and SQLite outputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from power_market.backtest import simulate
from power_market.data import demo_prices, load_exogenous, load_prices, quality_report, store_prices
from power_market.models import evaluate_forecasts, forecast_next_day
from power_market.risk import risk_metrics, stress_tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Local CSV in EUR/MWh")
    source.add_argument("--demo", action="store_true", help="Explicitly use synthetic data")
    parser.add_argument("--country", default="Germany")
    parser.add_argument("--date-column")
    parser.add_argument("--price-column")
    parser.add_argument("--country-column")
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--dayfirst", action="store_true")
    parser.add_argument("--exogenous", type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--refit-every", type=int, default=30)
    parser.add_argument("--model", choices=["persistence", "seasonal_naive", "autoregression", "ridge", "xgboost"], default="xgboost")
    parser.add_argument("--threshold", type=float, default=5)
    parser.add_argument("--volume-mwh", type=float, default=10)
    parser.add_argument("--cost-per-mwh", type=float, default=0.5)
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--shock", type=float, default=50)
    parser.add_argument("--output", type=Path, default=Path("artifacts/run"))
    args = parser.parse_args()
    try:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Output directory is nonempty; choose a new directory to preserve prior runs.")
        prices = demo_prices() if args.demo else load_prices(args.data, date_column=args.date_column,
                    price_column=args.price_column, country_column=args.country_column,
                    wide=args.wide, dayfirst=args.dayfirst)
        exogenous = load_exogenous(args.exogenous) if args.exogenous else None
        predictions, scores, audit = evaluate_forecasts(prices, args.country,
            test_fraction=args.test_fraction, refit_every=args.refit_every, exogenous=exogenous)
        backtest = simulate(predictions, args.model, threshold=args.threshold, volume_mwh=args.volume_mwh,
                            cost_per_mwh=args.cost_per_mwh, initial_capital=args.capital)
        metrics = risk_metrics(backtest, confidence=args.confidence)
        stress = stress_tests(backtest, shock_eur_mwh=args.shock)
        try:
            next_forecast = forecast_next_day(prices, args.country, exogenous)
            next_forecast_status = "Available for next dataset date; not necessarily a live forecast"
        except ValueError as exc:
            next_forecast = None
            next_forecast_status = str(exc)
        args.output.mkdir(parents=True, exist_ok=True)
        store_prices(prices, args.output / "prices.sqlite")
        for name, frame, index in [("forecasts", predictions, True), ("forecast_metrics", scores, False),
                                   ("training_audit", audit, False), ("backtest", backtest, True),
                                   ("stress_tests", stress, False), ("data_quality", quality_report(prices), False)]:
            frame.to_csv(args.output / f"{name}.csv", index=index)
        if next_forecast is not None:
            next_forecast.to_csv(args.output / "next_date_forecast.csv")
        (args.output / "risk_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        metadata = {
            "source": "SYNTHETIC DEMO" if args.demo else str(args.data),
            "source_sha256": None if args.demo else hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "exogenous_sha256": hashlib.sha256(args.exogenous.read_bytes()).hexdigest() if args.exogenous else None,
            "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "units": "EUR/MWh; daily equal-weight UTC averages",
            "import_report": prices.attrs.get("import_report", {}),
            "next_date_forecast_status": next_forecast_status,
            "forecast_protocol": "Rolling one-day forecast at end of D-1; expanding training with periodic refits",
            "strategy": "Hypothetical daily cash-settled price-change contract versus yesterday's average; not executable profits",
            "versions": {name: importlib.metadata.version(name) for name in ["power-market-quantitative-analysis", "numpy", "pandas", "scikit-learn", "xgboost", "statsmodels"]},
        }
        (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("SYNTHETIC DEMO" if args.demo else f"CSV source: {args.data}")
        print(scores.to_string(index=False))
        print(json.dumps(metrics, indent=2, allow_nan=False))
        print(f"Research artifacts saved to {args.output.resolve()}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Run failed: {exc}\n")


if __name__ == "__main__":
    main()
