from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_runs_forecasts_and_country_interaction():
    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app)).run(timeout=120)
    if (app.parent / "data/raw/european_wholesale_electricity_prices.csv").exists():
        assert not at.exception and not at.error
        assert len(at.metric) >= 5  # actual downloaded CSV renders the research tabs
    at.sidebar.radio[0].set_value("Synthetic demo").run(timeout=120)
    assert not at.exception
    assert not at.error
    assert any("SYNTHETIC DEMO" in warning.value for warning in at.warning)
    country = next(widget for widget in at.sidebar.selectbox if widget.label == "Country / bidding zone")
    country.set_value("France").run(timeout=120)
    assert not at.exception and not at.error
    assert len(at.metric) >= 5
    assert len(at.dataframe) >= 3
