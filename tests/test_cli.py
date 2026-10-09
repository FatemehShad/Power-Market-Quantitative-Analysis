import json
import subprocess
import sys

import pandas as pd


def test_cli_creates_auditable_research_outputs_and_preserves_runs(tmp_path):
    output = tmp_path / "research"
    command = [sys.executable, "-m", "power_market.cli", "--demo", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "SYNTHETIC DEMO" in result.stdout
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["source"] == "SYNTHETIC DEMO"
    assert metadata["source_sha256"] is None
    forecasts = pd.read_csv(output / "forecasts.csv")
    assert len(forecasts) > 20
    assert (output / "prices.sqlite").is_file()
    next_date = pd.read_csv(output / "next_date_forecast.csv")
    assert len(next_date) == 1
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeated.returncode == 2
    assert "nonempty" in repeated.stderr
