#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Power-Market-Quantitative-Analysis
mkdir -p /workspace/shared/power-market
if [ ! -x /workspace/shared/power-market/.venv/bin/python ]; then
    python3 -m venv /workspace/shared/power-market/.venv
fi
export PIP_CACHE_DIR=/workspace/shared/power-market/pip-cache
/workspace/shared/power-market/.venv/bin/python -m pip install -r requirements.lock
/workspace/shared/power-market/.venv/bin/python -m pip install --no-deps -e .
/workspace/shared/power-market/.venv/bin/python -m pip check
if [ ! -f data/raw/european_wholesale_electricity_prices.csv ]; then
    /workspace/shared/power-market/.venv/bin/python scripts/download_dataset.py
fi
