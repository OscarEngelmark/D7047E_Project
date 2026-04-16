#!/bin/bash
set -e

echo "== Setting up the environment for the project..."

echo "==> Installing system dependencies..."
apt-get install -y git libgl1

echo "==> Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python -m venv .venv
else
    echo "    .venv already exists, skipping."
fi

echo "==> Installing Python dependencies..."
.venv/bin/pip install -r requirements.txt

echo "==> Done! Run 'source .venv/bin/activate' to activate the environment."
