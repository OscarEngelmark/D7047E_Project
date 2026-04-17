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
.venv/bin/pip install --upgrade protobuf

echo "==> Configuring Ultralytics integrations..."
.venv/bin/python -c "from ultralytics import settings; settings.update({'wandb': True})"

echo "==> Emptying trash..."
rm -rf ~/.local/share/Trash/files/* ~/.local/share/Trash/info/* 2>/dev/null || true

echo "==> Done! Run 'source .venv/bin/activate' to activate the environment."
