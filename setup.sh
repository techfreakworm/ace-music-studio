#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .venv ]; then
  echo "Creating .venv (Python 3.11)…"
  python3.11 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip

if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "Apple Silicon detected — installing Mac requirements"
  pip install -r requirements.txt -r requirements-mac.txt
else
  echo "Non-Mac platform — installing CUDA-path requirements"
  pip install -r requirements.txt
fi

echo "Setup complete. Activate with: source .venv/bin/activate"
