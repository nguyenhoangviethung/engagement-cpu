#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Python not found: $PYTHON_BIN"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[init] Creating virtualenv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[init] Upgrading pip/setuptools/wheel"
pip install --upgrade pip setuptools wheel

echo "[init] Installing dependencies"
pip install -r requirements.txt

echo "[init] Smoke check imports"
PYTHONPATH="$ROOT_DIR/src" python - << 'PY'
import numpy, pandas, sklearn, xgboost, torch, torchvision, cv2, PIL, tqdm
print("OK: core imports")
PY

echo "[done] Project initialized."
echo "Use: source .venv/bin/activate"
echo "Then run with: PYTHONPATH=src python -m engagement_daisee.rnn.train --help"
