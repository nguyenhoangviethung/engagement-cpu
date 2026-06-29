#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-thesis}"

cd "$WORKDIR"

bash scripts/lib/run_python.sh --env "$CONDA_ENV" --workdir "$WORKDIR" \
  env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.mediapipe.extract_features "$@"
