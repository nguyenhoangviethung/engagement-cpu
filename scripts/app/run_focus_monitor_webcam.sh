#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."
ENV_NAME="thesis"

CAMERA_ID="${CAMERA_ID:-0}"
CALIB_SECS="${CALIB_SECS:-30}"

"$WORKDIR/scripts/lib/run_python.sh" --env "$ENV_NAME" --workdir "$WORKDIR" \
  env PYTHONPATH="$WORKDIR/src" \
  python -m engagement_daisee.app.focus_monitor \
  --camera-id "$CAMERA_ID" \
  --calibration-seconds "$CALIB_SECS"
