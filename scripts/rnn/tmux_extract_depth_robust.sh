#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="depth_robust_$(date +%Y%m%d_%H%M%S)"
LABELS_CSV="$WORKDIR/data/processed/runs/daisee_4class_final_dataset/video_labels_4class.csv"

if [[ ! -f "$LABELS_CSV" ]]; then
  echo "Missing 4-class labels CSV: $LABELS_CSV" >&2
  exit 1
fi

# Keep full input resolution: shrinking frames can erase useful detail when the
# face is already far from the camera. The generic tmux runner creates an
# isolated feature directory and manifest under data/processed/runs/.
#
# depth_robust_v2 produces 168 values per frame:
#   25 geometry/scale/depth proxies + 25 canonical landmarks * (x, y, z)
#   + 52 blendshapes + 16 canonical-face transformation values
# velocity_std then appends velocity and window standard deviation, producing
# (30, 504) windows. These features require retraining; they are intentionally
# not shape-compatible with the legacy (30, 90) models.
exec bash "$WORKDIR/scripts/rnn/tmux_extract.sh" "$@" \
  --session engagement_extract_depth_robust \
  --run-id "$RUN_ID" \
  --frame-stride 1 \
  --max-frames 0 \
  --resize-width 0 \
  --labels "$LABELS_CSV" \
  --feature-set depth_robust_v2 \
  --temporal-enrichment velocity_std \
  --label-mode four_class
