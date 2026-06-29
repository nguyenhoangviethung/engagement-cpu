#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

mkdir -p checkpoints/runs/product_4class_fixed_triple_xgb logs

LOG_PATH="logs/product_4class_reproduction.log"
REPORT_PATH="checkpoints/runs/product_4class_fixed_triple_xgb/summary.json"

echo "=== product 4-class reproduction started ===" | tee "$LOG_PATH"
bash scripts/lib/run_python.sh --env thesis --workdir "$WORKDIR" \
  env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.multiclass.fusion_fixed_xgb \
  --output-json "$REPORT_PATH" \
  --weights 0.84,0.14,0.02 \
  --bias-power 0.42 \
  --temperature 1.15 \
  --latency-warmup 30 \
  --latency-iters 200 2>&1 | tee -a "$LOG_PATH"
echo "=== product 4-class reproduction finished ===" | tee -a "$LOG_PATH"
