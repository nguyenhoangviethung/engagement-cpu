#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:?}"
CONDA_ENV="${CONDA_ENV:?}"
MANIFEST="${MANIFEST:?}"
RUN_ROOT="${RUN_ROOT:?}"
RUN_LOG="${RUN_LOG:?}"
REPORT_JSON="${REPORT_JSON:?}"
REPORT_CSV="${REPORT_CSV:?}"
LATEST_LOG_LINK="${LATEST_LOG_LINK:?}"
BATCH_SIZE="${BATCH_SIZE:-128}"
FEATURE_MODE="${FEATURE_MODE:-tsfresh}"
WEIGHT_STEP="${WEIGHT_STEP:-0.05}"
OBJECTIVE="${OBJECTIVE:-balanced_accuracy}"
POLL_SECONDS="${POLL_SECONDS:-120}"
LATENCY_THREADS="${LATENCY_THREADS:-2}"
LATENCY_WARMUP="${LATENCY_WARMUP:-30}"
LATENCY_ITERS="${LATENCY_ITERS:-200}"

mkdir -p "$(dirname "$RUN_LOG")" "$(dirname "$REPORT_JSON")"
ln -sfn "$RUN_LOG" "$LATEST_LOG_LINK"

echo "=== 4-class late-fusion stage started at $(date) ===" | tee -a "$RUN_LOG"
echo "manifest=$MANIFEST" | tee -a "$RUN_LOG"
echo "run_root=$RUN_ROOT" | tee -a "$RUN_LOG"
echo "report_json=$REPORT_JSON" | tee -a "$RUN_LOG"

"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" \
  env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.multiclass.late_fusion \
  --manifest "$MANIFEST" \
  --run-root "$RUN_ROOT" \
  --output-json "$REPORT_JSON" \
  --batch-size "$BATCH_SIZE" \
  --feature-mode "$FEATURE_MODE" \
  --weight-step "$WEIGHT_STEP" \
  --objective "$OBJECTIVE" \
  --wait-for-summary \
  --poll-seconds "$POLL_SECONDS" \
  --latency-threads "$LATENCY_THREADS" \
  --latency-warmup "$LATENCY_WARMUP" \
  --latency-iters "$LATENCY_ITERS" 2>&1 | tee -a "$RUN_LOG"

echo "=== 4-class late-fusion stage finished at $(date) ===" | tee -a "$RUN_LOG"
ln -sfn "$RUN_LOG" "$LATEST_LOG_LINK"
echo "report_json=$REPORT_JSON" | tee -a "$RUN_LOG"
