#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:?}"
CONDA_ENV="${CONDA_ENV:?}"
MANIFEST="${MANIFEST:?}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:?}"
OUTPUT_ROOT="${OUTPUT_ROOT:?}"
RUN_LOG="${RUN_LOG:?}"
STATUS_FILE="${STATUS_FILE:?}"
FEATURE_MODE="${FEATURE_MODE:-tsfresh}"
XGB_THREADS="${XGB_THREADS:-8}"
NEURAL_THREADS="${NEURAL_THREADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
WAIT_TIMEOUT_HOURS="${WAIT_TIMEOUT_HOURS:-24}"
RESUME="${RESUME:-1}"

mkdir -p "$OUTPUT_ROOT" "$(dirname "$RUN_LOG")"
touch "$RUN_LOG"

log() {
  echo "$*" | tee -a "$RUN_LOG"
}

wait_for_train_all() {
  local summary="$BASE_RUN_ROOT/train_all_summary.json"
  local waited=0
  local max_wait=$((WAIT_TIMEOUT_HOURS * 3600))
  while [[ ! -s "$summary" ]]; do
    if [[ "$max_wait" -gt 0 && "$waited" -ge "$max_wait" ]]; then
      log "=== timed out waiting for train_all summary: $summary ==="
      return 1
    fi
    log "Waiting for train_all to finish: $BASE_RUN_ROOT ($(date))"
    sleep "$WAIT_SECONDS"
    waited=$((waited + WAIT_SECONDS))
  done
  log "Found train_all summary: $summary"
}

run_stage() {
  local stage="$1"
  shift
  local stage_dir="$OUTPUT_ROOT/$stage"
  local success_marker="$stage_dir/.success"
  mkdir -p "$stage_dir"

  if [[ "$RESUME" == "1" && -f "$success_marker" ]]; then
    log "=== stage $stage already succeeded; skipping ==="
    return 0
  fi

  log "=== stage $stage started at $(date) ==="
  set +e
  OMP_NUM_THREADS="$XGB_THREADS" MKL_NUM_THREADS="$XGB_THREADS" \
    OPENBLAS_NUM_THREADS="$XGB_THREADS" NUMEXPR_NUM_THREADS="$XGB_THREADS" \
    "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" \
    env PYTHONPATH="$WORKDIR/src" "$@" 2>&1 | tee -a "$RUN_LOG"
  local status="${PIPESTATUS[0]}"
  set -e

  printf '%s\t%s\t%s\n' "$stage" "$status" "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
  if [[ "$status" -eq 0 ]]; then
    touch "$success_marker"
    log "=== stage $stage finished at $(date) ==="
  else
    overall_status=1
    log "=== stage $stage failed with exit code $status; continuing ==="
  fi
}

log "=== strong 4-class follow-ups started at $(date) ==="
log "base_run_root=$BASE_RUN_ROOT"
log "manifest=$MANIFEST"
log "output_root=$OUTPUT_ROOT"
wait_for_train_all

printf 'stage\texit_code\tfinished_at\n' > "$STATUS_FILE"
overall_status=0

BOOST_DIR="$OUTPUT_ROOT/accuracy_boost_xgb"
TARGETED_DIR="$OUTPUT_ROOT/accuracy_targeted_xgb"

run_stage accuracy_boost_xgb \
  python -u -m engagement_daisee.multiclass.accuracy_boost_xgb \
  --manifest "$MANIFEST" \
  --output-dir "$BOOST_DIR" \
  --report-json "$BOOST_DIR/summary.json" \
  --feature-mode "$FEATURE_MODE" \
  --n-estimators 800 \
  --round-step 25 \
  --cpu-threads "$XGB_THREADS" \
  --latency-warmup 30 \
  --latency-iters 200

run_stage accuracy_targeted_xgb \
  python -u -m engagement_daisee.multiclass.accuracy_targeted_xgb \
  --manifest "$MANIFEST" \
  --output-dir "$TARGETED_DIR" \
  --report-json "$TARGETED_DIR/summary.json" \
  --feature-mode "$FEATURE_MODE" \
  --n-estimators 1400 \
  --round-step 25 \
  --cpu-threads "$XGB_THREADS" \
  --min-accuracy 0.76 \
  --min-balanced-accuracy 0.70 \
  --only-candidates strong_weight_d6 \
  --latency-warmup 30 \
  --latency-iters 200

run_stage fusion_sweep_xgb \
  python -u -m engagement_daisee.multiclass.fusion_sweep_xgb \
  --manifest "$MANIFEST" \
  --output-json "$OUTPUT_ROOT/fusion_sweep_xgb/summary.json" \
  --final-xgb-model "$BASE_RUN_ROOT/xgboost/model.json" \
  --final-xgb-preprocessor "$BASE_RUN_ROOT/xgboost/preprocessor.npz" \
  --boost-xgb-model "$BOOST_DIR/model.json" \
  --boost-xgb-preprocessor "$BOOST_DIR/preprocessor.npz" \
  --targeted-xgb-model "$TARGETED_DIR/model.json" \
  --targeted-xgb-preprocessor "$TARGETED_DIR/preprocessor.npz" \
  --feature-mode "$FEATURE_MODE" \
  --weight-step 0.05 \
  --min-accuracy 0.76 \
  --min-balanced-accuracy 0.70 \
  --latency-warmup 30 \
  --latency-iters 200

run_stage late_fusion \
  python -u -m engagement_daisee.multiclass.late_fusion \
  --manifest "$MANIFEST" \
  --run-root "$BASE_RUN_ROOT" \
  --output-json "$OUTPUT_ROOT/late_fusion/summary.json" \
  --batch-size "$BATCH_SIZE" \
  --feature-mode "$FEATURE_MODE" \
  --weight-step 0.05 \
  --objective balanced_accuracy \
  --latency-threads 4 \
  --latency-warmup 30 \
  --latency-iters 200

run_stage ordinal \
  python -u -m engagement_daisee.multiclass.novel_models_4class \
  --method ordinal \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_ROOT/ordinal" \
  --report-json "$OUTPUT_ROOT/ordinal/summary.json" \
  --n-estimators 500 \
  --round-step 25 \
  --cpu-threads "$XGB_THREADS" \
  --latency-warmup 20 \
  --latency-iters 100

run_stage minirocket \
  python -u -m engagement_daisee.multiclass.novel_models_4class \
  --method minirocket \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_ROOT/minirocket" \
  --report-json "$OUTPUT_ROOT/minirocket/summary.json" \
  --num-kernels 128 \
  --cpu-threads "$XGB_THREADS" \
  --latency-warmup 20 \
  --latency-iters 100

run_stage deep_forest \
  python -u -m engagement_daisee.multiclass.novel_models_4class \
  --method deep_forest \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_ROOT/deep_forest" \
  --report-json "$OUTPUT_ROOT/deep_forest/summary.json" \
  --n-estimators 120 \
  --folds 3 \
  --cpu-threads "$XGB_THREADS" \
  --latency-warmup 20 \
  --latency-iters 100

run_stage inception_lite_ensemble \
  python -u -m engagement_daisee.multiclass.inception_lite_experiment \
  --manifest "$MANIFEST" \
  --xgb-run-root "$BASE_RUN_ROOT" \
  --output-dir "$OUTPUT_ROOT/inception_lite_ensemble" \
  --report-json "$OUTPUT_ROOT/inception_lite_ensemble/summary.json" \
  --report-csv "$OUTPUT_ROOT/inception_lite_ensemble/summary.csv" \
  --device cpu \
  --batch-size "$BATCH_SIZE" \
  --epochs 24 \
  --patience 8 \
  --min-epochs 6 \
  --lr 2.5e-4 \
  --weight-decay 1e-4 \
  --objective accuracy \
  --min-balanced-accuracy 0.70 \
  --feature-mode "$FEATURE_MODE" \
  --hidden-size 160 \
  --num-blocks 4 \
  --dropout 0.20 \
  --cpu-threads "$NEURAL_THREADS" \
  --latency-threads 4 \
  --latency-warmup 30 \
  --latency-iters 200 \
  --no-amp

log "=== strong 4-class follow-ups finished at $(date); overall_status=$overall_status ==="
exit "$overall_status"
