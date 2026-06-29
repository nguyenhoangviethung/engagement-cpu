#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:?}"
CONDA_ENV="${CONDA_ENV:?}"
MANIFEST="${MANIFEST:?}"
FULL_RUN_ROOT="${FULL_RUN_ROOT:?}"
TRAIN_ALL_RUN_ROOT="${TRAIN_ALL_RUN_ROOT:?}"
STRONG_RUN_ROOT="${STRONG_RUN_ROOT:?}"
FULL_RUN_LOG="${FULL_RUN_LOG:?}"
TRAIN_ALL_RUN_LOG="${TRAIN_ALL_RUN_LOG:?}"
STRONG_RUN_LOG="${STRONG_RUN_LOG:?}"
FULL_STATUS_FILE="${FULL_STATUS_FILE:?}"
FULL_LATEST_LOG_LINK="${FULL_LATEST_LOG_LINK:?}"
FULL_LATEST_RUN_LINK="${FULL_LATEST_RUN_LINK:?}"
TRAIN_ALL_LATEST_LOG_LINK="${TRAIN_ALL_LATEST_LOG_LINK:?}"
STRONG_LATEST_LOG_LINK="${STRONG_LATEST_LOG_LINK:?}"
MODELS="${MODELS:?}"
DEVICE="${DEVICE:?}"
BATCH_SIZE="${BATCH_SIZE:?}"
EPOCHS="${EPOCHS:?}"
PATIENCE="${PATIENCE:?}"
MIN_EPOCHS="${MIN_EPOCHS:?}"
LR="${LR:?}"
WEIGHT_DECAY="${WEIGHT_DECAY:?}"
OBJECTIVE="${OBJECTIVE:?}"
FEATURE_MODE="${FEATURE_MODE:?}"
DIM_REDUCTION="${DIM_REDUCTION:?}"
DIM_COMPONENTS="${DIM_COMPONENTS:?}"
OVERSAMPLE="${OVERSAMPLE:?}"
HIDDEN_SIZE="${HIDDEN_SIZE:?}"
NUM_LAYERS="${NUM_LAYERS:?}"
DROPOUT="${DROPOUT:?}"
NUM_HEADS="${NUM_HEADS:?}"
KERNEL_SIZE="${KERNEL_SIZE:?}"
TCN_BLOCKS="${TCN_BLOCKS:?}"
CPU_THREADS="${CPU_THREADS:?}"
XGB_THREADS="${XGB_THREADS:?}"
LATENCY_THREADS="${LATENCY_THREADS:?}"
LATENCY_WARMUP="${LATENCY_WARMUP:?}"
LATENCY_ITERS="${LATENCY_ITERS:?}"
USE_AMP="${USE_AMP:?}"
ISOLATE_MODELS="${ISOLATE_MODELS:?}"
NEURAL_THREADS="${NEURAL_THREADS:?}"
WAIT_SECONDS="${WAIT_SECONDS:?}"
WAIT_TIMEOUT_HOURS="${WAIT_TIMEOUT_HOURS:?}"
RESUME="${RESUME:?}"
SHUTDOWN_ON_COMPLETE="${SHUTDOWN_ON_COMPLETE:-1}"

mkdir -p "$FULL_RUN_ROOT" "$(dirname "$FULL_RUN_LOG")" "$(dirname "$TRAIN_ALL_RUN_LOG")" "$(dirname "$STRONG_RUN_LOG")"
touch "$FULL_RUN_LOG"
ln -sfn "$FULL_RUN_LOG" "$FULL_LATEST_LOG_LINK"
ln -sfn "$FULL_RUN_ROOT" "$FULL_LATEST_RUN_LINK"
printf 'stage\texit_code\tfinished_at\n' > "$FULL_STATUS_FILE"

log() {
  echo "$*" | tee -a "$FULL_RUN_LOG"
}

poweroff_now() {
  if sudo -n shutdown -h now; then
    return 0
  fi
  if sudo -n systemctl poweroff; then
    return 0
  fi
  if sudo -n poweroff; then
    return 0
  fi
  log "Unable to power off with sudo -n; check sudoers permissions."
  return 1
}

on_exit() {
  local status=$?
  set +e
  log "=== full 4-class pipeline finished at $(date); exit_status=$status ==="
  ln -sfn "$FULL_RUN_LOG" "$FULL_LATEST_LOG_LINK"
  ln -sfn "$FULL_RUN_ROOT" "$FULL_LATEST_RUN_LINK"
  if [[ "$SHUTDOWN_ON_COMPLETE" == "1" ]]; then
    log "Shutdown-on-complete enabled; syncing and powering off now."
    sync
    poweroff_now || true
  fi
}
trap on_exit EXIT

log "=== full 4-class pipeline started at $(date) ==="
log "manifest=$MANIFEST"
log "full_run_root=$FULL_RUN_ROOT"
log "train_all_run_root=$TRAIN_ALL_RUN_ROOT"
log "strong_run_root=$STRONG_RUN_ROOT"

train_all_status=0
set +e
WORKDIR="$WORKDIR" CONDA_ENV="$CONDA_ENV" MANIFEST="$MANIFEST" \
RUN_ROOT="$TRAIN_ALL_RUN_ROOT" RUN_LOG="$TRAIN_ALL_RUN_LOG" \
SUMMARY_JSON="$TRAIN_ALL_RUN_ROOT/train_all_summary.json" \
REPORT_JSON="$TRAIN_ALL_RUN_ROOT/report.json" \
REPORT_CSV="$TRAIN_ALL_RUN_ROOT/report.csv" \
HISTORY_JSONL="$TRAIN_ALL_RUN_ROOT/history.jsonl" \
LATEST_LOG_LINK="$TRAIN_ALL_LATEST_LOG_LINK" \
MODELS="$MODELS" DEVICE="$DEVICE" \
BATCH_SIZE="$BATCH_SIZE" EPOCHS="$EPOCHS" PATIENCE="$PATIENCE" MIN_EPOCHS="$MIN_EPOCHS" \
LR="$LR" WEIGHT_DECAY="$WEIGHT_DECAY" OBJECTIVE="$OBJECTIVE" FEATURE_MODE="$FEATURE_MODE" \
DIM_REDUCTION="$DIM_REDUCTION" DIM_COMPONENTS="$DIM_COMPONENTS" OVERSAMPLE="$OVERSAMPLE" \
HIDDEN_SIZE="$HIDDEN_SIZE" NUM_LAYERS="$NUM_LAYERS" DROPOUT="$DROPOUT" NUM_HEADS="$NUM_HEADS" \
KERNEL_SIZE="$KERNEL_SIZE" TCN_BLOCKS="$TCN_BLOCKS" CPU_THREADS="$CPU_THREADS" XGB_THREADS="$XGB_THREADS" \
LATENCY_THREADS="$LATENCY_THREADS" LATENCY_WARMUP="$LATENCY_WARMUP" LATENCY_ITERS="$LATENCY_ITERS" \
USE_AMP="$USE_AMP" ISOLATE_MODELS="$ISOLATE_MODELS" \
SHUTDOWN_ON_COMPLETE=0 \
bash "$WORKDIR/scripts/train_all_4class_runner.sh" 2>&1 | tee -a "$FULL_RUN_LOG"
train_all_status="${PIPESTATUS[0]}"
set -e
printf 'train_all\t%s\t%s\n' "$train_all_status" "$(date --iso-8601=seconds)" >> "$FULL_STATUS_FILE"
if [[ "$train_all_status" -ne 0 ]]; then
  log "train_all failed with exit code $train_all_status"
  exit "$train_all_status"
fi

log "train_all completed; starting strong follow-ups"
set +e
WORKDIR="$WORKDIR" CONDA_ENV="$CONDA_ENV" MANIFEST="$MANIFEST" \
BASE_RUN_ROOT="$TRAIN_ALL_RUN_ROOT" OUTPUT_ROOT="$STRONG_RUN_ROOT" \
RUN_LOG="$STRONG_RUN_LOG" STATUS_FILE="$STRONG_RUN_ROOT/status.tsv" \
FEATURE_MODE="$FEATURE_MODE" XGB_THREADS="$XGB_THREADS" NEURAL_THREADS="$NEURAL_THREADS" \
BATCH_SIZE="$BATCH_SIZE" WAIT_SECONDS="$WAIT_SECONDS" WAIT_TIMEOUT_HOURS="$WAIT_TIMEOUT_HOURS" \
RESUME="$RESUME" \
bash "$WORKDIR/scripts/strong_followups_4class_runner.sh" 2>&1 | tee -a "$FULL_RUN_LOG"
strong_status="${PIPESTATUS[0]}"
set -e
printf 'strong_followups\t%s\t%s\n' "$strong_status" "$(date --iso-8601=seconds)" >> "$FULL_STATUS_FILE"
if [[ "$strong_status" -ne 0 ]]; then
  log "strong follow-ups failed with exit code $strong_status"
  exit "$strong_status"
fi

log "Full 4-class pipeline complete."
