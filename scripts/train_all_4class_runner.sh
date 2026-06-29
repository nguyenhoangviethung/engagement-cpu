#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:?}"
CONDA_ENV="${CONDA_ENV:?}"
MANIFEST="${MANIFEST:?}"
RUN_ROOT="${RUN_ROOT:?}"
RUN_LOG="${RUN_LOG:?}"
SUMMARY_JSON="${SUMMARY_JSON:?}"
REPORT_JSON="${REPORT_JSON:?}"
REPORT_CSV="${REPORT_CSV:?}"
HISTORY_JSONL="${HISTORY_JSONL:?}"
LATEST_LOG_LINK="${LATEST_LOG_LINK:?}"
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
XGB_THREADS="${XGB_THREADS:-$CPU_THREADS}"
LATENCY_THREADS="${LATENCY_THREADS:?}"
LATENCY_WARMUP="${LATENCY_WARMUP:?}"
LATENCY_ITERS="${LATENCY_ITERS:?}"
USE_AMP="${USE_AMP:?}"
ISOLATE_MODELS="${ISOLATE_MODELS:-1}"
SHUTDOWN_ON_COMPLETE="${SHUTDOWN_ON_COMPLETE:-0}"

mkdir -p "$(dirname "$RUN_LOG")" "$RUN_ROOT" "$(dirname "$REPORT_JSON")"
ln -sfn "$RUN_LOG" "$LATEST_LOG_LINK"

echo "=== 4-class train_all started at $(date) ===" | tee -a "$RUN_LOG"
echo "manifest=$MANIFEST" | tee -a "$RUN_LOG"
echo "run_root=$RUN_ROOT" | tee -a "$RUN_LOG"
echo "models=$MODELS" | tee -a "$RUN_LOG"

cmd=(
  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR"
  env OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"
  OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"
  PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.multiclass.train_all
  --manifest "$MANIFEST"
  --output-dir "$RUN_ROOT"
  --models $MODELS
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --min-epochs "$MIN_EPOCHS"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --objective "$OBJECTIVE"
  --feature-mode "$FEATURE_MODE"
  --dim-reduction "$DIM_REDUCTION"
  --dim-components "$DIM_COMPONENTS"
  --oversample "$OVERSAMPLE"
  --hidden-size "$HIDDEN_SIZE"
  --num-layers "$NUM_LAYERS"
  --dropout "$DROPOUT"
  --num-heads "$NUM_HEADS"
  --kernel-size "$KERNEL_SIZE"
  --tcn-blocks "$TCN_BLOCKS"
  --cpu-threads "$CPU_THREADS"
  --xgb-threads "$XGB_THREADS"
  --latency-threads "$LATENCY_THREADS"
  --latency-warmup "$LATENCY_WARMUP"
  --latency-iters "$LATENCY_ITERS"
  --report-json "$REPORT_JSON"
  --report-csv "$REPORT_CSV"
  --history-jsonl "$HISTORY_JSONL"
)

[[ "$USE_AMP" == "0" ]] && cmd+=(--no-amp)
[[ "$ISOLATE_MODELS" == "1" ]] && cmd+=(--isolate-models --resume)

"${cmd[@]}" 2>&1 | tee -a "$RUN_LOG"

echo "=== 4-class train_all finished at $(date) ===" | tee -a "$RUN_LOG"
ln -sfn "$RUN_LOG" "$LATEST_LOG_LINK"
echo "summary_json=$SUMMARY_JSON" | tee -a "$RUN_LOG"
echo "report_json=$REPORT_JSON" | tee -a "$RUN_LOG"
echo "report_csv=$REPORT_CSV" | tee -a "$RUN_LOG"

if [[ "$SHUTDOWN_ON_COMPLETE" == "1" ]]; then
  echo "Shutdown-on-complete enabled; powering off now." | tee -a "$RUN_LOG"
  sync
  if ! sudo -n shutdown -h now; then
    if ! sudo -n systemctl poweroff; then
      if ! sudo -n poweroff; then
        echo "Unable to power off with sudo -n; check sudoers permissions." | tee -a "$RUN_LOG"
        exit 1
      fi
    fi
  fi
fi
