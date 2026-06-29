#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_train_full_4class"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_DIR="$WORKDIR/checkpoints/runs"
LATEST_LOG_LINK="$LOG_DIR/latest_full_train_4class.log"
LATEST_RUN_LINK="$RUNS_DIR/full_train_4class_latest"

COMMAND="start"
MANIFEST="$WORKDIR/data/processed/final_feature_manifest.csv"
DEVICE="cpu"
USE_AMP=0
BATCH_SIZE=256
EPOCHS=20
PATIENCE=6
MIN_EPOCHS=5
LR=3e-4
WEIGHT_DECAY=1e-4
OBJECTIVE="balanced_accuracy"
FEATURE_MODE="tsfresh"
DIM_REDUCTION="none"
DIM_COMPONENTS=128
OVERSAMPLE="none"
HIDDEN_SIZE=128
NUM_LAYERS=2
DROPOUT=0.25
NUM_HEADS=4
KERNEL_SIZE=5
TCN_BLOCKS=4
CPU_THREADS=8
XGB_THREADS=8
LATENCY_THREADS=4
LATENCY_WARMUP=30
LATENCY_ITERS=200
NEURAL_THREADS=8
WAIT_SECONDS=30
WAIT_TIMEOUT_HOURS=24
ISOLATE_MODELS=1
RESUME=1
SHUTDOWN_ON_COMPLETE=1

shell_quote() {
  printf "%q" "$1"
}

usage() {
  cat <<'USAGE'
Usage: scripts/tmux_train_full_4class.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --session NAME
  --manifest PATH
  --device NAME
  --no-amp
  --batch-size N
  --epochs N
  --patience N
  --min-epochs N
  --lr V
  --weight-decay V
  --objective balanced_accuracy|accuracy|f1_macro
  --feature-mode basic|tsfresh|copur
  --dim-reduction none|pca|svd
  --dim-components N
  --oversample none|random
  --hidden-size N
  --num-layers N
  --dropout V
  --num-heads N
  --kernel-size N
  --tcn-blocks N
  --cpu-threads N
  --xgb-threads N
  --neural-threads N
  --latency-threads N
  --latency-warmup N
  --latency-iters N
  --wait-seconds N
  --wait-timeout-hours N
  --no-isolate-models
  --no-resume
  --no-shutdown-on-complete
  --env NAME
  --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --no-amp) USE_AMP=0; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --min-epochs) MIN_EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --objective) OBJECTIVE="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --dim-reduction) DIM_REDUCTION="$2"; shift 2 ;;
    --dim-components) DIM_COMPONENTS="$2"; shift 2 ;;
    --oversample) OVERSAMPLE="$2"; shift 2 ;;
    --hidden-size) HIDDEN_SIZE="$2"; shift 2 ;;
    --num-layers) NUM_LAYERS="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --num-heads) NUM_HEADS="$2"; shift 2 ;;
    --kernel-size) KERNEL_SIZE="$2"; shift 2 ;;
    --tcn-blocks) TCN_BLOCKS="$2"; shift 2 ;;
    --cpu-threads) CPU_THREADS="$2"; shift 2 ;;
    --xgb-threads) XGB_THREADS="$2"; shift 2 ;;
    --neural-threads) NEURAL_THREADS="$2"; shift 2 ;;
    --latency-threads) LATENCY_THREADS="$2"; shift 2 ;;
    --latency-warmup) LATENCY_WARMUP="$2"; shift 2 ;;
    --latency-iters) LATENCY_ITERS="$2"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    --wait-timeout-hours) WAIT_TIMEOUT_HOURS="$2"; shift 2 ;;
    --no-isolate-models) ISOLATE_MODELS=0; shift ;;
    --no-resume) RESUME=0; shift ;;
    --no-shutdown-on-complete) SHUTDOWN_ON_COMPLETE=0; shift ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$RUNS_DIR"

start_session() {
  local timestamp full_run_root train_all_run_root strong_run_root full_run_log train_all_run_log strong_run_log status_file runner_script
  timestamp="$(date +%Y%m%d_%H%M%S)"
  full_run_root="$RUNS_DIR/full_train_4class_${timestamp}"
  train_all_run_root="$full_run_root/train_all"
  strong_run_root="$full_run_root/strong_followups"
  full_run_log="$LOG_DIR/full_train_4class_${timestamp}.log"
  train_all_run_log="$LOG_DIR/train_all_4class_${timestamp}.log"
  strong_run_log="$LOG_DIR/strong_followups_4class_${timestamp}.log"
  status_file="$full_run_root/status.tsv"
  runner_script="$LOG_DIR/run_full_train_4class_${timestamp}.sh"

  cat >"$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(shell_quote "$WORKDIR")
ln -sfn $(shell_quote "$full_run_log") $(shell_quote "$LATEST_LOG_LINK")
ln -sfn $(shell_quote "$full_run_root") $(shell_quote "$LATEST_RUN_LINK")
WORKDIR=$(shell_quote "$WORKDIR") CONDA_ENV=$(shell_quote "$CONDA_ENV") MANIFEST=$(shell_quote "$MANIFEST") \
FULL_RUN_ROOT=$(shell_quote "$full_run_root") TRAIN_ALL_RUN_ROOT=$(shell_quote "$train_all_run_root") \
STRONG_RUN_ROOT=$(shell_quote "$strong_run_root") FULL_RUN_LOG=$(shell_quote "$full_run_log") \
TRAIN_ALL_RUN_LOG=$(shell_quote "$train_all_run_log") STRONG_RUN_LOG=$(shell_quote "$strong_run_log") \
FULL_STATUS_FILE=$(shell_quote "$status_file") FULL_LATEST_LOG_LINK=$(shell_quote "$LATEST_LOG_LINK") \
FULL_LATEST_RUN_LINK=$(shell_quote "$LATEST_RUN_LINK") TRAIN_ALL_LATEST_LOG_LINK=$(shell_quote "$LOG_DIR/latest_train_all_4class.log") \
STRONG_LATEST_LOG_LINK=$(shell_quote "$LOG_DIR/latest_strong_followups_4class.log") \
MODELS=$(shell_quote "gru tcn gru_basic tiny_transformer bilstm cnn_gru_fusion hybrid residual_bigru_attn xgboost") \
DEVICE=$(shell_quote "$DEVICE") BATCH_SIZE=$(shell_quote "$BATCH_SIZE") EPOCHS=$(shell_quote "$EPOCHS") \
PATIENCE=$(shell_quote "$PATIENCE") MIN_EPOCHS=$(shell_quote "$MIN_EPOCHS") LR=$(shell_quote "$LR") \
WEIGHT_DECAY=$(shell_quote "$WEIGHT_DECAY") OBJECTIVE=$(shell_quote "$OBJECTIVE") FEATURE_MODE=$(shell_quote "$FEATURE_MODE") \
DIM_REDUCTION=$(shell_quote "$DIM_REDUCTION") DIM_COMPONENTS=$(shell_quote "$DIM_COMPONENTS") OVERSAMPLE=$(shell_quote "$OVERSAMPLE") \
HIDDEN_SIZE=$(shell_quote "$HIDDEN_SIZE") NUM_LAYERS=$(shell_quote "$NUM_LAYERS") DROPOUT=$(shell_quote "$DROPOUT") \
NUM_HEADS=$(shell_quote "$NUM_HEADS") KERNEL_SIZE=$(shell_quote "$KERNEL_SIZE") TCN_BLOCKS=$(shell_quote "$TCN_BLOCKS") \
CPU_THREADS=$(shell_quote "$CPU_THREADS") XGB_THREADS=$(shell_quote "$XGB_THREADS") LATENCY_THREADS=$(shell_quote "$LATENCY_THREADS") \
LATENCY_WARMUP=$(shell_quote "$LATENCY_WARMUP") LATENCY_ITERS=$(shell_quote "$LATENCY_ITERS") USE_AMP=$(shell_quote "$USE_AMP") \
ISOLATE_MODELS=$(shell_quote "$ISOLATE_MODELS") NEURAL_THREADS=$(shell_quote "$NEURAL_THREADS") \
WAIT_SECONDS=$(shell_quote "$WAIT_SECONDS") WAIT_TIMEOUT_HOURS=$(shell_quote "$WAIT_TIMEOUT_HOURS") RESUME=$(shell_quote "$RESUME") \
SHUTDOWN_ON_COMPLETE=$(shell_quote "$SHUTDOWN_ON_COMPLETE") \
bash $(shell_quote "$WORKDIR/scripts/train_full_4class_runner.sh")
EOF
  chmod +x "$runner_script"

  tmux new-session -d -s "$SESSION_NAME" "bash '$runner_script'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Manifest: $MANIFEST"
  echo "Full run root: $full_run_root"
  echo "Attach: scripts/tmux_train_full_4class.sh attach --session $SESSION_NAME"
  echo "Check status: scripts/tmux_train_full_4class.sh status --session $SESSION_NAME"
  echo "Log: $full_run_log"
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists."
      exit 1
    fi
    start_session
    ;;
  attach)
    tmux attach -t "$SESSION_NAME"
    ;;
  status)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux list-sessions | grep "^$SESSION_NAME:"
    else
      echo "Session '$SESSION_NAME' is not running."
      exit 1
    fi
    ;;
  stop)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "$SESSION_NAME"
      echo "Stopped tmux session: $SESSION_NAME"
    else
      echo "Session '$SESSION_NAME' is not running."
      exit 1
    fi
    ;;
  logs)
    if [[ -L "$LATEST_LOG_LINK" || -f "$LATEST_LOG_LINK" ]]; then
      tail -n 200 "$LATEST_LOG_LINK"
    else
      echo "No latest log found at $LATEST_LOG_LINK"
      exit 1
    fi
    ;;
  *)
    usage
    exit 1
    ;;
esac
