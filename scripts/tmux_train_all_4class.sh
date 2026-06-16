#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_train_all_4class"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
LATEST_LOG_LINK="$LOG_DIR/latest_train_all_4class.log"

COMMAND="start"
RUN_ID_PREFIX="daisee4"
MANIFEST="$WORKDIR/data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv"
MODELS="gru xgboost tcn gru_basic tiny_transformer bilstm stgcn cnn_gru_fusion hybrid residual_bigru_attn"
DEVICE="auto"
USE_AMP=1
BATCH_SIZE=128
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
CPU_THREADS=2
LATENCY_THREADS=2
LATENCY_WARMUP=30
LATENCY_ITERS=200

shell_quote() {
  printf "%q" "$1"
}

usage() {
  cat <<'USAGE'
Usage: scripts/tmux_train_all_4class.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --session NAME
  --run-id-prefix ID
  --manifest PATH
  --models "gru xgboost tcn ..."
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
  --latency-threads N
  --latency-warmup N
  --latency-iters N
  --env NAME
  --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --run-id-prefix) RUN_ID_PREFIX="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
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
    --latency-threads) LATENCY_THREADS="$2"; shift 2 ;;
    --latency-warmup) LATENCY_WARMUP="$2"; shift 2 ;;
    --latency-iters) LATENCY_ITERS="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$RUNS_CHECKPOINT_DIR"

start_session() {
  local timestamp run_root run_log summary_json report_json report_csv history_jsonl runner_script
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_root="$RUNS_CHECKPOINT_DIR/train_all_4class_${RUN_ID_PREFIX}_${timestamp}"
  run_log="$LOG_DIR/train_all_4class_${RUN_ID_PREFIX}_${timestamp}.log"
  summary_json="$run_root/train_all_summary.json"
  report_json="$run_root/report.json"
  report_csv="$run_root/report.csv"
  history_jsonl="$run_root/history.jsonl"
  runner_script="$LOG_DIR/run_train_all_4class_${RUN_ID_PREFIX}_${timestamp}.sh"

  cat >"$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(shell_quote "$WORKDIR")
ln -sfn $(shell_quote "$run_log") $(shell_quote "$LATEST_LOG_LINK")
echo "=== 4-class tmux pipeline started at \$(date) ===" | tee -a $(shell_quote "$run_log")
WORKDIR=$(shell_quote "$WORKDIR") CONDA_ENV=$(shell_quote "$CONDA_ENV") MANIFEST=$(shell_quote "$MANIFEST") \
RUN_ROOT=$(shell_quote "$run_root") RUN_LOG=$(shell_quote "$run_log") SUMMARY_JSON=$(shell_quote "$summary_json") \
REPORT_JSON=$(shell_quote "$report_json") REPORT_CSV=$(shell_quote "$report_csv") HISTORY_JSONL=$(shell_quote "$history_jsonl") \
LATEST_LOG_LINK=$(shell_quote "$LATEST_LOG_LINK") MODELS=$(shell_quote "$MODELS") DEVICE=$(shell_quote "$DEVICE") \
BATCH_SIZE=$(shell_quote "$BATCH_SIZE") EPOCHS=$(shell_quote "$EPOCHS") PATIENCE=$(shell_quote "$PATIENCE") \
MIN_EPOCHS=$(shell_quote "$MIN_EPOCHS") LR=$(shell_quote "$LR") WEIGHT_DECAY=$(shell_quote "$WEIGHT_DECAY") \
OBJECTIVE=$(shell_quote "$OBJECTIVE") FEATURE_MODE=$(shell_quote "$FEATURE_MODE") DIM_REDUCTION=$(shell_quote "$DIM_REDUCTION") \
DIM_COMPONENTS=$(shell_quote "$DIM_COMPONENTS") OVERSAMPLE=$(shell_quote "$OVERSAMPLE") HIDDEN_SIZE=$(shell_quote "$HIDDEN_SIZE") \
NUM_LAYERS=$(shell_quote "$NUM_LAYERS") DROPOUT=$(shell_quote "$DROPOUT") NUM_HEADS=$(shell_quote "$NUM_HEADS") \
KERNEL_SIZE=$(shell_quote "$KERNEL_SIZE") TCN_BLOCKS=$(shell_quote "$TCN_BLOCKS") CPU_THREADS=$(shell_quote "$CPU_THREADS") \
LATENCY_THREADS=$(shell_quote "$LATENCY_THREADS") LATENCY_WARMUP=$(shell_quote "$LATENCY_WARMUP") LATENCY_ITERS=$(shell_quote "$LATENCY_ITERS") \
USE_AMP=$(shell_quote "$USE_AMP") bash $(shell_quote "$WORKDIR/scripts/train_all_4class_runner.sh")
echo "=== 4-class tmux pipeline finished at \$(date) ===" | tee -a $(shell_quote "$run_log")
ln -sfn $(shell_quote "$run_log") $(shell_quote "$LATEST_LOG_LINK")
EOF
  chmod +x "$runner_script"

  tmux new-session -d -s "$SESSION_NAME" "bash '$runner_script'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Runner: $runner_script"
  echo "Log: $run_log"
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
      tail -n 120 "$LATEST_LOG_LINK"
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
