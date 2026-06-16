#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_inception_lite_ensemble_4class"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
LATEST_LOG_LINK="$LOG_DIR/latest_inception_lite_ensemble_4class.log"

COMMAND="start"
MANIFEST="$WORKDIR/data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv"
XGB_RUN_ROOT="$WORKDIR/checkpoints/runs/train_all_4class_gpu_final"
RUN_ID_PREFIX="inception_lite_ensemble"
DEVICE="auto"
BATCH_SIZE=128
EPOCHS=24
PATIENCE=8
MIN_EPOCHS=6
LR=2.5e-4
WEIGHT_DECAY=1e-4
OBJECTIVE="accuracy"
MIN_BALANCED_ACCURACY=0.70
FEATURE_MODE="tsfresh"
HIDDEN_SIZE=160
NUM_BLOCKS=4
DROPOUT=0.20
CPU_THREADS=2
LATENCY_THREADS=2
LATENCY_WARMUP=30
LATENCY_ITERS=200
USE_AMP=1

shell_quote() {
  printf "%q" "$1"
}

usage() {
  cat <<'USAGE'
Usage: scripts/tmux_inception_lite_ensemble_4class.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --session NAME
  --run-id-prefix ID
  --manifest PATH
  --xgb-run-root PATH
  --device NAME
  --no-amp
  --batch-size N
  --epochs N
  --patience N
  --min-epochs N
  --lr V
  --weight-decay V
  --objective balanced_accuracy|accuracy|f1_macro
  --min-balanced-accuracy N
  --feature-mode basic|tsfresh|copur
  --hidden-size N
  --num-blocks N
  --dropout V
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
    --xgb-run-root) XGB_RUN_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --no-amp) USE_AMP=0; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --min-epochs) MIN_EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --objective) OBJECTIVE="$2"; shift 2 ;;
    --min-balanced-accuracy) MIN_BALANCED_ACCURACY="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --hidden-size) HIDDEN_SIZE="$2"; shift 2 ;;
    --num-blocks) NUM_BLOCKS="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --cpu-threads) CPU_THREADS="$2"; shift 2 ;;
    --latency-threads) LATENCY_THREADS="$2"; shift 2 ;;
    --latency-warmup) LATENCY_WARMUP="$2"; shift 2 ;;
    --latency-iters) LATENCY_ITERS="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

start_session() {
  local timestamp run_root run_log report_json report_csv runner_script
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_root="$WORKDIR/checkpoints/runs/${RUN_ID_PREFIX}_${timestamp}"
  run_log="$LOG_DIR/${RUN_ID_PREFIX}_${timestamp}.log"
  report_json="$WORKDIR/checkpoints/runs/${RUN_ID_PREFIX}_${timestamp}/summary.json"
  report_csv="$WORKDIR/checkpoints/runs/${RUN_ID_PREFIX}_${timestamp}/summary.csv"
  runner_script="$LOG_DIR/run_${RUN_ID_PREFIX}_${timestamp}.sh"

  cat >"$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(shell_quote "$WORKDIR")
ln -sfn $(shell_quote "$run_log") $(shell_quote "$LATEST_LOG_LINK")
trap 'status=\$?; if [[ \$status -ne 0 ]]; then echo "=== inception_lite ensemble failed with exit code \$status at \$(date) ===" | tee -a $(shell_quote "$run_log"); fi' EXIT
echo "=== inception_lite ensemble started at \$(date) ===" | tee -a $(shell_quote "$run_log")
WORKDIR=$(shell_quote "$WORKDIR") MANIFEST=$(shell_quote "$MANIFEST") XGB_RUN_ROOT=$(shell_quote "$XGB_RUN_ROOT") \
RUN_ID_PREFIX=$(shell_quote "$RUN_ID_PREFIX") REPORT_JSON=$(shell_quote "$report_json") REPORT_CSV=$(shell_quote "$report_csv") \
RUN_ROOT=$(shell_quote "$run_root") DEVICE=$(shell_quote "$DEVICE") BATCH_SIZE=$(shell_quote "$BATCH_SIZE") \
EPOCHS=$(shell_quote "$EPOCHS") PATIENCE=$(shell_quote "$PATIENCE") MIN_EPOCHS=$(shell_quote "$MIN_EPOCHS") \
LR=$(shell_quote "$LR") WEIGHT_DECAY=$(shell_quote "$WEIGHT_DECAY") OBJECTIVE=$(shell_quote "$OBJECTIVE") \
MIN_BALANCED_ACCURACY=$(shell_quote "$MIN_BALANCED_ACCURACY") \
FEATURE_MODE=$(shell_quote "$FEATURE_MODE") HIDDEN_SIZE=$(shell_quote "$HIDDEN_SIZE") NUM_BLOCKS=$(shell_quote "$NUM_BLOCKS") \
DROPOUT=$(shell_quote "$DROPOUT") CPU_THREADS=$(shell_quote "$CPU_THREADS") LATENCY_THREADS=$(shell_quote "$LATENCY_THREADS") \
LATENCY_WARMUP=$(shell_quote "$LATENCY_WARMUP") LATENCY_ITERS=$(shell_quote "$LATENCY_ITERS") USE_AMP=$(shell_quote "$USE_AMP") \
bash $(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") \
env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.multiclass.inception_lite_experiment \
  --manifest $(shell_quote "$MANIFEST") \
  --xgb-run-root $(shell_quote "$XGB_RUN_ROOT") \
  --output-dir $(shell_quote "$run_root") \
  --report-json $(shell_quote "$report_json") \
  --report-csv $(shell_quote "$report_csv") \
  --device $(shell_quote "$DEVICE") \
  --batch-size $(shell_quote "$BATCH_SIZE") \
  --epochs $(shell_quote "$EPOCHS") \
  --patience $(shell_quote "$PATIENCE") \
  --min-epochs $(shell_quote "$MIN_EPOCHS") \
  --lr $(shell_quote "$LR") \
  --weight-decay $(shell_quote "$WEIGHT_DECAY") \
  --objective $(shell_quote "$OBJECTIVE") \
  --min-balanced-accuracy $(shell_quote "$MIN_BALANCED_ACCURACY") \
  --feature-mode $(shell_quote "$FEATURE_MODE") \
  --hidden-size $(shell_quote "$HIDDEN_SIZE") \
  --num-blocks $(shell_quote "$NUM_BLOCKS") \
  --dropout $(shell_quote "$DROPOUT") \
  --cpu-threads $(shell_quote "$CPU_THREADS") \
  --latency-threads $(shell_quote "$LATENCY_THREADS") \
  --latency-warmup $(shell_quote "$LATENCY_WARMUP") \
  --latency-iters $(shell_quote "$LATENCY_ITERS") \
  $( [[ "$USE_AMP" -eq 0 ]] && printf "%s" "--no-amp" ) 2>&1 | tee -a $(shell_quote "$run_log")
echo "=== inception_lite ensemble finished at \$(date) ===" | tee -a $(shell_quote "$run_log")
ln -sfn $(shell_quote "$run_log") $(shell_quote "$LATEST_LOG_LINK")
EOF
  chmod +x "$runner_script"

  tmux new-session -d -s "$SESSION_NAME" "bash '$runner_script'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Runner: $runner_script"
  echo "Log: $run_log"
  echo "Report: $report_json"
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
