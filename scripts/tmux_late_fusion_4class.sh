#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_late_fusion_4class_final"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_DIR="$WORKDIR/checkpoints/runs"
LATEST_LOG_LINK="$LOG_DIR/latest_late_fusion_4class.log"
RUN_ROOT="$WORKDIR/checkpoints/runs/train_all_4class_gpu_final"
MANIFEST="$WORKDIR/data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv"
RUN_ID_PREFIX="daisee4_fusion"
COMMAND="start"
BATCH_SIZE=128
FEATURE_MODE="tsfresh"
WEIGHT_STEP=0.05
OBJECTIVE="balanced_accuracy"
POLL_SECONDS=120
LATENCY_THREADS=2
LATENCY_WARMUP=30
LATENCY_ITERS=200

shell_quote() {
  printf "%q" "$1"
}

usage() {
  cat <<'USAGE'
Usage: scripts/tmux_late_fusion_4class.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --session NAME
  --run-root PATH
  --manifest PATH
  --batch-size N
  --feature-mode basic|tsfresh|copur
  --weight-step V
  --objective balanced_accuracy|accuracy|f1_macro
  --poll-seconds N
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
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --weight-step) WEIGHT_STEP="$2"; shift 2 ;;
    --objective) OBJECTIVE="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --latency-threads) LATENCY_THREADS="$2"; shift 2 ;;
    --latency-warmup) LATENCY_WARMUP="$2"; shift 2 ;;
    --latency-iters) LATENCY_ITERS="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$RUNS_DIR"

start_session() {
  local timestamp run_log report_json report_csv runner_script
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_log="$LOG_DIR/late_fusion_4class_${RUN_ID_PREFIX}_${timestamp}.log"
  report_json="$RUN_ROOT/late_fusion_4class_${RUN_ID_PREFIX}_${timestamp}/summary.json"
  report_csv="$RUN_ROOT/late_fusion_4class_${RUN_ID_PREFIX}_${timestamp}/summary.csv"
  runner_script="$LOG_DIR/run_late_fusion_4class_${RUN_ID_PREFIX}_${timestamp}.sh"

  cat >"$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(shell_quote "$WORKDIR")
ln -sfn $(shell_quote "$run_log") $(shell_quote "$LATEST_LOG_LINK")
echo "=== 4-class late-fusion tmux started at \$(date) ===" | tee -a $(shell_quote "$run_log")
WORKDIR=$(shell_quote "$WORKDIR") CONDA_ENV=$(shell_quote "$CONDA_ENV") MANIFEST=$(shell_quote "$MANIFEST") \
RUN_ROOT=$(shell_quote "$RUN_ROOT") RUN_LOG=$(shell_quote "$run_log") REPORT_JSON=$(shell_quote "$report_json") \
REPORT_CSV=$(shell_quote "$report_csv") LATEST_LOG_LINK=$(shell_quote "$LATEST_LOG_LINK") \
BATCH_SIZE=$(shell_quote "$BATCH_SIZE") FEATURE_MODE=$(shell_quote "$FEATURE_MODE") WEIGHT_STEP=$(shell_quote "$WEIGHT_STEP") \
OBJECTIVE=$(shell_quote "$OBJECTIVE") POLL_SECONDS=$(shell_quote "$POLL_SECONDS") \
LATENCY_THREADS=$(shell_quote "$LATENCY_THREADS") LATENCY_WARMUP=$(shell_quote "$LATENCY_WARMUP") \
LATENCY_ITERS=$(shell_quote "$LATENCY_ITERS") bash $(shell_quote "$WORKDIR/scripts/late_fusion_4class_runner.sh")
echo "=== 4-class late-fusion tmux finished at \$(date) ===" | tee -a $(shell_quote "$run_log")
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
