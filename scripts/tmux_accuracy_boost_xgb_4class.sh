#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_accuracy_boost_xgb_4class"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
LATEST_LOG_LINK="$LOG_DIR/latest_accuracy_boost_xgb_4class.log"
MANIFEST="$WORKDIR/data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv"
COMMAND="start"
CPU_THREADS=4
N_ESTIMATORS=800
ROUND_STEP=25
FEATURE_MODE="tsfresh"

shell_quote() {
  printf "%q" "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --cpu-threads) CPU_THREADS="$2"; shift 2 ;;
    --n-estimators) N_ESTIMATORS="$2"; shift 2 ;;
    --round-step) ROUND_STEP="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$WORKDIR/checkpoints/runs"

start_session() {
  local timestamp run_root run_log report_json runner_script
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_root="$WORKDIR/checkpoints/runs/accuracy_boost_xgb_4class_${timestamp}"
  run_log="$LOG_DIR/accuracy_boost_xgb_4class_${timestamp}.log"
  report_json="$WORKDIR/checkpoints/runs/accuracy_boost_xgb_4class_${timestamp}/summary.json"
  runner_script="$LOG_DIR/run_accuracy_boost_xgb_4class_${timestamp}.sh"

  cat >"$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(shell_quote "$WORKDIR")
ln -sfn $(shell_quote "$run_log") $(shell_quote "$LATEST_LOG_LINK")
trap 'status=\$?; if [[ \$status -ne 0 ]]; then echo "=== accuracy boost xgb failed with exit code \$status at \$(date) ===" | tee -a $(shell_quote "$run_log"); fi' EXIT
echo "=== accuracy boost xgb started at \$(date) ===" | tee -a $(shell_quote "$run_log")
bash $(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") \
  env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.multiclass.accuracy_boost_xgb \
  --manifest $(shell_quote "$MANIFEST") \
  --output-dir $(shell_quote "$run_root") \
  --report-json $(shell_quote "$report_json") \
  --feature-mode $(shell_quote "$FEATURE_MODE") \
  --n-estimators $(shell_quote "$N_ESTIMATORS") \
  --round-step $(shell_quote "$ROUND_STEP") \
  --cpu-threads $(shell_quote "$CPU_THREADS") \
  --latency-warmup 30 \
  --latency-iters 200 2>&1 | tee -a $(shell_quote "$run_log")
echo "=== accuracy boost xgb finished at \$(date) ===" | tee -a $(shell_quote "$run_log")
EOF
  chmod +x "$runner_script"
  tmux new-session -d -s "$SESSION_NAME" "bash '$runner_script'"
  echo "Started tmux session: $SESSION_NAME"
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
  attach) tmux attach -t "$SESSION_NAME" ;;
  status) tmux list-sessions 2>/dev/null | grep "^$SESSION_NAME:" || { echo "Session '$SESSION_NAME' is not running."; exit 1; } ;;
  stop) tmux kill-session -t "$SESSION_NAME" && echo "Stopped tmux session: $SESSION_NAME" || { echo "Session '$SESSION_NAME' is not running."; exit 1; } ;;
  logs) tail -n 120 "$LATEST_LOG_LINK" ;;
esac
