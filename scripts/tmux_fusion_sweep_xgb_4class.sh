#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_fusion_sweep_xgb_4class"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_DIR="$WORKDIR/checkpoints/runs"
LATEST_LOG_LINK="$LOG_DIR/latest_fusion_sweep_xgb_4class.log"
COMMAND="start"
RUN_ID_PREFIX="fusion_sweep_xgb_4class"
WEIGHT_STEP=0.05
MIN_ACCURACY=0.75
MIN_BALANCED_ACCURACY=0.75
MAX_ACCURACY=1.0
MAX_BALANCED_ACCURACY=1.0

shell_quote() {
  printf "%q" "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --run-id-prefix) RUN_ID_PREFIX="$2"; shift 2 ;;
    --weight-step) WEIGHT_STEP="$2"; shift 2 ;;
    --min-accuracy) MIN_ACCURACY="$2"; shift 2 ;;
    --min-balanced-accuracy) MIN_BALANCED_ACCURACY="$2"; shift 2 ;;
    --max-accuracy) MAX_ACCURACY="$2"; shift 2 ;;
    --max-balanced-accuracy) MAX_BALANCED_ACCURACY="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$RUNS_DIR"

start_session() {
  local timestamp run_log report_json runner_script
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_log="$LOG_DIR/${RUN_ID_PREFIX}_${timestamp}.log"
  report_json="$RUNS_DIR/${RUN_ID_PREFIX}_${timestamp}/summary.json"
  runner_script="$LOG_DIR/run_${RUN_ID_PREFIX}_${timestamp}.sh"

  cat >"$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(shell_quote "$WORKDIR")
ln -sfn $(shell_quote "$run_log") $(shell_quote "$LATEST_LOG_LINK")
trap 'status=\$?; if [[ \$status -ne 0 ]]; then echo "=== fusion sweep xgb failed with exit code \$status at \$(date) ===" | tee -a $(shell_quote "$run_log"); fi' EXIT
echo "=== fusion sweep xgb started at \$(date) ===" | tee -a $(shell_quote "$run_log")
bash $(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") \
  env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.multiclass.fusion_sweep_xgb \
  --output-json $(shell_quote "$report_json") \
  --weight-step $(shell_quote "$WEIGHT_STEP") \
  --min-accuracy $(shell_quote "$MIN_ACCURACY") \
  --min-balanced-accuracy $(shell_quote "$MIN_BALANCED_ACCURACY") \
  --max-accuracy $(shell_quote "$MAX_ACCURACY") \
  --max-balanced-accuracy $(shell_quote "$MAX_BALANCED_ACCURACY") \
  --latency-warmup 30 \
  --latency-iters 200 2>&1 | tee -a $(shell_quote "$run_log")
echo "=== fusion sweep xgb finished at \$(date) ===" | tee -a $(shell_quote "$run_log")
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
  status) tmux list-sessions 2>/dev/null | grep "^$SESSION_NAME:" ;;
  stop) tmux kill-session -t "$SESSION_NAME" ;;
  logs) tail -n 120 "$LATEST_LOG_LINK" ;;
esac
