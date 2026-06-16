#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_novel_models_4class"
WAIT_SESSION="engagement_accuracy_boost_xgb_4class"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUN_DIR="$WORKDIR/checkpoints/runs"
MANIFEST="$WORKDIR/data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv"
LATEST_LOG_LINK="$LOG_DIR/latest_novel_models_4class.log"
COMMAND="start"

shell_quote() {
  printf "%q" "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --wait-session) WAIT_SESSION="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$RUN_DIR"

start_session() {
  local timestamp run_root run_log runner
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_root="$RUN_DIR/novel_models_4class_${timestamp}"
  run_log="$LOG_DIR/novel_models_4class_${timestamp}.log"
  runner="$LOG_DIR/run_novel_models_4class_${timestamp}.sh"

  cat >"$runner" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(shell_quote "$WORKDIR")
mkdir -p $(shell_quote "$run_root")
ln -sfn $(shell_quote "$run_log") $(shell_quote "$LATEST_LOG_LINK")
trap 'status=\$?; if [[ \$status -ne 0 ]]; then echo "=== novel models failed with exit code \$status at \$(date) ===" | tee -a $(shell_quote "$run_log"); fi' EXIT
echo "=== novel models pipeline started at \$(date) ===" | tee -a $(shell_quote "$run_log")
while tmux has-session -t $(shell_quote "$WAIT_SESSION") 2>/dev/null; do
  echo "Waiting for resource session $(shell_quote "$WAIT_SESSION") at \$(date)" | tee -a $(shell_quote "$run_log")
  sleep 30
done

run_method() {
  local method="\$1"
  shift
  echo "=== method \$method started at \$(date) ===" | tee -a $(shell_quote "$run_log")
  set +e
  bash $(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") \
    env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.multiclass.novel_models_4class \
    --method "\$method" \
    --manifest $(shell_quote "$MANIFEST") \
    --output-dir $(shell_quote "$run_root")/"\$method" \
    --report-json $(shell_quote "$run_root")/"\${method}"/summary.json \
    --cpu-threads 4 --latency-warmup 20 --latency-iters 100 "\$@" 2>&1 | tee -a $(shell_quote "$run_log")
  local status="\${PIPESTATUS[0]}"
  set -e
  if [[ "\$status" -eq 0 ]]; then
    echo "=== method \$method finished at \$(date) ===" | tee -a $(shell_quote "$run_log")
  else
    echo "=== method \$method failed with exit code \$status at \$(date); continuing ===" | tee -a $(shell_quote "$run_log")
    overall_status=1
  fi
}

overall_status=0
run_method ordinal --n-estimators 500 --round-step 25
run_method minirocket --num-kernels 128
run_method deep_forest --n-estimators 120 --folds 3
echo "=== novel models pipeline finished at \$(date) ===" | tee -a $(shell_quote "$run_log")
exit "\$overall_status"
EOF
  chmod +x "$runner"
  tmux new-session -d -s "$SESSION_NAME" "bash '$runner'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Runner: $runner"
  echo "Log: $run_log"
  echo "Run root: $run_root"
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
  logs) tail -n 160 "$LATEST_LOG_LINK" ;;
esac
