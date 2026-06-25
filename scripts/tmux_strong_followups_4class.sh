#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_strong_followups_4class"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_DIR="$WORKDIR/checkpoints/runs"
LATEST_LOG_LINK="$LOG_DIR/latest_strong_followups_4class.log"
LATEST_RUN_LINK="$RUNS_DIR/strong_followups_4class_latest"
MANIFEST=""
BASE_RUN_ROOT=""
COMMAND="start"
FEATURE_MODE="tsfresh"
XGB_THREADS=8
NEURAL_THREADS=8
BATCH_SIZE=256
WAIT_SECONDS=30
WAIT_TIMEOUT_HOURS=24
RESUME=1

shell_quote() {
  printf '%q' "$1"
}

resolve_latest_depth_manifest() {
  local latest
  latest="$(find "$WORKDIR/data/processed/runs" -maxdepth 2 -type f -name feature_manifest.csv \
    -path '*/extract_depth_robust*/*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -n "$latest" ]]; then
    printf '%s\n' "$latest"
    return 0
  fi
  printf '%s\n' "$WORKDIR/data/processed/runs/extract_depth_robust_5w_20260620_061230/feature_manifest.csv"
}

usage() {
  cat <<'USAGE'
Usage: scripts/tmux_strong_followups_4class.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --session NAME
  --base-run-root PATH   train_all run to follow; default: newest depth-robust run
  --manifest PATH
  --feature-mode basic|tsfresh|copur
  --xgb-threads N
  --neural-threads N
  --batch-size N
  --wait-seconds N
  --wait-timeout-hours N
  --no-resume
  --env NAME
  --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --base-run-root) BASE_RUN_ROOT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --xgb-threads) XGB_THREADS="$2"; shift 2 ;;
    --neural-threads) NEURAL_THREADS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    --wait-timeout-hours) WAIT_TIMEOUT_HOURS="$2"; shift 2 ;;
    --no-resume) RESUME=0; shift ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$MANIFEST" ]]; then
  MANIFEST="$(resolve_latest_depth_manifest)"
fi

resolve_base_run_root() {
  if [[ -n "$BASE_RUN_ROOT" ]]; then
    return 0
  fi
  BASE_RUN_ROOT="$(find "$RUNS_DIR" -maxdepth 1 -type d -name 'train_all_4class_depth_robust_20*' -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "$BASE_RUN_ROOT" ]]; then
    echo "No depth-robust train_all run found under $RUNS_DIR"
    exit 1
  fi
}

mkdir -p "$LOG_DIR" "$RUNS_DIR"

start_session() {
  local timestamp output_root run_log status_file runner_script
  resolve_base_run_root
  timestamp="$(date +%Y%m%d_%H%M%S)"
  output_root="$RUNS_DIR/strong_followups_4class_${timestamp}"
  run_log="$LOG_DIR/strong_followups_4class_${timestamp}.log"
  status_file="$output_root/status.tsv"
  runner_script="$LOG_DIR/run_strong_followups_4class_${timestamp}.sh"
  mkdir -p "$output_root"

  cat > "$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(shell_quote "$WORKDIR")
ln -sfn $(shell_quote "$run_log") $(shell_quote "$LATEST_LOG_LINK")
trap 'status=\$?; echo "=== follow-up runner exited with code \$status at \$(date) ===" | tee -a $(shell_quote "$run_log"); ln -sfn $(shell_quote "$output_root") $(shell_quote "$LATEST_RUN_LINK")' EXIT
WORKDIR=$(shell_quote "$WORKDIR") CONDA_ENV=$(shell_quote "$CONDA_ENV") MANIFEST=$(shell_quote "$MANIFEST") \
BASE_RUN_ROOT=$(shell_quote "$BASE_RUN_ROOT") OUTPUT_ROOT=$(shell_quote "$output_root") RUN_LOG=$(shell_quote "$run_log") \
STATUS_FILE=$(shell_quote "$status_file") FEATURE_MODE=$(shell_quote "$FEATURE_MODE") XGB_THREADS=$(shell_quote "$XGB_THREADS") \
NEURAL_THREADS=$(shell_quote "$NEURAL_THREADS") BATCH_SIZE=$(shell_quote "$BATCH_SIZE") WAIT_SECONDS=$(shell_quote "$WAIT_SECONDS") \
WAIT_TIMEOUT_HOURS=$(shell_quote "$WAIT_TIMEOUT_HOURS") RESUME=$(shell_quote "$RESUME") \
bash $(shell_quote "$WORKDIR/scripts/strong_followups_4class_runner.sh")
EOF
  chmod +x "$runner_script"

  tmux new-session -d -s "$SESSION_NAME" "bash '$runner_script'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Base train_all: $BASE_RUN_ROOT"
  echo "Output: $output_root"
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
  attach) tmux attach -t "$SESSION_NAME" ;;
  status)
    tmux list-sessions 2>/dev/null | grep "^$SESSION_NAME:" || {
      echo "Session '$SESSION_NAME' is not running."
      exit 1
    }
    ;;
  stop)
    tmux kill-session -t "$SESSION_NAME" && echo "Stopped tmux session: $SESSION_NAME" || {
      echo "Session '$SESSION_NAME' is not running."
      exit 1
    }
    ;;
  logs)
    if [[ -f "$LATEST_LOG_LINK" || -L "$LATEST_LOG_LINK" ]]; then
      tail -n 160 "$LATEST_LOG_LINK"
    else
      echo "No latest log found at $LATEST_LOG_LINK"
      exit 1
    fi
    ;;
esac
