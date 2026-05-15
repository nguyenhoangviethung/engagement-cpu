#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_extract"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$WORKDIR/logs"
LOG_EVERY=10
SAMPLE_MODE=0
RUN_ID=""
RUNS_PROCESSED_DIR="$WORKDIR/data/processed/runs"
COMMAND="start"

usage() {
  cat <<'EOF'
Usage: scripts/tmux_extract.sh [command] [options]

Commands:
  start       Start extract in tmux (default)
  attach      Attach to active tmux session
  status      Show tmux session status
  stop        Stop the tmux session
  logs        Show latest extract log (tail)

Options:
  --sample            Run extract in sample mode
  --session NAME      tmux session name (default: engagement_extract)
  --env NAME          conda environment name (default: thesis)
  --log-every N       Progress log frequency for extract (videos)
  --run-id ID         Custom run id for output isolation (default: timestamp)
  --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs)
      COMMAND="$1"
      shift
      ;;
    --sample)
      SAMPLE_MODE=1
      shift
      ;;
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --log-every)
      LOG_EVERY="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_extract.log"

build_command() {
  local sample_flag=""
  if [[ "$SAMPLE_MODE" -eq 1 ]]; then
    sample_flag=" --sample"
  fi

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local active_run_id="$RUN_ID"
  if [[ -z "$active_run_id" ]]; then
    active_run_id="$timestamp"
  fi

  local run_processed_dir="$RUNS_PROCESSED_DIR/extract_$active_run_id"
  local run_features_dir="$run_processed_dir/features"
  local run_manifest="$run_processed_dir/feature_manifest.csv"

  local run_log
  run_log="$LOG_DIR/extract_${timestamp}.log"

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== Extract started at \$(date) ===" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Run features dir: $run_features_dir" | tee -a "$run_log"
echo "Run manifest: $run_manifest" | tee -a "$run_log"
mkdir -p "$run_features_dir"
conda run --no-capture-output -n "$CONDA_ENV" python -u extract_features.py$sample_flag --features-dir "$run_features_dir" --manifest "$run_manifest" --log-every "$LOG_EVERY" 2>&1 | tee -a "$run_log"
echo "=== Extract finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists. Use: scripts/tmux_extract.sh attach"
      exit 1
    fi

    EXTRACT_SCRIPT="$(build_command)"
    tmux new-session -d -s "$SESSION_NAME" "bash -lc '$EXTRACT_SCRIPT'"
    echo "Started tmux session: $SESSION_NAME"
    echo "Attach: scripts/tmux_extract.sh attach --session $SESSION_NAME"
    echo "Check status: scripts/tmux_extract.sh status --session $SESSION_NAME"
    ;;

  attach)
    tmux attach -t "$SESSION_NAME"
    ;;

  status)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' is running."
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
