#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_trainml"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$WORKDIR/logs"
SAMPLE_MODE=0
RUN_ID=""
THRESHOLD=""
SAMPLE_VIDEOS=10
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
COMMAND="start"

usage() {
  cat <<'EOF'
Usage: scripts/tmux_trainml.sh [command] [options]

Commands:
  start       Start train_ml in tmux (default)
  attach      Attach to active tmux session
  status      Show tmux session status
  stop        Stop the tmux session
  logs        Show latest train_ml log (tail)

Options:
  --sample            Run train_ml in sample mode
  --sample-videos N   Number of videos when --sample is on (default: 10)
  --threshold T       Force inference threshold (example: 0.30)
  --session NAME      tmux session name (default: engagement_trainml)
  --env NAME          conda environment name (default: thesis)
  --run-id ID         Custom run id for checkpoint isolation (default: timestamp)
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
    --sample-videos)
      SAMPLE_VIDEOS="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --env)
      CONDA_ENV="$2"
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
LATEST_LOG_LINK="$LOG_DIR/latest_trainml.log"

build_command() {
  local sample_flag=""
  if [[ "$SAMPLE_MODE" -eq 1 ]]; then
    sample_flag=" --sample --sample-videos $SAMPLE_VIDEOS"
  fi

  local threshold_flag=""
  if [[ -n "$THRESHOLD" ]]; then
    threshold_flag=" --threshold $THRESHOLD"
  fi

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local active_run_id="$RUN_ID"
  if [[ -z "$active_run_id" ]]; then
    active_run_id="$timestamp"
  fi

  local run_checkpoint_dir="$RUNS_CHECKPOINT_DIR/trainml_$active_run_id"
  local run_checkpoint="$run_checkpoint_dir/engagement_xgb.json"

  local run_log
  run_log="$LOG_DIR/trainml_${timestamp}.log"

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== TrainML started at \\$(date) ===" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Run checkpoint: $run_checkpoint" | tee -a "$run_log"
mkdir -p "$run_checkpoint_dir"
conda run --no-capture-output -n "$CONDA_ENV" python train_ml.py$sample_flag$threshold_flag --output "$run_checkpoint" --run-id "$active_run_id" 2>&1 | tee -a "$run_log"
echo "=== TrainML finished at \\$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists. Use: scripts/tmux_trainml.sh attach"
      exit 1
    fi

    TRAIN_SCRIPT="$(build_command)"
    tmux new-session -d -s "$SESSION_NAME" "bash -lc '$TRAIN_SCRIPT'"
    echo "Started tmux session: $SESSION_NAME"
    echo "Attach: scripts/tmux_trainml.sh attach --session $SESSION_NAME"
    echo "Check status: scripts/tmux_trainml.sh status --session $SESSION_NAME"
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
