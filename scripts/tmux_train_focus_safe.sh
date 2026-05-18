#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_train_focus"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$WORKDIR/logs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
DEFAULT_MANIFEST="$WORKDIR/data/processed/runs/pipeline_2/feature_manifest.csv"

LOG_EVERY=1
SAMPLE_MODE=0
RUN_ID=""
THRESHOLD_OBJECTIVE="balanced_accuracy"
MIN_RECALL_POS="0.75"
MANIFEST_PATH="$DEFAULT_MANIFEST"
COMMAND="start"

usage() {
  cat <<'EOF'
Usage: scripts/tmux_train_focus_safe.sh [command] [options]

Commands:
  start       Start train in tmux (default)
  attach      Attach to active tmux session
  status      Show tmux session status
  stop        Stop the tmux session
  logs        Show latest train log (tail)

Options:
  --manifest PATH      Feature manifest path
  --sample             Run sample training
  --session NAME       tmux session name (default: engagement_train_focus)
  --env NAME           conda environment name (default: thesis)
  --log-every N        Log frequency for train.py
  --run-id ID          Custom run id (default: timestamp)
  --objective NAME     f1_pos | focused_recall | balanced_accuracy (default: balanced_accuracy)
  --min-recall-pos V   Min recall for class 1 (used only for focused_recall)
  --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs)
      COMMAND="$1"
      shift
      ;;
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
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
    --objective)
      THRESHOLD_OBJECTIVE="$2"
      shift 2
      ;;
    --min-recall-pos)
      MIN_RECALL_POS="$2"
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
LATEST_LOG_LINK="$LOG_DIR/latest_train_focus.log"

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

  local run_checkpoint_dir="$RUNS_CHECKPOINT_DIR/train_$active_run_id"
  local run_checkpoint="$run_checkpoint_dir/engagement_gru.pt"
  local run_log="$LOG_DIR/train_focus_${active_run_id}_${timestamp}.log"

  if [[ -e "$run_checkpoint" ]]; then
    echo "Checkpoint exists: $run_checkpoint"
    echo "Please choose another --run-id to avoid overwrite."
    exit 1
  fi

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== Train started at \$(date) ===" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Manifest: $MANIFEST_PATH" | tee -a "$run_log"
echo "Checkpoint: $run_checkpoint" | tee -a "$run_log"
echo "Objective: $THRESHOLD_OBJECTIVE (min_recall_pos=$MIN_RECALL_POS)" | tee -a "$run_log"
mkdir -p "$run_checkpoint_dir"
conda run --no-capture-output -n "$CONDA_ENV" python -u train.py$sample_flag \\
  --manifest "$MANIFEST_PATH" \\
  --output "$run_checkpoint" \\
  --log-every "$LOG_EVERY" \\
  --threshold-objective "$THRESHOLD_OBJECTIVE" \\
  --min-recall-pos "$MIN_RECALL_POS" \\
  --run-id "$active_run_id" 2>&1 | tee -a "$run_log"
echo "=== Train finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists. Use attach/status/stop first."
      exit 1
    fi
    TRAIN_SCRIPT="$(build_command)"
    tmux new-session -d -s "$SESSION_NAME" "bash -lc '$TRAIN_SCRIPT'"
    echo "Started tmux session: $SESSION_NAME"
    echo "Attach: scripts/tmux_train_focus_safe.sh attach --session $SESSION_NAME"
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
