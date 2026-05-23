#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_ml_train"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
SAMPLE_MODE=0
RUN_ID=""
THRESHOLD=""
SAMPLE_VIDEOS=10
MANIFEST="$WORKDIR/data/processed/feature_manifest.csv"
BACKEND="auto"
FEATURE_MODE="tsfresh"
CPU_WORKERS=2
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
COMMAND="start"
EXAMPLE_MODE=0

usage() {
  cat <<'EOF'
Usage: scripts/ml/tmux_train.sh [command] [options]

Commands:
  start|attach|status|stop|logs

Options:
  --manifest PATH
  --sample
  --sample-videos N
  --threshold T
  --backend auto|xgboost|lightgbm
  --feature-mode basic|tsfresh
  --cpu-workers N
  --session NAME
  --env NAME
  --run-id ID
  --example           Run tmux smoke test (python --help only)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --sample) SAMPLE_MODE=1; shift ;;
    --sample-videos) SAMPLE_VIDEOS="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --example) EXAMPLE_MODE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_ml_train.log"

build_command() {
  local sample_flag=""
  if [[ "$SAMPLE_MODE" -eq 1 ]]; then
    sample_flag=" --sample --sample-videos $SAMPLE_VIDEOS"
  fi

  local threshold_flag=""
  [[ -n "$THRESHOLD" ]] && threshold_flag=" --threshold $THRESHOLD"

  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local active_run_id="$RUN_ID"
  if [[ -z "$active_run_id" ]]; then
    active_run_id="$timestamp"
  fi

  local run_checkpoint_dir="$RUNS_CHECKPOINT_DIR/ml_train_$active_run_id"
  local run_checkpoint="$run_checkpoint_dir/engagement_xgb.json"
  local run_log="$LOG_DIR/ml_train_${active_run_id}_${timestamp}.log"

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== ML train example started at \$(date) ===" | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.train --help 2>&1 | tee -a "$run_log"
echo "=== ML train example finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
    return
  fi

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== ML train started at \$(date) ===" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Run checkpoint: $run_checkpoint" | tee -a "$run_log"
mkdir -p "$run_checkpoint_dir"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.train$sample_flag$threshold_flag \\
  --manifest "$MANIFEST" \\
  --output "$run_checkpoint" \\
  --run-id "$active_run_id" \\
  --backend "$BACKEND" \\
  --feature-mode "$FEATURE_MODE" \\
  --cpu-workers "$CPU_WORKERS" 2>&1 | tee -a "$run_log"
echo "=== ML train finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then echo "Session '$SESSION_NAME' already exists."; exit 1; fi
    SCRIPT="$(build_command)"
    tmux new-session -d -s "$SESSION_NAME" "bash -lc '$SCRIPT'"
    echo "Started tmux session: $SESSION_NAME"
    ;;
  attach) tmux attach -t "$SESSION_NAME" ;;
  status) tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux list-sessions | grep "^$SESSION_NAME:" || { echo "Session '$SESSION_NAME' is not running."; exit 1; } ;;
  stop) tmux kill-session -t "$SESSION_NAME" 2>/dev/null && echo "Stopped tmux session: $SESSION_NAME" || { echo "Session '$SESSION_NAME' is not running."; exit 1; } ;;
  logs) [[ -L "$LATEST_LOG_LINK" || -f "$LATEST_LOG_LINK" ]] && tail -n 120 "$LATEST_LOG_LINK" || { echo "No latest log found"; exit 1; } ;;
  *) usage; exit 1 ;;
esac
