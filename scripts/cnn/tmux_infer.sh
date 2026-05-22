#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_cnn_infer"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
COMMAND="start"
CHECKPOINT="$WORKDIR/checkpoints/engagement_cnn.pt"
IMAGE=""
THRESHOLD=""
IMAGE_SIZE=""
EXAMPLE_MODE=0

usage() {
  cat <<'EOF'
Usage: scripts/cnn/tmux_infer.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --checkpoint PATH
  --image PATH           (required unless --example)
  --threshold V
  --image-size N
  --session NAME
  --env NAME
  --example              Run tmux smoke test (python --help only)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --example) EXAMPLE_MODE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_cnn_infer.log"

build_command() {
  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local run_log="$LOG_DIR/cnn_infer_${timestamp}.log"

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== CNN infer example started at \$(date) ===" | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.infer --help 2>&1 | tee -a "$run_log"
echo "=== CNN infer example finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
    return
  fi

  local threshold_flag=""
  [[ -n "$THRESHOLD" ]] && threshold_flag=" --threshold $THRESHOLD"
  local image_size_flag=""
  [[ -n "$IMAGE_SIZE" ]] && image_size_flag=" --image-size $IMAGE_SIZE"

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== CNN infer started at \$(date) ===" | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.infer \\
  --checkpoint "$CHECKPOINT" \\
  --image "$IMAGE"$threshold_flag$image_size_flag 2>&1 | tee -a "$run_log"
echo "=== CNN infer finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
}

case "$COMMAND" in
  start)
    if [[ "$EXAMPLE_MODE" -ne 1 && -z "$IMAGE" ]]; then
      echo "--image is required"
      exit 1
    fi
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
