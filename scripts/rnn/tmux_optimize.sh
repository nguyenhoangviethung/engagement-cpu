#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_optimize"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
COMMAND="start"
CHECKPOINT="$WORKDIR/checkpoints/engagement_gru.pt"
OUTPUT_DIR="$WORKDIR/checkpoints/deploy"
CPU_THREADS=2
BENCHMARK_ITERS=200
EXAMPLE_MODE=0

usage() {
  cat <<'EOF'
Usage: scripts/rnn/tmux_optimize.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --checkpoint PATH
  --output-dir PATH
  --cpu-threads N
  --benchmark-iters N
  --session NAME
  --env NAME
  --example           Run tmux smoke test (python --help only)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --cpu-threads) CPU_THREADS="$2"; shift 2 ;;
    --benchmark-iters) BENCHMARK_ITERS="$2"; shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --example) EXAMPLE_MODE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_optimize.log"

build_command() {
  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local run_log="$LOG_DIR/optimize_${timestamp}.log"

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== Optimize example started at \$(date) ===" | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.optimize_inference --help 2>&1 | tee -a "$run_log"
echo "=== Optimize example finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
    return
  fi

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== Optimize started at \$(date) ===" | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.optimize_inference \\
  --checkpoint "$CHECKPOINT" \\
  --output-dir "$OUTPUT_DIR" \\
  --cpu-threads "$CPU_THREADS" \\
  --benchmark-iters "$BENCHMARK_ITERS" 2>&1 | tee -a "$run_log"
echo "=== Optimize finished at \$(date) ===" | tee -a "$run_log"
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
