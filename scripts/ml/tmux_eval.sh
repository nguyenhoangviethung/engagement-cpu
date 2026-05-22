#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_ml_eval"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
COMMAND="start"
MANIFEST="$WORKDIR/data/processed/feature_manifest.csv"
MODEL="$WORKDIR/checkpoints/engagement_xgb.json"
SPLIT="test"
FEATURE_MODE="tsfresh"
THRESHOLD=""
SUMMARY_JSON=""
OUTPUT_JSON=""
EXAMPLE_MODE=0

usage() {
  cat <<'EOF'
Usage: scripts/ml/tmux_eval.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --manifest PATH
  --model PATH
  --split train|validation|test
  --feature-mode basic|tsfresh
  --threshold V
  --summary-json PATH
  --output-json PATH
  --session NAME
  --env NAME
  --example                Run tmux smoke test (python --help only)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --summary-json) SUMMARY_JSON="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --example) EXAMPLE_MODE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_ml_eval.log"

build_command() {
  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local run_log="$LOG_DIR/ml_eval_${timestamp}.log"

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== ML eval example started at \$(date) ===" | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.evaluate --help 2>&1 | tee -a "$run_log"
echo "=== ML eval example finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
    return
  fi

  local threshold_flag=""
  [[ -n "$THRESHOLD" ]] && threshold_flag=" --threshold $THRESHOLD"
  local summary_flag=""
  [[ -n "$SUMMARY_JSON" ]] && summary_flag=" --summary-json $SUMMARY_JSON"
  local output_flag=""
  [[ -n "$OUTPUT_JSON" ]] && output_flag=" --output-json $OUTPUT_JSON"

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== ML eval started at \$(date) ===" | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.evaluate \\
  --manifest "$MANIFEST" \\
  --model "$MODEL" \\
  --split "$SPLIT" \\
  --feature-mode "$FEATURE_MODE"$threshold_flag$summary_flag$output_flag 2>&1 | tee -a "$run_log"
echo "=== ML eval finished at \$(date) ===" | tee -a "$run_log"
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
