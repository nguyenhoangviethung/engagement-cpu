#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_ml_infer"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
COMMAND="start"
MODEL="$WORKDIR/checkpoints/engagement_xgb.json"
SEQUENCE=""
FEATURE_MODE="tsfresh"
THRESHOLD=""
SUMMARY_JSON=""
EXAMPLE_MODE=0

usage() {
  cat <<'EOF'
Usage: scripts/ml/tmux_infer.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --model PATH
  --sequence PATH        (required unless --example)
  --feature-mode basic|tsfresh
  --threshold V
  --summary-json PATH
  --session NAME
  --env NAME
  --example              Run tmux smoke test (python --help only)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --sequence) SEQUENCE="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --summary-json) SUMMARY_JSON="$2"; shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --example) EXAMPLE_MODE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_ml_infer.log"

build_command() {
  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local run_log="$LOG_DIR/ml_infer_${timestamp}.log"

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== ML infer example started at \$(date) ===" | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.infer --help 2>&1 | tee -a "$run_log"
echo "=== ML infer example finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
    return
  fi

  local threshold_flag=""
  [[ -n "$THRESHOLD" ]] && threshold_flag=" --threshold $THRESHOLD"
  local summary_flag=""
  [[ -n "$SUMMARY_JSON" ]] && summary_flag=" --summary-json $SUMMARY_JSON"

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== ML infer started at \$(date) ===" | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.infer \\
  --model "$MODEL" \\
  --sequence "$SEQUENCE" \\
  --feature-mode "$FEATURE_MODE"$threshold_flag$summary_flag 2>&1 | tee -a "$run_log"
echo "=== ML infer finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
}

case "$COMMAND" in
  start)
    if [[ "$EXAMPLE_MODE" -ne 1 && -z "$SEQUENCE" ]]; then
      echo "--sequence is required"
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
