#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"
shift || true

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/.venv/bin/python}"
SESSION_NAME="mediapipe_prior_window"
RUN_ID="mediapipe_prior_window_$(date +%Y%m%d_%H%M%S)"
MANIFEST="$WORKDIR/data/processed/runs/mediapipe_product_features/mediapipe_feature_manifest.csv"
OUTPUT_DIR="$WORKDIR/checkpoints/runs/$RUN_ID"
WINDOW_SIZES="8 12 16"
STRIDE="4"
AGGREGATIONS="mean p75"
MIN_RECALL_NEGS="0.0 0.1 0.15 0.2 0.25 0.3 0.4"
TARGET_POS_PRIOR=""

usage() {
  cat <<USAGE
Usage:
  $0 start [options]
  $0 status [--session NAME]
  $0 logs [--session NAME]
  $0 attach [--session NAME]
  $0 stop [--session NAME]

Options:
  --session NAME
  --run-id ID
  --manifest PATH
  --output-dir PATH
  --window-sizes "8 12 16"
  --stride N
  --aggregations "mean p75"
  --min-recall-negs "0.0 0.1 0.2 0.3"
  --target-pos-prior FLOAT
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION_NAME="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; OUTPUT_DIR="$WORKDIR/checkpoints/runs/$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --window-sizes) WINDOW_SIZES="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --aggregations) AGGREGATIONS="$2"; shift 2 ;;
    --min-recall-negs) MIN_RECALL_NEGS="$2"; shift 2 ;;
    --target-pos-prior) TARGET_POS_PRIOR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

case "$ACTION" in
  status)
    tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux list-panes -t "$SESSION_NAME" -F '#S #{pane_pid} #{pane_current_command}' || true
    exit 0
    ;;
  logs)
    tmux capture-pane -t "$SESSION_NAME" -p -S -260
    exit 0
    ;;
  attach)
    tmux attach -t "$SESSION_NAME"
    exit 0
    ;;
  stop)
    tmux kill-session -t "$SESSION_NAME"
    exit 0
    ;;
  start) ;;
  *) echo "Unknown action: $ACTION" >&2; usage; exit 2 ;;
esac

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session already exists: $SESSION_NAME"
  exit 1
fi

prior_arg=""
if [[ -n "$TARGET_POS_PRIOR" ]]; then
  prior_arg="--target-pos-prior $TARGET_POS_PRIOR"
fi

mkdir -p "$OUTPUT_DIR"
cmd=$(cat <<EOF
set -euo pipefail
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src"
echo "[mediapipe-prior-window] run_id=$RUN_ID"
"$PYTHON_BIN" -u -m engagement_daisee.mediapipe.train_prior_window \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --window-sizes $WINDOW_SIZES \
  --stride "$STRIDE" \
  --aggregations $AGGREGATIONS \
  --min-recall-negs $MIN_RECALL_NEGS \
  $prior_arg 2>&1 | tee "$OUTPUT_DIR/train.log"
echo "[mediapipe-prior-window] done. leaderboard=$OUTPUT_DIR/leaderboard.json"
EOF
)

tmux new-session -d -s "$SESSION_NAME" "$cmd"
echo "Started tmux session: $SESSION_NAME"
echo "Run id: $RUN_ID"
echo "Output: $OUTPUT_DIR"
echo "Logs: $0 logs --session $SESSION_NAME"
