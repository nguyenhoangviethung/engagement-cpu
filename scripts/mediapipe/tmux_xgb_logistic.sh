#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"
shift || true

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/.venv/bin/python}"
SESSION_NAME="mediapipe_xgb_logistic"
RUN_ID="mediapipe_xgb_logistic_$(date +%Y%m%d_%H%M%S)"
MANIFEST="$WORKDIR/data/processed/runs/mediapipe_product_features/mediapipe_feature_manifest.csv"
OUTPUT_DIR="$WORKDIR/checkpoints/runs/$RUN_ID"
SEQ_LEN="30"
OBJECTIVE="balanced_accuracy"
VARIANTS="all no_absolute_position behavior_plus_size eyes_head_mouth"

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
  --seq-len N
  --objective balanced_accuracy|f1_macro|recall_neg
  --variants "all no_absolute_position behavior_plus_size eyes_head_mouth"
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION_NAME="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; OUTPUT_DIR="$WORKDIR/checkpoints/runs/$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --objective) OBJECTIVE="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
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
    tmux capture-pane -t "$SESSION_NAME" -p -S -240
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

mkdir -p "$OUTPUT_DIR"
cmd=$(cat <<EOF
set -euo pipefail
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src"
echo "[mediapipe-xgb-logistic] run_id=$RUN_ID"
echo "[mediapipe-xgb-logistic] manifest=$MANIFEST"
"$PYTHON_BIN" -u -m engagement_daisee.mediapipe.train_xgb_logistic \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --seq-len "$SEQ_LEN" \
  --objective "$OBJECTIVE" \
  --variants $VARIANTS 2>&1 | tee "$OUTPUT_DIR/train.log"
echo "[mediapipe-xgb-logistic] done. leaderboard=$OUTPUT_DIR/leaderboard.json"
EOF
)

tmux new-session -d -s "$SESSION_NAME" "$cmd"
echo "Started tmux session: $SESSION_NAME"
echo "Run id: $RUN_ID"
echo "Output: $OUTPUT_DIR"
echo "Logs: $0 logs --session $SESSION_NAME"
