#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"
shift || true

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/.venv/bin/python}"
SESSION_NAME="mediapipe_product_models"
RUN_ID="mediapipe_product_$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="$WORKDIR/data/processed/runs/$RUN_ID"
CHECKPOINT_ROOT="$WORKDIR/checkpoints/runs/$RUN_ID"
FRAMES_PER_VIDEO="30"
SAMPLE_VIDEOS="0"
DEVICE="auto"
TCN_EPOCHS="40"
OBJECTIVE="balanced_accuracy"

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
  --frames-per-video N      default: 30
  --sample-videos N         default: 0 means full dataset
  --device auto|cpu|cuda    default: auto
  --tcn-epochs N            default: 40
  --objective NAME          balanced_accuracy|f1_macro|recall_neg
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION_NAME="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; OUTPUT_ROOT="$WORKDIR/data/processed/runs/$2"; CHECKPOINT_ROOT="$WORKDIR/checkpoints/runs/$2"; shift 2 ;;
    --frames-per-video) FRAMES_PER_VIDEO="$2"; shift 2 ;;
    --sample-videos) SAMPLE_VIDEOS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --tcn-epochs) TCN_EPOCHS="$2"; shift 2 ;;
    --objective) OBJECTIVE="$2"; shift 2 ;;
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

mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_ROOT"
cmd=$(cat <<EOF
set -euo pipefail
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src"
echo "[mediapipe-product] run_id=$RUN_ID"
echo "[mediapipe-product] extracting features -> $OUTPUT_ROOT"
"$PYTHON_BIN" -u -m engagement_daisee.mediapipe.extract_features \
  --output-root "$OUTPUT_ROOT" \
  --frames-per-video "$FRAMES_PER_VIDEO" \
  --sample-videos "$SAMPLE_VIDEOS" \
  --log-every 100 2>&1 | tee "$CHECKPOINT_ROOT/extract.log"
echo "[mediapipe-product] training product models -> $CHECKPOINT_ROOT"
"$PYTHON_BIN" -u -m engagement_daisee.mediapipe.train_product_models \
  --manifest "$OUTPUT_ROOT/mediapipe_feature_manifest.csv" \
  --output-dir "$CHECKPOINT_ROOT" \
  --seq-len "$FRAMES_PER_VIDEO" \
  --objective "$OBJECTIVE" \
  --device "$DEVICE" \
  --tcn-epochs "$TCN_EPOCHS" 2>&1 | tee "$CHECKPOINT_ROOT/train.log"
echo "[mediapipe-product] done. leaderboard=$CHECKPOINT_ROOT/leaderboard.json"
EOF
)

tmux new-session -d -s "$SESSION_NAME" "$cmd"
echo "Started tmux session: $SESSION_NAME"
echo "Run id: $RUN_ID"
echo "Data output: $OUTPUT_ROOT"
echo "Checkpoints: $CHECKPOINT_ROOT"
echo "Logs: $0 logs --session $SESSION_NAME"
