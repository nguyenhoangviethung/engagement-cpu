#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="paper_cnn_reproduction"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/.venv/bin/python}"
MANIFEST="$WORKDIR/data/processed/runs/openface709_binary_pca_features/openface709/feature_manifest.csv"
LABELS="$WORKDIR/data/processed/engagement_only_labels.csv"
OUTPUT_DIR="$WORKDIR/checkpoints/runs/paper_cnn_reproduction_$(date +%Y%m%d_%H%M%S)"
DEVICE="auto"
REPEATS="10"
MODE="grid"
ACTION="${1:-start}"
shift || true

usage() {
  cat <<USAGE
Usage:
  $0 start [options]
  $0 status [--session NAME]
  $0 logs [--session NAME]
  $0 attach [--session NAME]
  $0 stop [--session NAME]

Options for start:
  --session NAME       tmux session name (default: $SESSION_NAME)
  --manifest PATH      OpenFace709 manifest
  --labels PATH        engagement_only_labels.csv with engagement_raw 0..3
  --output-dir PATH    output directory
  --device NAME        auto|cpu|cuda (default: auto)
  --repeats N          paper used 10 iterations per model (default: 10)
  --best-only          train only paper best PCA/SVD configs
  --grid               train full paper grid for PCA and SVD (default)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION_NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --best-only) MODE="best"; shift ;;
    --grid) MODE="grid"; shift ;;
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
    tmux capture-pane -t "$SESSION_NAME" -p -S -200
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
  *)
    echo "Unknown action: $ACTION" >&2
    usage
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_DIR"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session already exists: $SESSION_NAME"
  exit 1
fi

best_flag=""
if [[ "$MODE" == "best" ]]; then
  best_flag=" --best-only"
fi

cmd=$(cat <<EOF
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src"
echo "[paper-cnn] output=$OUTPUT_DIR"
"$PYTHON_BIN" -u -m engagement_daisee.ml.reproduce_paper_cnn \
  --manifest "$MANIFEST" \
  --labels "$LABELS" \
  --output-dir "$OUTPUT_DIR" \
  --features pca svd \
  --paper-selection \
  --repeats "$REPEATS" \
  --device "$DEVICE"$best_flag 2>&1 | tee "$OUTPUT_DIR/run.log"
EOF
)

tmux new-session -d -s "$SESSION_NAME" "$cmd"
echo "Started tmux session: $SESSION_NAME"
echo "Output: $OUTPUT_DIR"
echo "Attach: tmux attach -t $SESSION_NAME"
