#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_paper_baseline"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"

COMMAND="start"
RUN_ID="paper_baseline"
MANIFEST="$WORKDIR/data/processed/runs/phase34_sequence_features/feature_manifest.csv"
SPLIT_MODE="video_80_20"
FEATURE_MODE="copur"
BACKEND="xgboost"
THRESHOLD_OBJECTIVE="accuracy"
DIM_REDUCTION="svd"
DIM_COMPONENTS=300
OVERSAMPLE="smote_lite"
CPU_WORKERS=2

usage() {
  cat <<'EOF'
Usage: scripts/ml/tmux_paper_baseline.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --manifest PATH
  --run-id ID
  --split-mode video_80_20|official
  --feature-mode basic|tsfresh|copur
  --backend auto|xgboost|lightgbm
  --threshold-objective accuracy|balanced_accuracy|f1_pos|f2_pos
  --dim-reduction none|pca|svd
  --dim-components N
  --oversample none|random|smote_lite
  --cpu-workers N
  --session NAME
  --env NAME
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --split-mode) SPLIT_MODE="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --threshold-objective) THRESHOLD_OBJECTIVE="$2"; shift 2 ;;
    --dim-reduction) DIM_REDUCTION="$2"; shift 2 ;;
    --dim-components) DIM_COMPONENTS="$2"; shift 2 ;;
    --oversample) OVERSAMPLE="$2"; shift 2 ;;
    --cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_paper_baseline.log"

build_command() {
  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local run_dir="$RUNS_CHECKPOINT_DIR/${RUN_ID}_${timestamp}"
  local run_log="$LOG_DIR/${RUN_ID}_${timestamp}.log"

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

mkdir -p "$run_dir"
echo "=== Paper baseline started at \$(date) ===" | tee -a "$run_log"
echo "Run ID: $RUN_ID" | tee -a "$run_log"
echo "Manifest: $MANIFEST" | tee -a "$run_log"
echo "Split mode: $SPLIT_MODE" | tee -a "$run_log"
echo "Feature mode: $FEATURE_MODE" | tee -a "$run_log"
echo "Output dir: $run_dir" | tee -a "$run_log"

"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.paper_baseline \
  --manifest "$MANIFEST" \
  --output-dir "$run_dir" \
  --split-mode "$SPLIT_MODE" \
  --feature-mode "$FEATURE_MODE" \
  --backend "$BACKEND" \
  --threshold-objective "$THRESHOLD_OBJECTIVE" \
  --dim-reduction "$DIM_REDUCTION" \
  --dim-components "$DIM_COMPONENTS" \
  --oversample "$OVERSAMPLE" \
  --cpu-workers "$CPU_WORKERS" 2>&1 | tee -a "$run_log"

echo "=== Paper baseline finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then echo "Session '$SESSION_NAME' already exists."; exit 1; fi
    SCRIPT="$(build_command)"
    tmux new-session -d -s "$SESSION_NAME" "bash -lc '$SCRIPT'"
    echo "Started tmux session: $SESSION_NAME"
    echo "Attach: tmux attach -t $SESSION_NAME"
    ;;
  attach) tmux attach -t "$SESSION_NAME" ;;
  status) tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux list-sessions | grep "^$SESSION_NAME:" || { echo "Session '$SESSION_NAME' is not running."; exit 1; } ;;
  stop) tmux kill-session -t "$SESSION_NAME" 2>/dev/null && echo "Stopped tmux session: $SESSION_NAME" || { echo "Session '$SESSION_NAME' is not running."; exit 1; } ;;
  logs) [[ -L "$LATEST_LOG_LINK" || -f "$LATEST_LOG_LINK" ]] && tail -n 180 "$LATEST_LOG_LINK" || { echo "No latest log found"; exit 1; } ;;
  *) usage; exit 1 ;;
esac
