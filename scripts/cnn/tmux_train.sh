#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_cnn_train"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"

COMMAND="start"
RUN_ID=""
MANIFEST="$WORKDIR/data/processed/cnn_frame_manifest.csv"
SAMPLE_MODE=0
MODEL_NAME="mobilenet_v3_small"
BATCH_SIZE=64
EPOCHS=20
LEARNING_RATE=3e-4
WEIGHT_DECAY=1e-4
PATIENCE=6
IMAGE_SIZE=112
USE_PRETRAINED=0
FREEZE_BACKBONE=0
TRAIN_SAMPLER="weighted"
EXAMPLE_MODE=0

usage() {
  cat <<'EOF'
Usage: scripts/cnn/tmux_train.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --manifest PATH
  --run-id ID
  --sample
  --model NAME             mobilenet_v3_small | efficientnet_b0 | tinycnn
  --batch-size N
  --epochs N
  --lr V
  --weight-decay V
  --patience N
  --image-size N
  --pretrained
  --freeze-backbone
  --train-sampler NAME     weighted | shuffle
  --session NAME
  --env NAME
  --example                Run tmux smoke test (python --help only)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --sample) SAMPLE_MODE=1; shift ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LEARNING_RATE="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --pretrained) USE_PRETRAINED=1; shift ;;
    --freeze-backbone) FREEZE_BACKBONE=1; shift ;;
    --train-sampler) TRAIN_SAMPLER="$2"; shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --example) EXAMPLE_MODE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_cnn_train.log"

build_command() {
  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local active_run_id="$RUN_ID"
  if [[ -z "$active_run_id" ]]; then
    active_run_id="$timestamp"
  fi

  local run_checkpoint_dir="$RUNS_CHECKPOINT_DIR/cnn_train_$active_run_id"
  local run_checkpoint="$run_checkpoint_dir/engagement_cnn.pt"
  local run_log="$LOG_DIR/cnn_train_${active_run_id}_${timestamp}.log"

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== CNN train example started at \$(date) ===" | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.train --help 2>&1 | tee -a "$run_log"
echo "=== CNN train example finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
    return
  fi

  local sample_flag=""
  [[ "$SAMPLE_MODE" -eq 1 ]] && sample_flag=" --sample"

  local pretrained_flag=""
  [[ "$USE_PRETRAINED" -eq 1 ]] && pretrained_flag=" --pretrained"

  local freeze_flag=""
  [[ "$FREEZE_BACKBONE" -eq 1 ]] && freeze_flag=" --freeze-backbone"

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== CNN train started at \$(date) ===" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Checkpoint: $run_checkpoint" | tee -a "$run_log"
mkdir -p "$run_checkpoint_dir"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.cnn.train$sample_flag \\
  --manifest "$MANIFEST" \\
  --output "$run_checkpoint" \\
  --model "$MODEL_NAME" \\
  --image-size "$IMAGE_SIZE" \\
  --batch-size "$BATCH_SIZE" \\
  --epochs "$EPOCHS" \\
  --lr "$LEARNING_RATE" \\
  --weight-decay "$WEIGHT_DECAY" \\
  --patience "$PATIENCE" \\
  --train-sampler "$TRAIN_SAMPLER" \\
  --run-id "$active_run_id"$pretrained_flag$freeze_flag 2>&1 | tee -a "$run_log"
echo "=== CNN train finished at \$(date) ===" | tee -a "$run_log"
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
