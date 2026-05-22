#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_cnn_pipeline"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
RUNS_PROCESSED_DIR="$WORKDIR/data/processed/runs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"

COMMAND="start"
RUN_ID=""
SAMPLE_MODE=0
FRAME_SIZE=112
FRAMES_PER_VIDEO=8
MODEL_NAME="mobilenet_v3_small"
BATCH_SIZE=64
EPOCHS=20
LEARNING_RATE=3e-4
WEIGHT_DECAY=1e-4
PATIENCE=6
USE_PRETRAINED=0
FREEZE_BACKBONE=0
TRAIN_SAMPLER="weighted"

usage() {
  cat <<'EOF'
Usage: scripts/tmux_cnn_pipeline.sh [command] [options]

Commands:
  start       Start CNN pipeline in tmux (default)
  attach      Attach to active tmux session
  status      Show tmux session status
  stop        Stop tmux session
  logs        Show latest CNN pipeline log (tail)

Options:
  --sample                 Run sample mode
  --run-id ID              Custom run id (default: timestamp)
  --session NAME           tmux session name
  --env NAME               conda env (default: thesis)
  --frame-size N           Frame size for extraction (default: 112)
  --frames-per-video N     Uniform frames per video (default: 8)
  --model NAME             mobilenet_v3_small | efficientnet_b0 | tinycnn
  --batch-size N           Train batch size (default: 64)
  --epochs N               Train epochs (default: 20)
  --lr V                   Learning rate (default: 3e-4)
  --weight-decay V         Weight decay (default: 1e-4)
  --patience N             Early stopping patience (default: 6)
  --pretrained             Use ImageNet weights if available
  --freeze-backbone        Freeze backbone and train head only
  --train-sampler NAME     weighted | shuffle
  --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs)
      COMMAND="$1"
      shift
      ;;
    --sample)
      SAMPLE_MODE=1
      shift
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --frame-size)
      FRAME_SIZE="$2"
      shift 2
      ;;
    --frames-per-video)
      FRAMES_PER_VIDEO="$2"
      shift 2
      ;;
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --weight-decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --pretrained)
      USE_PRETRAINED=1
      shift
      ;;
    --freeze-backbone)
      FREEZE_BACKBONE=1
      shift
      ;;
    --train-sampler)
      TRAIN_SAMPLER="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_cnn_pipeline.log"

build_command() {
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local active_run_id="$RUN_ID"
  if [[ -z "$active_run_id" ]]; then
    active_run_id="$timestamp"
  fi

  local sample_flag=""
  if [[ "$SAMPLE_MODE" -eq 1 ]]; then
    sample_flag=" --sample"
  fi

  local pretrained_flag=""
  if [[ "$USE_PRETRAINED" -eq 1 ]]; then
    pretrained_flag=" --pretrained"
  fi

  local freeze_flag=""
  if [[ "$FREEZE_BACKBONE" -eq 1 ]]; then
    freeze_flag=" --freeze-backbone"
  fi

  local run_processed_dir="$RUNS_PROCESSED_DIR/cnn_$active_run_id"
  local frame_dir="$run_processed_dir/frames"
  local manifest="$run_processed_dir/cnn_frame_manifest.csv"
  local run_checkpoint_dir="$RUNS_CHECKPOINT_DIR/cnn_$active_run_id"
  local checkpoint="$run_checkpoint_dir/engagement_cnn.pt"
  local run_log="$LOG_DIR/cnn_pipeline_${active_run_id}_${timestamp}.log"

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail
echo "=== CNN pipeline started at \$(date) ===" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Frames dir: $frame_dir" | tee -a "$run_log"
echo "Manifest: $manifest" | tee -a "$run_log"
echo "Checkpoint: $checkpoint" | tee -a "$run_log"
mkdir -p "$frame_dir" "$run_checkpoint_dir"

echo "[1/3] preprocess_labels.py" | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.preprocess_labels 2>&1 | tee -a "$run_log"

echo "[2/3] cnn_extract_frames.py$sample_flag" | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.cnn.extract_frames$sample_flag \\
  --output-dir "$frame_dir" \\
  --manifest "$manifest" \\
  --frame-size "$FRAME_SIZE" \\
  --frames-per-video "$FRAMES_PER_VIDEO" 2>&1 | tee -a "$run_log"

echo "[3/3] train_cnn.py$sample_flag" | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.cnn.train$sample_flag \\
  --manifest "$manifest" \\
  --output "$checkpoint" \\
  --model "$MODEL_NAME" \\
  --batch-size "$BATCH_SIZE" \\
  --epochs "$EPOCHS" \\
  --lr "$LEARNING_RATE" \\
  --weight-decay "$WEIGHT_DECAY" \\
  --patience "$PATIENCE" \\
  --train-sampler "$TRAIN_SAMPLER"$pretrained_flag$freeze_flag 2>&1 | tee -a "$run_log"

echo "=== CNN pipeline finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists."
      exit 1
    fi
    RUN_SCRIPT="$(build_command)"
    tmux new-session -d -s "$SESSION_NAME" "bash -lc '$RUN_SCRIPT'"
    echo "Started tmux session: $SESSION_NAME"
    echo "Attach: scripts/tmux_cnn_pipeline.sh attach --session $SESSION_NAME"
    ;;
  attach)
    tmux attach -t "$SESSION_NAME"
    ;;
  status)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' is running."
      tmux list-sessions | grep "^$SESSION_NAME:"
    else
      echo "Session '$SESSION_NAME' is not running."
      exit 1
    fi
    ;;
  stop)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "$SESSION_NAME"
      echo "Stopped tmux session: $SESSION_NAME"
    else
      echo "Session '$SESSION_NAME' is not running."
      exit 1
    fi
    ;;
  logs)
    if [[ -L "$LATEST_LOG_LINK" || -f "$LATEST_LOG_LINK" ]]; then
      tail -n 160 "$LATEST_LOG_LINK"
    else
      echo "No latest log found at $LATEST_LOG_LINK"
      exit 1
    fi
    ;;
  *)
    usage
    exit 1
    ;;
esac

