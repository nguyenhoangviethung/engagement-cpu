#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_pipeline"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
SAMPLE_MODE=0
EXTRACT_LOG_EVERY=10
TRAIN_LOG_EVERY=1
RUN_ID=""
MODEL_NAME="gru"
CPU_THREADS=2
DEVICE="auto"
USE_AMP=0
RESUME_FROM=""
RESUME_LAST=0
RUNS_PROCESSED_DIR="$WORKDIR/data/processed/runs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
COMMAND="start"

usage() {
  cat <<'EOF'
Usage: scripts/rnn/tmux_pipeline.sh [command] [options]

Commands:
  start       Start pipeline in tmux (default)
  attach      Attach to active tmux session
  status      Show tmux session status
  stop        Stop the tmux session
  logs        Show latest pipeline log (tail)

Options:
  --sample            Run extract/train in sample mode
  --session NAME      tmux session name (default: engagement_pipeline)
  --env NAME          conda environment name (default: thesis)
  --extract-log-every N       Progress log frequency for extract_features.py (videos)
  --train-log-every N         Progress log frequency for train.py (epochs)
  --run-id ID         Custom run id for output isolation (default: timestamp)
  --model NAME        gru | gru_basic | simple_gru | tcn | 1dcnn | temporal_cnn | transformer | tiny_transformer (default: gru)
  --cpu-threads N     PyTorch CPU threads for training (default: 2)
  --device NAME       auto | cpu | cuda | cuda:0 (default: auto)
  --amp               Enable AMP mixed precision (CUDA only)
  --resume-from PATH  Resume train step from specific checkpoint
  --resume-last       Resume train step from <run_checkpoint>.last.pt
  --help              Show this help

Pipeline steps:
  1) preprocess_labels.py
  2) extract_features.py [--sample optional]
  3) train.py [--sample optional]
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
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --extract-log-every)
      EXTRACT_LOG_EVERY="$2"
      shift 2
      ;;
    --train-log-every)
      TRAIN_LOG_EVERY="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --cpu-threads)
      CPU_THREADS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --amp)
      USE_AMP=1
      shift
      ;;
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    --resume-last)
      RESUME_LAST=1
      shift
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
LATEST_LOG_LINK="$LOG_DIR/latest.log"

build_pipeline_command() {
  local sample_flag=""
  if [[ "$SAMPLE_MODE" -eq 1 ]]; then
    sample_flag=" --sample"
  fi
  local amp_flag=""
  if [[ "$USE_AMP" -eq 1 ]]; then
    amp_flag=" --amp"
  fi
  local resume_flag=""

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local active_run_id="$RUN_ID"
  if [[ -z "$active_run_id" ]]; then
    active_run_id="$timestamp"
  fi

  local run_processed_dir="$RUNS_PROCESSED_DIR/pipeline_$active_run_id"
  local run_features_dir="$run_processed_dir/features"
  local run_manifest="$run_processed_dir/feature_manifest.csv"
  local run_checkpoint_dir="$RUNS_CHECKPOINT_DIR/pipeline_$active_run_id"
  local run_checkpoint="$run_checkpoint_dir/engagement_gru.pt"
  local run_last_checkpoint="$run_checkpoint_dir/engagement_gru.last.pt"

  if [[ -n "$RESUME_FROM" ]]; then
    resume_flag=" --resume-from \"$RESUME_FROM\""
  elif [[ "$RESUME_LAST" -eq 1 && -f "$run_last_checkpoint" ]]; then
    resume_flag=" --resume-from \"$run_last_checkpoint\""
  fi

  local run_log
  run_log="$LOG_DIR/pipeline_${timestamp}.log"

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== Pipeline started at \$(date) ===" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Run features dir: $run_features_dir" | tee -a "$run_log"
echo "Run manifest: $run_manifest" | tee -a "$run_log"
echo "Run checkpoint: $run_checkpoint" | tee -a "$run_log"

mkdir -p "$run_features_dir"
mkdir -p "$run_checkpoint_dir"

echo "[1/3] preprocess_labels.py" | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.preprocess_labels 2>&1 | tee -a "$run_log"

echo "[2/3] extract_features.py$sample_flag" | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.rnn.extract_features$sample_flag --features-dir "$run_features_dir" --manifest "$run_manifest" --log-every "$EXTRACT_LOG_EVERY" 2>&1 | tee -a "$run_log"

echo "[3/3] train.py$sample_flag" | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.rnn.train$sample_flag$amp_flag --manifest "$run_manifest" --output "$run_checkpoint" --log-every "$TRAIN_LOG_EVERY" --model "$MODEL_NAME" --cpu-threads "$CPU_THREADS" --device "$DEVICE"$resume_flag 2>&1 | tee -a "$run_log"

echo "=== Pipeline finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists. Use: scripts/rnn/tmux_pipeline.sh attach"
      exit 1
    fi

    PIPELINE_SCRIPT="$(build_pipeline_command)"
    tmux new-session -d -s "$SESSION_NAME" "bash -lc '$PIPELINE_SCRIPT'"
    echo "Started tmux session: $SESSION_NAME"
    echo "Attach: scripts/rnn/tmux_pipeline.sh attach --session $SESSION_NAME"
    echo "Check status: scripts/rnn/tmux_pipeline.sh status --session $SESSION_NAME"
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
      tail -n 120 "$LATEST_LOG_LINK"
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
