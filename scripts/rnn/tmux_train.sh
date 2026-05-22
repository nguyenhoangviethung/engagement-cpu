#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_train"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
MANIFEST="$WORKDIR/data/processed/feature_manifest.csv"
LOG_EVERY=1
SAMPLE_MODE=0
RUN_ID=""
MODEL_NAME="gru"
CPU_THREADS=2
DEVICE="auto"
USE_AMP=0
RESUME_FROM=""
RESUME_LAST=0
EXAMPLE_MODE=0
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
COMMAND="start"

usage() {
  cat <<'EOF'
Usage: scripts/rnn/tmux_train.sh [command] [options]

Commands:
  start       Start train in tmux (default)
  attach      Attach to active tmux session
  status      Show tmux session status
  stop        Stop the tmux session
  logs        Show latest train log (tail)

Options:
  --sample            Run train in sample mode
  --manifest PATH     Feature manifest CSV (default: data/processed/feature_manifest.csv)
  --session NAME      tmux session name (default: engagement_train)
  --env NAME          conda environment name (default: thesis)
  --log-every N       Progress log frequency for train (epochs)
  --run-id ID         Custom run id for checkpoint isolation (default: timestamp)
  --model NAME        gru | gru_basic | simple_gru | tcn | 1dcnn | temporal_cnn | transformer | tiny_transformer (default: gru)
  --cpu-threads N     PyTorch CPU threads (default: 2)
  --device NAME       auto | cpu | cuda | cuda:0 (default: auto)
  --amp               Enable AMP mixed precision (CUDA only)
  --resume-from PATH  Resume from a specific checkpoint
  --resume-last       Resume from <run_checkpoint>.last.pt (use with same --run-id)
  --example           Run tmux smoke test (python --help only)
  --help              Show this help
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
    --manifest)
      MANIFEST="$2"
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
    --log-every)
      LOG_EVERY="$2"
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
    --example)
      EXAMPLE_MODE=1
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
LATEST_LOG_LINK="$LOG_DIR/latest_train.log"

build_command() {
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

  local run_checkpoint_dir="$RUNS_CHECKPOINT_DIR/train_$active_run_id"
  local run_checkpoint="$run_checkpoint_dir/engagement_gru.pt"
  local run_last_checkpoint="$run_checkpoint_dir/engagement_gru.last.pt"

  if [[ -n "$RESUME_FROM" ]]; then
    resume_flag=" --resume-from \"$RESUME_FROM\""
  elif [[ "$RESUME_LAST" -eq 1 && -f "$run_last_checkpoint" ]]; then
    resume_flag=" --resume-from \"$run_last_checkpoint\""
  fi

  local run_log="$LOG_DIR/train_${timestamp}.log"

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== Train example started at \$(date) ===" | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.train --help 2>&1 | tee -a "$run_log"
echo "=== Train example finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
    return
  fi

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== Train started at \$(date) ===" | tee -a "$run_log"
echo "Manifest: $MANIFEST" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Run checkpoint: $run_checkpoint" | tee -a "$run_log"
mkdir -p "$run_checkpoint_dir"
conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.train$sample_flag$amp_flag --manifest "$MANIFEST" --log-every "$LOG_EVERY" --output "$run_checkpoint" --run-id "$active_run_id" --model "$MODEL_NAME" --cpu-threads "$CPU_THREADS" --device "$DEVICE"$resume_flag 2>&1 | tee -a "$run_log"
echo "=== Train finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists. Use: scripts/rnn/tmux_train.sh attach"
      exit 1
    fi

    TRAIN_SCRIPT="$(build_command)"
    tmux new-session -d -s "$SESSION_NAME" "bash -lc '$TRAIN_SCRIPT'"
    echo "Started tmux session: $SESSION_NAME"
    echo "Attach: scripts/rnn/tmux_train.sh attach --session $SESSION_NAME"
    echo "Check status: scripts/rnn/tmux_train.sh status --session $SESSION_NAME"
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
