#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_train_all"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
REPORTS_DIR="$WORKDIR/checkpoints/reports"

COMMAND="start"
RUN_ID_PREFIX="p2"
RNN_MANIFEST="$WORKDIR/data/processed/runs/pipeline_2/feature_manifest.csv"
ML_MANIFEST="$WORKDIR/data/processed/runs/pipeline_2/feature_manifest.csv"
CNN_MANIFEST="$WORKDIR/data/processed/cnn_frame_manifest.csv"

RNN_MODELS="gru gru_basic tcn tiny_transformer"
INCLUDE_ML=1
INCLUDE_CNN=0

SAMPLE_MODE=0
DEVICE="cuda"
USE_AMP=1
RNN_CPU_THREADS=2
RNN_HIDDEN_SIZE=192
RNN_NUM_LAYERS=3
RNN_DROPOUT=0.25
RNN_BATCH_SIZE=128
RNN_EPOCHS=40
RNN_PATIENCE=10
RNN_MIN_EPOCHS=12
RNN_TCN_BLOCKS=4
RNN_TCN_KERNEL_SIZE=5
RNN_THRESHOLD_OBJECTIVE="balanced_accuracy"
RNN_LOSS="bce_weighted"
ML_CPU_WORKERS=2
CNN_MODEL="mobilenet_v3_small"
CNN_BATCH_SIZE=64
CNN_EPOCHS=20
CNN_IMAGE_SIZE=112

EXAMPLE_MODE=0

usage() {
  cat <<'USAGE'
Usage: scripts/tmux_train_all.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --run-id-prefix ID
  --rnn-manifest PATH
  --ml-manifest PATH
  --cnn-manifest PATH
  --rnn-models "gru gru_basic tcn tiny_transformer"
  --no-ml
  --with-cnn
  --sample
  --device NAME
  --no-amp
  --rnn-cpu-threads N
  --rnn-hidden-size N
  --rnn-num-layers N
  --rnn-dropout V
  --rnn-batch-size N
  --rnn-epochs N
  --rnn-patience N
  --rnn-min-epochs N
  --rnn-tcn-blocks N
  --rnn-tcn-kernel-size N
  --rnn-threshold-objective NAME
  --rnn-loss NAME
  --ml-cpu-workers N
  --cnn-model NAME
  --cnn-batch-size N
  --cnn-epochs N
  --cnn-image-size N
  --session NAME
  --env NAME
  --example            Help-only smoke run in tmux
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --run-id-prefix) RUN_ID_PREFIX="$2"; shift 2 ;;
    --rnn-manifest) RNN_MANIFEST="$2"; shift 2 ;;
    --ml-manifest) ML_MANIFEST="$2"; shift 2 ;;
    --cnn-manifest) CNN_MANIFEST="$2"; shift 2 ;;
    --rnn-models) RNN_MODELS="$2"; shift 2 ;;
    --no-ml) INCLUDE_ML=0; shift ;;
    --with-cnn) INCLUDE_CNN=1; shift ;;
    --sample) SAMPLE_MODE=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --no-amp) USE_AMP=0; shift ;;
    --rnn-cpu-threads) RNN_CPU_THREADS="$2"; shift 2 ;;
    --rnn-hidden-size) RNN_HIDDEN_SIZE="$2"; shift 2 ;;
    --rnn-num-layers) RNN_NUM_LAYERS="$2"; shift 2 ;;
    --rnn-dropout) RNN_DROPOUT="$2"; shift 2 ;;
    --rnn-batch-size) RNN_BATCH_SIZE="$2"; shift 2 ;;
    --rnn-epochs) RNN_EPOCHS="$2"; shift 2 ;;
    --rnn-patience) RNN_PATIENCE="$2"; shift 2 ;;
    --rnn-min-epochs) RNN_MIN_EPOCHS="$2"; shift 2 ;;
    --rnn-tcn-blocks) RNN_TCN_BLOCKS="$2"; shift 2 ;;
    --rnn-tcn-kernel-size) RNN_TCN_KERNEL_SIZE="$2"; shift 2 ;;
    --rnn-threshold-objective) RNN_THRESHOLD_OBJECTIVE="$2"; shift 2 ;;
    --rnn-loss) RNN_LOSS="$2"; shift 2 ;;
    --ml-cpu-workers) ML_CPU_WORKERS="$2"; shift 2 ;;
    --cnn-model) CNN_MODEL="$2"; shift 2 ;;
    --cnn-batch-size) CNN_BATCH_SIZE="$2"; shift 2 ;;
    --cnn-epochs) CNN_EPOCHS="$2"; shift 2 ;;
    --cnn-image-size) CNN_IMAGE_SIZE="$2"; shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --example) EXAMPLE_MODE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$REPORTS_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_train_all.log"

start_session() {
  local timestamp run_root run_log summary_json history_jsonl cmd
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_root="$RUNS_CHECKPOINT_DIR/train_all_${RUN_ID_PREFIX}_${timestamp}"
  run_log="$LOG_DIR/train_all_${RUN_ID_PREFIX}_${timestamp}.log"
  summary_json="$run_root/train_all_summary.json"
  history_jsonl="$REPORTS_DIR/train_all_history.jsonl"

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cmd="cd '$WORKDIR' && set -euo pipefail && echo '=== train_all example started at '$(date) | tee -a '$run_log' && ./scripts/rnn/tmux_train_eval.sh --help 2>&1 | tee -a '$run_log' && ./scripts/ml/tmux_train_eval.sh --help 2>&1 | tee -a '$run_log' && ./scripts/cnn/tmux_train_eval.sh --help 2>&1 | tee -a '$run_log' && echo '=== train_all example finished at '$(date) | tee -a '$run_log' && ln -sfn '$run_log' '$LATEST_LOG_LINK'"
    tmux new-session -d -s "$SESSION_NAME" "bash -lc \"$cmd\""
    return
  fi

  cmd="cd '$WORKDIR' && \
WORKDIR='$WORKDIR' CONDA_ENV='$CONDA_ENV' RUN_ID_PREFIX='$RUN_ID_PREFIX' \
RNN_MANIFEST='$RNN_MANIFEST' ML_MANIFEST='$ML_MANIFEST' CNN_MANIFEST='$CNN_MANIFEST' \
RNN_MODELS='$RNN_MODELS' INCLUDE_ML='$INCLUDE_ML' INCLUDE_CNN='$INCLUDE_CNN' \
SAMPLE_MODE='$SAMPLE_MODE' DEVICE='$DEVICE' USE_AMP='$USE_AMP' \
RNN_CPU_THREADS='$RNN_CPU_THREADS' ML_CPU_WORKERS='$ML_CPU_WORKERS' \
RNN_HIDDEN_SIZE='$RNN_HIDDEN_SIZE' RNN_NUM_LAYERS='$RNN_NUM_LAYERS' RNN_DROPOUT='$RNN_DROPOUT' \
RNN_BATCH_SIZE='$RNN_BATCH_SIZE' RNN_EPOCHS='$RNN_EPOCHS' RNN_PATIENCE='$RNN_PATIENCE' \
RNN_MIN_EPOCHS='$RNN_MIN_EPOCHS' RNN_TCN_BLOCKS='$RNN_TCN_BLOCKS' RNN_TCN_KERNEL_SIZE='$RNN_TCN_KERNEL_SIZE' \
RNN_THRESHOLD_OBJECTIVE='$RNN_THRESHOLD_OBJECTIVE' RNN_LOSS='$RNN_LOSS' \
CNN_MODEL='$CNN_MODEL' CNN_BATCH_SIZE='$CNN_BATCH_SIZE' CNN_EPOCHS='$CNN_EPOCHS' CNN_IMAGE_SIZE='$CNN_IMAGE_SIZE' \
RUN_ROOT='$run_root' RUN_LOG='$run_log' SUMMARY_JSON='$summary_json' HISTORY_JSONL='$history_jsonl' \
LATEST_LOG_LINK='$LATEST_LOG_LINK' RUNS_CHECKPOINT_DIR='$RUNS_CHECKPOINT_DIR' \
bash '$WORKDIR/scripts/train_all_runner.sh'"

  tmux new-session -d -s "$SESSION_NAME" "bash -lc \"$cmd\""
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists."
      exit 1
    fi
    start_session
    echo "Started tmux session: $SESSION_NAME"
    ;;
  attach)
    tmux attach -t "$SESSION_NAME"
    ;;
  status)
    tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux list-sessions | grep "^$SESSION_NAME:" || {
      echo "Session '$SESSION_NAME' is not running."
      exit 1
    }
    ;;
  stop)
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null && echo "Stopped tmux session: $SESSION_NAME" || {
      echo "Session '$SESSION_NAME' is not running."
      exit 1
    }
    ;;
  logs)
    [[ -L "$LATEST_LOG_LINK" || -f "$LATEST_LOG_LINK" ]] && tail -n 240 "$LATEST_LOG_LINK" || {
      echo "No latest log found"
      exit 1
    }
    ;;
  *)
    usage
    exit 1
    ;;
esac
