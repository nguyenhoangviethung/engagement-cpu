#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="phase34_nocnn_opt"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$WORKDIR/logs"
RUNS_PROCESSED_DIR="$WORKDIR/data/processed/runs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
REPORTS_DIR="$WORKDIR/checkpoints/reports"

COMMAND="start"
RUN_ID_PREFIX="phase34_nocnn_opt"
DEVICE="cuda"
USE_AMP=1
ALLOW_ACTIVE=0
RNN_MODELS="gru gru_basic tcn tiny_transformer"
RNN_THRESHOLD_OBJECTIVE="accuracy"
RNN_LOSS="bce_weighted"
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
ML_CPU_WORKERS=2
ML_THRESHOLD_OBJECTIVE="accuracy"
ML_DIM_REDUCTION="svd"
ML_DIM_COMPONENTS=128
ML_OVERSAMPLE="random"
EXTRACT_LOG_EVERY=25
EXTRACT_FRAME_STRIDE=1
EXTRACT_MAX_FRAMES=120
EXTRACT_RESIZE_WIDTH=320
EXTRACT_FEATURE_SET="enhanced"
EVAL_AGGREGATION="video"

usage() {
  cat <<'USAGE'
Usage: scripts/tmux_train_nocnn_optimized.sh [command] [options]

Commands: start|attach|status|stop|logs

This pipeline does NOT train CNN. It re-extracts optimized MediaPipe features,
then trains/evaluates RNN and ML models in one isolated tmux session.

Options:
  --session NAME
  --run-id-prefix ID
  --env NAME
  --device NAME
  --allow-active              Allow start while phase34 tmux is still running
  --rnn-models "gru gru_basic tcn tiny_transformer"
  --rnn-threshold-objective NAME
  --ml-dim-reduction none|pca|svd
  --ml-dim-components N
  --ml-oversample none|random
  --extract-max-frames N      Uniform frames/video for RNN/ML features
  --extract-resize-width N    Resize width before MediaPipe
  --extract-frame-stride N
  --extract-feature-set base|enhanced
  --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --run-id-prefix) RUN_ID_PREFIX="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --allow-active) ALLOW_ACTIVE=1; shift ;;
    --rnn-models) RNN_MODELS="$2"; shift 2 ;;
    --rnn-threshold-objective) RNN_THRESHOLD_OBJECTIVE="$2"; shift 2 ;;
    --ml-dim-reduction) ML_DIM_REDUCTION="$2"; shift 2 ;;
    --ml-dim-components) ML_DIM_COMPONENTS="$2"; shift 2 ;;
    --ml-oversample) ML_OVERSAMPLE="$2"; shift 2 ;;
    --extract-max-frames) EXTRACT_MAX_FRAMES="$2"; shift 2 ;;
    --extract-resize-width) EXTRACT_RESIZE_WIDTH="$2"; shift 2 ;;
    --extract-frame-stride) EXTRACT_FRAME_STRIDE="$2"; shift 2 ;;
    --extract-feature-set) EXTRACT_FEATURE_SET="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$REPORTS_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_nocnn_optimized.log"

shell_quote() {
  printf "%q" "$1"
}

start_session() {
  if [[ "$ALLOW_ACTIVE" != "1" ]] && tmux has-session -t phase34 2>/dev/null; then
    echo "Refusing to start: tmux session 'phase34' is still running."
    echo "This avoids CPU/GPU/disk contention. Re-run after phase34 finishes, or pass --allow-active intentionally."
    exit 2
  fi

  local timestamp processed_root features_dir manifest run_root run_log summary_json history_jsonl runner_script
  timestamp="$(date +%Y%m%d_%H%M%S)"
  processed_root="$RUNS_PROCESSED_DIR/${RUN_ID_PREFIX}_${timestamp}"
  features_dir="$processed_root/features"
  manifest="$processed_root/feature_manifest.csv"
  run_root="$RUNS_CHECKPOINT_DIR/train_all_${RUN_ID_PREFIX}_${timestamp}"
  run_log="$LOG_DIR/train_nocnn_opt_${RUN_ID_PREFIX}_${timestamp}.log"
  summary_json="$run_root/train_all_summary.json"
  history_jsonl="$REPORTS_DIR/train_all_history.jsonl"

  runner_script="$LOG_DIR/run_${RUN_ID_PREFIX}_${timestamp}.sh"
  cat >"$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(shell_quote "$WORKDIR")
ln -sfn $(shell_quote "$run_log") $(shell_quote "$LATEST_LOG_LINK")
echo "=== no-CNN optimized pipeline started at \$(date) ===" | tee -a $(shell_quote "$run_log")
echo "[1/3] preprocess_labels.py" | tee -a $(shell_quote "$run_log")
$(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -m engagement_daisee.rnn.preprocess_labels 2>&1 | tee -a $(shell_quote "$run_log")
echo "[2/3] optimized extract_features.py" | tee -a $(shell_quote "$run_log")
mkdir -p $(shell_quote "$features_dir") $(shell_quote "$run_root")
$(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.rnn.extract_features \
  --features-dir $(shell_quote "$features_dir") --manifest $(shell_quote "$manifest") --log-every $(shell_quote "$EXTRACT_LOG_EVERY") \
  --frame-stride $(shell_quote "$EXTRACT_FRAME_STRIDE") --max-frames $(shell_quote "$EXTRACT_MAX_FRAMES") --resize-width $(shell_quote "$EXTRACT_RESIZE_WIDTH") \
  --feature-set $(shell_quote "$EXTRACT_FEATURE_SET") 2>&1 | tee -a $(shell_quote "$run_log")
echo "[3/3] train_all_runner.sh without CNN" | tee -a $(shell_quote "$run_log")
WORKDIR=$(shell_quote "$WORKDIR") CONDA_ENV=$(shell_quote "$CONDA_ENV") RUN_ID_PREFIX=$(shell_quote "$RUN_ID_PREFIX") \
RNN_MANIFEST=$(shell_quote "$manifest") ML_MANIFEST=$(shell_quote "$manifest") CNN_MANIFEST='' RNN_MODELS=$(shell_quote "$RNN_MODELS") \
INCLUDE_ML='1' INCLUDE_CNN='0' SAMPLE_MODE='0' DEVICE=$(shell_quote "$DEVICE") USE_AMP=$(shell_quote "$USE_AMP") \
RNN_CPU_THREADS=$(shell_quote "$RNN_CPU_THREADS") ML_CPU_WORKERS=$(shell_quote "$ML_CPU_WORKERS") ML_THRESHOLD_OBJECTIVE=$(shell_quote "$ML_THRESHOLD_OBJECTIVE") \
ML_DIM_REDUCTION=$(shell_quote "$ML_DIM_REDUCTION") ML_DIM_COMPONENTS=$(shell_quote "$ML_DIM_COMPONENTS") ML_OVERSAMPLE=$(shell_quote "$ML_OVERSAMPLE") \
RNN_HIDDEN_SIZE=$(shell_quote "$RNN_HIDDEN_SIZE") RNN_NUM_LAYERS=$(shell_quote "$RNN_NUM_LAYERS") RNN_DROPOUT=$(shell_quote "$RNN_DROPOUT") \
RNN_BATCH_SIZE=$(shell_quote "$RNN_BATCH_SIZE") RNN_EPOCHS=$(shell_quote "$RNN_EPOCHS") RNN_PATIENCE=$(shell_quote "$RNN_PATIENCE") \
RNN_MIN_EPOCHS=$(shell_quote "$RNN_MIN_EPOCHS") RNN_TCN_BLOCKS=$(shell_quote "$RNN_TCN_BLOCKS") RNN_TCN_KERNEL_SIZE=$(shell_quote "$RNN_TCN_KERNEL_SIZE") \
RNN_THRESHOLD_OBJECTIVE=$(shell_quote "$RNN_THRESHOLD_OBJECTIVE") RNN_LOSS=$(shell_quote "$RNN_LOSS") \
RUN_ROOT=$(shell_quote "$run_root") RUN_LOG=$(shell_quote "$run_log") SUMMARY_JSON=$(shell_quote "$summary_json") HISTORY_JSONL=$(shell_quote "$history_jsonl") \
LATEST_LOG_LINK=$(shell_quote "$LATEST_LOG_LINK") RUNS_CHECKPOINT_DIR=$(shell_quote "$RUNS_CHECKPOINT_DIR") RUN_PROCESSED_ROOT=$(shell_quote "$processed_root") \
bash $(shell_quote "$WORKDIR/scripts/train_all_runner.sh")
echo "=== no-CNN optimized pipeline finished at \$(date) ===" | tee -a $(shell_quote "$run_log")
EOF
  chmod +x "$runner_script"

  tmux new-session -d -s "$SESSION_NAME" "bash '$runner_script'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Log: tail -f $run_log"
  echo "Runner: $runner_script"
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists. Use attach/status/logs."
      exit 1
    fi
    start_session
    ;;
  attach) tmux attach -t "$SESSION_NAME" ;;
  status)
    tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux list-sessions | grep "^$SESSION_NAME:" || {
      echo "Session '$SESSION_NAME' is not running."; exit 1;
    }
    ;;
  stop)
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || {
      echo "Session '$SESSION_NAME' is not running."; exit 1;
    }
    echo "Stopped tmux session: $SESSION_NAME"
    ;;
  logs)
    [[ -e "$LATEST_LOG_LINK" ]] || { echo "No latest log found at $LATEST_LOG_LINK"; exit 1; }
    tail -n 160 "$LATEST_LOG_LINK"
    ;;
  *) usage; exit 1 ;;
esac
