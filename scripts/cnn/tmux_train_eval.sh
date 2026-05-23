#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_cnn_train_eval"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
REPORTS_DIR="$WORKDIR/checkpoints/reports"

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
IMAGE_SIZE=224
USE_PRETRAINED=1
FREEZE_BACKBONE=0
TRAIN_SAMPLER="weighted"
SPLIT="test"
EVAL_BATCH_SIZE=256
THRESHOLD=""
EXAMPLE_MODE=0

usage() {
  cat <<'EOF'
Usage: scripts/cnn/tmux_train_eval.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --manifest PATH
  --run-id ID
  --sample
  --model NAME             mobilenet_v3_small|efficientnet_b0|tinycnn
  --batch-size N
  --epochs N
  --lr V
  --weight-decay V
  --patience N
  --image-size N
  --pretrained
  --no-pretrained
  --freeze-backbone
  --train-sampler NAME     weighted|shuffle
  --split train|validation|test
  --eval-batch-size N
  --threshold V
  --session NAME
  --env NAME
  --example                Run tmux smoke test (help-only)
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
    --no-pretrained) USE_PRETRAINED=0; shift ;;
    --freeze-backbone) FREEZE_BACKBONE=1; shift ;;
    --train-sampler) TRAIN_SAMPLER="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --example) EXAMPLE_MODE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$REPORTS_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_cnn_train_eval.log"

build_command() {
  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local active_run_id="$RUN_ID"
  if [[ -z "$active_run_id" ]]; then
    active_run_id="$timestamp"
  fi

  local run_dir="$RUNS_CHECKPOINT_DIR/cnn_train_eval_$active_run_id"
  local checkpoint="$run_dir/engagement_cnn.pt"
  local train_summary="$run_dir/engagement_cnn.json"
  local eval_json="$run_dir/eval_${SPLIT}.json"
  local aggregate_json="$run_dir/train_eval_summary.json"
  local history_jsonl="$REPORTS_DIR/cnn_train_eval_history.jsonl"
  local run_log="$LOG_DIR/cnn_train_eval_${active_run_id}_${timestamp}.log"

  local sample_flag=""
  [[ "$SAMPLE_MODE" -eq 1 ]] && sample_flag=" --sample"

  local pretrained_flag=""
  [[ "$USE_PRETRAINED" -eq 1 ]] && pretrained_flag=" --pretrained"

  local freeze_flag=""
  [[ "$FREEZE_BACKBONE" -eq 1 ]] && freeze_flag=" --freeze-backbone"

  local threshold_flag=""
  [[ -n "$THRESHOLD" ]] && threshold_flag=" --threshold $THRESHOLD"

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== CNN train+eval example started at \$(date) ===" | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.train --help 2>&1 | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.evaluate --help 2>&1 | tee -a "$run_log"
echo "=== CNN train+eval example finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
    return
  fi

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

mkdir -p "$run_dir"
echo "=== CNN train+eval started at \$(date) ===" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Model: $MODEL_NAME" | tee -a "$run_log"
echo "Manifest: $MANIFEST" | tee -a "$run_log"
echo "Checkpoint: $checkpoint" | tee -a "$run_log"

"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.cnn.train$sample_flag \
  --manifest "$MANIFEST" \
  --output "$checkpoint" \
  --model "$MODEL_NAME" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --patience "$PATIENCE" \
  --train-sampler "$TRAIN_SAMPLER" \
  --run-id "$active_run_id"$pretrained_flag$freeze_flag 2>&1 | tee -a "$run_log"

"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.evaluate \
  --manifest "$MANIFEST" \
  --checkpoint "$checkpoint" \
  --split "$SPLIT" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --output-json "$eval_json"$threshold_flag 2>&1 | tee -a "$run_log"

"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" python - <<"PY" "$train_summary" "$eval_json" "$aggregate_json" "$history_jsonl" "$active_run_id" "$MODEL_NAME" "$MANIFEST" "$checkpoint"
import json
import sys
from pathlib import Path

train_summary_path = Path(sys.argv[1])
eval_json_path = Path(sys.argv[2])
aggregate_json_path = Path(sys.argv[3])
history_jsonl_path = Path(sys.argv[4])
run_id = sys.argv[5]
model_name = sys.argv[6]
manifest = sys.argv[7]
checkpoint = sys.argv[8]

train_payload = json.loads(train_summary_path.read_text()) if train_summary_path.exists() else {}
eval_payload = json.loads(eval_json_path.read_text()) if eval_json_path.exists() else {}

summary = {
    "run_id": run_id,
    "module": "cnn",
    "model": model_name,
    "manifest": manifest,
    "checkpoint": checkpoint,
    "train_summary_path": str(train_summary_path),
    "eval_summary_path": str(eval_json_path),
    "train": train_payload,
    "eval": eval_payload,
}
aggregate_json_path.write_text(json.dumps(summary, indent=2))
with history_jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(summary, ensure_ascii=True) + "\\n")
print(json.dumps({"saved": str(aggregate_json_path), "history": str(history_jsonl_path)}, indent=2))
PY

echo "=== CNN train+eval finished at \$(date) ===" | tee -a "$run_log"
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
  logs) [[ -L "$LATEST_LOG_LINK" || -f "$LATEST_LOG_LINK" ]] && tail -n 180 "$LATEST_LOG_LINK" || { echo "No latest log found"; exit 1; } ;;
  *) usage; exit 1 ;;
esac
