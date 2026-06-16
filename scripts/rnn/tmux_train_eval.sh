#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_rnn_train_eval"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
REPORTS_DIR="$WORKDIR/checkpoints/runs"

COMMAND="start"
RUN_ID=""
MANIFEST="$WORKDIR/data/processed/runs/baseline_pipeline_features/feature_manifest.csv"
SAMPLE_MODE=0
MODEL_NAME="gru"
CPU_THREADS=2
DEVICE="auto"
USE_AMP=0
LOG_EVERY=1
SPLIT="test"
BATCH_SIZE=128
EPOCHS=40
PATIENCE=10
MIN_EPOCHS=12
HIDDEN_SIZE=192
NUM_LAYERS=3
DROPOUT=0.25
LR=3e-4
WEIGHT_DECAY=1e-4
SCHEDULER="plateau"
FREEZE_FEATURE_EPOCHS=0
THRESHOLD=""
RESUME_FROM=""
RESUME_LAST=0
EXAMPLE_MODE=0

usage() {
  cat <<'EOF'
Usage: scripts/rnn/tmux_train_eval.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --manifest PATH
  --run-id ID
  --sample
  --model NAME           gru|hybrid_attn|gru_basic|bilstm|tcn|transformer|tiny_transformer
  --cpu-threads N
  --device NAME
  --amp
  --log-every N
  --split train|validation|test
  --batch-size N
  --epochs N
  --patience N
  --min-epochs N
  --hidden-size N
  --num-layers N
  --dropout V
  --lr V
  --weight-decay V
  --scheduler plateau|cosine|none
  --freeze-feature-epochs N
  --threshold V
  --resume-from PATH
  --resume-last
  --session NAME
  --env NAME
  --example              Run tmux smoke test (help-only)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --sample) SAMPLE_MODE=1; shift ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    --cpu-threads) CPU_THREADS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --amp) USE_AMP=1; shift ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --min-epochs) MIN_EPOCHS="$2"; shift 2 ;;
    --hidden-size) HIDDEN_SIZE="$2"; shift 2 ;;
    --num-layers) NUM_LAYERS="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --scheduler) SCHEDULER="$2"; shift 2 ;;
    --freeze-feature-epochs) FREEZE_FEATURE_EPOCHS="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --resume-last) RESUME_LAST=1; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --example) EXAMPLE_MODE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$REPORTS_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_rnn_train_eval.log"

build_command() {
  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local active_run_id="$RUN_ID"
  if [[ -z "$active_run_id" ]]; then
    active_run_id="$timestamp"
  fi

  local run_dir="$RUNS_CHECKPOINT_DIR/rnn_train_eval_$active_run_id"
  local safe_model_name="${MODEL_NAME//[^A-Za-z0-9_]/_}"
  local checkpoint="$run_dir/engagement_${safe_model_name}.pt"
  local train_summary="$run_dir/engagement_${safe_model_name}.json"
  local eval_json="$run_dir/eval_${SPLIT}.json"
  local aggregate_json="$run_dir/train_eval_summary.json"
  local history_jsonl="$RUNS_CHECKPOINT_DIR/rnn_train_eval_$active_run_id/history.jsonl"
  local run_log="$LOG_DIR/rnn_train_eval_${active_run_id}_${timestamp}.log"

  local sample_flag=""
  [[ "$SAMPLE_MODE" -eq 1 ]] && sample_flag=" --sample"

  local amp_flag=""
  [[ "$USE_AMP" -eq 1 ]] && amp_flag=" --amp"

  local threshold_flag=""
  [[ -n "$THRESHOLD" ]] && threshold_flag=" --threshold $THRESHOLD"

  local resume_flag=""
  local last_ckpt="$run_dir/engagement_${safe_model_name}.last.pt"
  if [[ -n "$RESUME_FROM" ]]; then
    resume_flag=" --resume-from \"$RESUME_FROM\""
  elif [[ "$RESUME_LAST" -eq 1 && -f "$last_ckpt" ]]; then
    resume_flag=" --resume-from \"$last_ckpt\""
  fi

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== RNN train+eval example started at \$(date) ===" | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.train --help 2>&1 | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.evaluate --help 2>&1 | tee -a "$run_log"
echo "=== RNN train+eval example finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
    return
  fi

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

mkdir -p "$run_dir"
echo "=== RNN train+eval started at \$(date) ===" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Model: $MODEL_NAME" | tee -a "$run_log"
echo "Manifest: $MANIFEST" | tee -a "$run_log"
echo "Checkpoint: $checkpoint" | tee -a "$run_log"

"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.train$sample_flag$amp_flag \
  --manifest "$MANIFEST" \
  --output "$checkpoint" \
  --run-id "$active_run_id" \
  --model "$MODEL_NAME" \
  --cpu-threads "$CPU_THREADS" \
  --device "$DEVICE" \
  --hidden-size "$HIDDEN_SIZE" \
  --num-layers "$NUM_LAYERS" \
  --dropout "$DROPOUT" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --min-epochs "$MIN_EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --scheduler "$SCHEDULER" \
  --freeze-feature-epochs "$FREEZE_FEATURE_EPOCHS" \
  --log-every "$LOG_EVERY"$resume_flag 2>&1 | tee -a "$run_log"

"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.evaluate \
  --manifest "$MANIFEST" \
  --checkpoint "$checkpoint" \
  --split "$SPLIT" \
  --batch-size "$BATCH_SIZE" \
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
    "module": "rnn",
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

echo "=== RNN train+eval finished at \$(date) ===" | tee -a "$run_log"
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
