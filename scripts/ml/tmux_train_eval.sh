#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_ml_train_eval"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$WORKDIR/logs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
REPORTS_DIR="$WORKDIR/checkpoints/runs"

COMMAND="start"
RUN_ID=""
MANIFEST="$WORKDIR/data/processed/runs/baseline_pipeline_features/feature_manifest.csv"
SAMPLE_MODE=0
SAMPLE_VIDEOS=10
BACKEND="auto"
FEATURE_MODE="tsfresh"
CPU_WORKERS=2
SPLIT="test"
THRESHOLD=""
EXAMPLE_MODE=0

usage() {
  cat <<'EOF'
Usage: scripts/ml/tmux_train_eval.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --manifest PATH
  --run-id ID
  --sample
  --sample-videos N
  --backend auto|xgboost|lightgbm
  --feature-mode basic|tsfresh|copur
  --cpu-workers N
  --split train|validation|test
  --threshold V
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
    --sample-videos) SAMPLE_VIDEOS="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --example) EXAMPLE_MODE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$REPORTS_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_ml_train_eval.log"

build_command() {
  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local active_run_id="$RUN_ID"
  if [[ -z "$active_run_id" ]]; then
    active_run_id="$timestamp"
  fi

  local run_dir="$RUNS_CHECKPOINT_DIR/ml_train_eval_$active_run_id"
  local model_path="$run_dir/engagement_xgb.json"
  local train_summary="$run_dir/engagement_xgb.summary.json"
  local eval_json="$run_dir/eval_${SPLIT}.json"
  local aggregate_json="$run_dir/train_eval_summary.json"
  local history_jsonl="$RUNS_CHECKPOINT_DIR/ml_train_eval_$active_run_id/history.jsonl"
  local run_log="$LOG_DIR/ml_train_eval_${active_run_id}_${timestamp}.log"

  local sample_flag=""
  [[ "$SAMPLE_MODE" -eq 1 ]] && sample_flag=" --sample --sample-videos $SAMPLE_VIDEOS"

  local threshold_flag=""
  [[ -n "$THRESHOLD" ]] && threshold_flag=" --threshold $THRESHOLD"

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== ML train+eval example started at \$(date) ===" | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.train --help 2>&1 | tee -a "$run_log"
"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.evaluate --help 2>&1 | tee -a "$run_log"
echo "=== ML train+eval example finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
    return
  fi

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

mkdir -p "$run_dir"
echo "=== ML train+eval started at \$(date) ===" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Backend: $BACKEND" | tee -a "$run_log"
echo "Manifest: $MANIFEST" | tee -a "$run_log"
echo "Model path: $model_path" | tee -a "$run_log"

"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.train$sample_flag$threshold_flag \
  --manifest "$MANIFEST" \
  --output "$model_path" \
  --run-id "$active_run_id" \
  --backend "$BACKEND" \
  --feature-mode "$FEATURE_MODE" \
  --cpu-workers "$CPU_WORKERS" 2>&1 | tee -a "$run_log"

"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.evaluate \
  --manifest "$MANIFEST" \
  --model "$model_path" \
  --split "$SPLIT" \
  --feature-mode "$FEATURE_MODE" \
  --summary-json "$train_summary" \
  --output-json "$eval_json"$threshold_flag 2>&1 | tee -a "$run_log"

"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" python - <<"PY" "$train_summary" "$eval_json" "$aggregate_json" "$history_jsonl" "$active_run_id" "$BACKEND" "$FEATURE_MODE" "$MANIFEST" "$model_path"
import json
import sys
from pathlib import Path

train_summary_path = Path(sys.argv[1])
eval_json_path = Path(sys.argv[2])
aggregate_json_path = Path(sys.argv[3])
history_jsonl_path = Path(sys.argv[4])
run_id = sys.argv[5]
backend = sys.argv[6]
feature_mode = sys.argv[7]
manifest = sys.argv[8]
model_path = sys.argv[9]

train_payload = json.loads(train_summary_path.read_text()) if train_summary_path.exists() else {}
eval_payload = json.loads(eval_json_path.read_text()) if eval_json_path.exists() else {}

summary = {
    "run_id": run_id,
    "module": "ml",
    "backend": backend,
    "feature_mode": feature_mode,
    "manifest": manifest,
    "model_path": model_path,
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

echo "=== ML train+eval finished at \$(date) ===" | tee -a "$run_log"
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
