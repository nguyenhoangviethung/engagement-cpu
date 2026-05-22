#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_train_all"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
REPORTS_DIR="$WORKDIR/checkpoints/reports"

COMMAND="start"
RUN_ID_PREFIX="nightly"
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
ML_CPU_WORKERS=2
CNN_MODEL="mobilenet_v3_small"
CNN_BATCH_SIZE=64
CNN_EPOCHS=20
CNN_IMAGE_SIZE=112

EXAMPLE_MODE=0

usage() {
  cat <<'EOF'
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
  --ml-cpu-workers N
  --cnn-model NAME
  --cnn-batch-size N
  --cnn-epochs N
  --cnn-image-size N
  --session NAME
  --env NAME
  --example            Help-only smoke run in tmux
EOF
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

build_command() {
  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local run_root="$RUNS_CHECKPOINT_DIR/train_all_${RUN_ID_PREFIX}_${timestamp}"
  local run_log="$LOG_DIR/train_all_${RUN_ID_PREFIX}_${timestamp}.log"
  local summary_json="$run_root/train_all_summary.json"
  local history_jsonl="$REPORTS_DIR/train_all_history.jsonl"

  local sample_flag=""
  [[ "$SAMPLE_MODE" -eq 1 ]] && sample_flag=" --sample"

  local amp_flag=""
  [[ "$USE_AMP" -eq 1 ]] && amp_flag=" --amp"

  if [[ "$EXAMPLE_MODE" -eq 1 ]]; then
    cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== train_all example started at \$(date) ===" | tee -a "$run_log"
./scripts/rnn/tmux_train_eval.sh --help 2>&1 | tee -a "$run_log"
./scripts/ml/tmux_train_eval.sh --help 2>&1 | tee -a "$run_log"
./scripts/cnn/tmux_train_eval.sh --help 2>&1 | tee -a "$run_log"
echo "=== train_all example finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
    return
  fi

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

mkdir -p "$run_root"

echo "=== train_all started at \$(date) ===" | tee -a "$run_log"
echo "run_root=$run_root" | tee -a "$run_log"

results_jsonl="$run_root/results.jsonl"
: > "\$results_jsonl"

for model in $RNN_MODELS; do
  rid="${RUN_ID_PREFIX}_rnn_\$model"
  out_dir="$run_root/rnn_\$model"
  mkdir -p "\$out_dir"
  ckpt="\$out_dir/engagement_gru.pt"
  eval_json="\$out_dir/eval_test.json"
  agg_json="\$out_dir/train_eval_summary.json"

  echo "[RNN] model=\$model rid=\$rid" | tee -a "$run_log"
  conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.train$sample_flag$amp_flag \\
    --manifest "$RNN_MANIFEST" \\
    --output "\$ckpt" \\
    --run-id "\$rid" \\
    --model "\$model" \\
    --device "$DEVICE" \\
    --cpu-threads "$RNN_CPU_THREADS" 2>&1 | tee -a "$run_log"

  conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.evaluate \\
    --manifest "$RNN_MANIFEST" \\
    --checkpoint "\$ckpt" \\
    --split test \\
    --batch-size 128 \\
    --output-json "\$eval_json" 2>&1 | tee -a "$run_log"

  conda run --no-capture-output -n "$CONDA_ENV" python - <<"PY" "\$out_dir/engagement_gru.json" "\$eval_json" "\$agg_json" "\$rid" "\$model"
import json, sys
from pathlib import Path
train_path, eval_path, agg_path, rid, model = [Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]]
train = json.loads(train_path.read_text()) if train_path.exists() else {}
evalp = json.loads(eval_path.read_text()) if eval_path.exists() else {}
payload = {"run_id": rid, "module": "rnn", "model": model, "train": train, "eval": evalp}
agg_path.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload))
PY | tee -a "$run_log" >> "\$results_jsonl"
done

if [[ "$INCLUDE_ML" -eq 1 ]]; then
  rid="${RUN_ID_PREFIX}_ml"
  out_dir="$run_root/ml"
  mkdir -p "\$out_dir"
  model_path="\$out_dir/engagement_xgb.json"
  eval_json="\$out_dir/eval_test.json"
  agg_json="\$out_dir/train_eval_summary.json"

  echo "[ML] rid=\$rid" | tee -a "$run_log"
  conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.train$sample_flag \\
    --manifest "$ML_MANIFEST" \\
    --output "\$model_path" \\
    --run-id "\$rid" \\
    --backend auto \\
    --feature-mode tsfresh \\
    --cpu-workers "$ML_CPU_WORKERS" 2>&1 | tee -a "$run_log"

  conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.evaluate \\
    --manifest "$ML_MANIFEST" \\
    --model "\$model_path" \\
    --split test \\
    --feature-mode tsfresh \\
    --summary-json "\$out_dir/engagement_xgb.summary.json" \\
    --output-json "\$eval_json" 2>&1 | tee -a "$run_log"

  conda run --no-capture-output -n "$CONDA_ENV" python - <<"PY" "\$out_dir/engagement_xgb.summary.json" "\$eval_json" "\$agg_json" "\$rid"
import json, sys
from pathlib import Path
train_path, eval_path, agg_path, rid = [Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]]
train = json.loads(train_path.read_text()) if train_path.exists() else {}
evalp = json.loads(eval_path.read_text()) if eval_path.exists() else {}
payload = {"run_id": rid, "module": "ml", "train": train, "eval": evalp}
agg_path.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload))
PY | tee -a "$run_log" >> "\$results_jsonl"
fi

if [[ "$INCLUDE_CNN" -eq 1 ]]; then
  if [[ ! -f "$CNN_MANIFEST" ]]; then
    echo "[CNN] skipped: manifest not found at $CNN_MANIFEST" | tee -a "$run_log"
  else
    rid="${RUN_ID_PREFIX}_cnn"
    out_dir="$run_root/cnn"
    mkdir -p "\$out_dir"
    ckpt="\$out_dir/engagement_cnn.pt"
    eval_json="\$out_dir/eval_test.json"
    agg_json="\$out_dir/train_eval_summary.json"

    echo "[CNN] rid=\$rid" | tee -a "$run_log"
    conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.train$sample_flag \\
      --manifest "$CNN_MANIFEST" \\
      --output "\$ckpt" \\
      --run-id "\$rid" \\
      --model "$CNN_MODEL" \\
      --image-size "$CNN_IMAGE_SIZE" \\
      --batch-size "$CNN_BATCH_SIZE" \\
      --epochs "$CNN_EPOCHS" 2>&1 | tee -a "$run_log"

    conda run --no-capture-output -n "$CONDA_ENV" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.evaluate \\
      --manifest "$CNN_MANIFEST" \\
      --checkpoint "\$ckpt" \\
      --split test \\
      --batch-size 256 \\
      --output-json "\$eval_json" 2>&1 | tee -a "$run_log"

    conda run --no-capture-output -n "$CONDA_ENV" python - <<"PY" "\$out_dir/engagement_cnn.json" "\$eval_json" "\$agg_json" "\$rid" "$CNN_MODEL"
import json, sys
from pathlib import Path
train_path, eval_path, agg_path, rid, model = [Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]]
train = json.loads(train_path.read_text()) if train_path.exists() else {}
evalp = json.loads(eval_path.read_text()) if eval_path.exists() else {}
payload = {"run_id": rid, "module": "cnn", "model": model, "train": train, "eval": evalp}
agg_path.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload))
PY | tee -a "$run_log" >> "\$results_jsonl"
  fi
fi

conda run --no-capture-output -n "$CONDA_ENV" python - <<"PY" "\$results_jsonl" "$summary_json" "$history_jsonl" "$RUN_ID_PREFIX"
import json, sys
from pathlib import Path
results_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
history_path = Path(sys.argv[3])
prefix = sys.argv[4]

items = []
for line in results_path.read_text().splitlines():
    line = line.strip()
    if line:
        items.append(json.loads(line))

payload = {"run_id_prefix": prefix, "items": items}
summary_path.write_text(json.dumps(payload, indent=2))
with history_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=True) + "\\n")
print(json.dumps({"saved": str(summary_path), "history": str(history_path), "count": len(items)}, indent=2))
PY | tee -a "$run_log"

echo "=== train_all finished at \$(date) ===" | tee -a "$run_log"
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
  logs) [[ -L "$LATEST_LOG_LINK" || -f "$LATEST_LOG_LINK" ]] && tail -n 240 "$LATEST_LOG_LINK" || { echo "No latest log found"; exit 1; } ;;
  *) usage; exit 1 ;;
esac
