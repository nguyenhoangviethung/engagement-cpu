#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/bear/engagement-cpu"
CONDA_ENV="thesis"
SOURCE_PROCESSED="$WORKDIR/data/processed/runs/daisee709_sweep_20260525/openface709_compressed"
RUN_ID="openface709_safe_train_20260525"
RUN_ROOT="$WORKDIR/checkpoints/runs/$RUN_ID"
LOG="$WORKDIR/logs/${RUN_ID}.log"
RESULTS="$RUN_ROOT/results.jsonl"
LATEST="$WORKDIR/logs/openface709_safe_train_latest.log"

# Conservative settings for a 15 GiB RAM host with no swap.
ML_DIMS="96 128 160 192 224 256"
RNN_DIMS="128 192 256 300 384"
RNN_MODELS="gru_basic simple_gru bilstm tiny_transformer tcn"
ML_CPU_WORKERS="1"
CPU_THREADS="2"
RNN_BATCH_SIZE="32"
RNN_HIDDEN_SIZE="128"
RNN_NUM_LAYERS="2"
RNN_DROPOUT="0.30"
RNN_EPOCHS="30"
RNN_PATIENCE="6"
RNN_MIN_EPOCHS="8"
DEVICE="cpu"

mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
ln -sfn "$LOG" "$LATEST"
exec > >(tee -a "$LOG") 2>&1
: > "$RESULTS"

run_py() {
  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" "$@"
}

mem_snapshot() {
  echo "--- memory $(date -Is) ---"
  free -h
  ps -eo pid,ppid,stat,%mem,%cpu,rss,etime,cmd --sort=-rss | head -12
}

append_result() {
  local kind="$1" name="$2" train_path="$3" eval_path="$4" manifest="$5"
  KIND="$kind" NAME="$name" TRAIN_PATH="$train_path" EVAL_PATH="$eval_path" MANIFEST_PATH="$manifest" run_py python - <<'PY' >> "$RESULTS"
import json, os
from pathlib import Path
train = Path(os.environ['TRAIN_PATH'])
evalp = Path(os.environ['EVAL_PATH'])
print(json.dumps({
    'kind': os.environ['KIND'],
    'name': os.environ['NAME'],
    'manifest': os.environ['MANIFEST_PATH'],
    'train': json.loads(train.read_text()) if train.exists() else {},
    'eval': json.loads(evalp.read_text()) if evalp.exists() else {},
}))
PY
}

run_ml() {
  local dim="$1"
  local name="ml_pca${dim}_tsfresh"
  local manifest="$SOURCE_PROCESSED/pca${dim}/feature_manifest.csv"
  local out_dir="$RUN_ROOT/$name"
  local model="$out_dir/engagement_xgb.json"
  local summary="$model"
  summary="${summary%.json}.summary.json"
  local eval_json="$out_dir/eval_test.json"
  local done_file="$out_dir/.done"
  mkdir -p "$out_dir"
  if [[ -f "$done_file" ]]; then
    echo "[skip][ML] $name already done"
    append_result "ml" "$name" "$summary" "$eval_json" "$manifest"
    return
  fi
  echo "[start][ML] $name manifest=$manifest"
  mem_snapshot
  run_py python -u -m engagement_daisee.ml.train \
    --manifest "$manifest" --output "$model" --run-id "${RUN_ID}_${name}" \
    --backend xgboost --feature-mode tsfresh --cpu-workers "$ML_CPU_WORKERS" \
    --threshold-objective accuracy --oversample none --dim-reduction none
  local effective_model="$model"
  local effective_summary="$summary"
  local alt="$WORKDIR/checkpoints/runs/trainml_${RUN_ID}_${name}/engagement_xgb.json"
  if [[ -f "$alt" ]]; then
    effective_model="$alt"
    effective_summary="${alt%.json}.summary.json"
  fi
  run_py python -u -m engagement_daisee.ml.evaluate \
    --manifest "$manifest" --model "$effective_model" --summary-json "$effective_summary" \
    --split test --feature-mode tsfresh --aggregation video --output-json "$eval_json"
  touch "$done_file"
  append_result "ml" "$name" "$effective_summary" "$eval_json" "$manifest"
  mem_snapshot
  echo "[done][ML] $name"
}

run_rnn() {
  local dim="$1" model_name="$2"
  local name="rnn_pca${dim}_${model_name}"
  local manifest="$SOURCE_PROCESSED/pca${dim}/feature_manifest.csv"
  local out_dir="$RUN_ROOT/$name"
  local ckpt="$out_dir/engagement_${model_name}.pt"
  local train_json="${ckpt%.pt}.json"
  local eval_json="$out_dir/eval_test.json"
  local done_file="$out_dir/.done"
  mkdir -p "$out_dir"
  if [[ -f "$done_file" ]]; then
    echo "[skip][RNN] $name already done"
    append_result "rnn" "$name" "$train_json" "$eval_json" "$manifest"
    return
  fi
  echo "[start][RNN] $name manifest=$manifest"
  mem_snapshot
  run_py python -u -m engagement_daisee.rnn.train \
    --manifest "$manifest" --output "$ckpt" --run-id "${RUN_ID}_${name}" \
    --model "$model_name" --device "$DEVICE" --cpu-threads "$CPU_THREADS" \
    --hidden-size "$RNN_HIDDEN_SIZE" --num-layers "$RNN_NUM_LAYERS" --dropout "$RNN_DROPOUT" \
    --batch-size "$RNN_BATCH_SIZE" --epochs "$RNN_EPOCHS" --patience "$RNN_PATIENCE" \
    --min-epochs "$RNN_MIN_EPOCHS" --threshold-objective accuracy --loss focal --train-sampler weighted --no-amp
  run_py python -u -m engagement_daisee.rnn.evaluate \
    --manifest "$manifest" --checkpoint "$ckpt" --split test --batch-size "$RNN_BATCH_SIZE" \
    --aggregation video --output-json "$eval_json"
  touch "$done_file"
  append_result "rnn" "$name" "$train_json" "$eval_json" "$manifest"
  mem_snapshot
  echo "[done][RNN] $name"
}

echo "=== safe OpenFace709 train started at $(date -Is) ==="
echo "run_root=$RUN_ROOT"
echo "source_processed=$SOURCE_PROCESSED"
echo "ML_DIMS=$ML_DIMS RNN_DIMS=$RNN_DIMS RNN_MODELS=$RNN_MODELS"
echo "settings: ml_workers=$ML_CPU_WORKERS rnn_batch=$RNN_BATCH_SIZE hidden=$RNN_HIDDEN_SIZE layers=$RNN_NUM_LAYERS cpu_threads=$CPU_THREADS"
mem_snapshot

for dim in $ML_DIMS; do
  run_ml "$dim"
done

for dim in $RNN_DIMS; do
  for model_name in $RNN_MODELS; do
    run_rnn "$dim" "$model_name"
  done
done

SUMMARY="$RUN_ROOT/openface709_safe_train_summary.json"
RESULTS="$RESULTS" RUN_ID="$RUN_ID" RUN_ROOT="$RUN_ROOT" SOURCE_PROCESSED="$SOURCE_PROCESSED" run_py python - <<'PY'
import json, os
from pathlib import Path
results = Path(os.environ['RESULTS'])
items = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
summary = {
    'run_id': os.environ['RUN_ID'],
    'checkpoint_root': os.environ['RUN_ROOT'],
    'source_processed': os.environ['SOURCE_PROCESSED'],
    'items': items,
}
out = Path(os.environ['RUN_ROOT']) / 'openface709_safe_train_summary.json'
out.write_text(json.dumps(summary, indent=2))
hist = Path('/home/bear/engagement-cpu/checkpoints/reports/openface709_safe_train_history.jsonl')
hist.parent.mkdir(parents=True, exist_ok=True)
with hist.open('a', encoding='utf-8') as f:
    f.write(json.dumps(summary) + '\n')
print('saved_summary=', out)
print('history=', hist)
PY

echo "=== safe OpenFace709 train finished at $(date -Is) ==="
