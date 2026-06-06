#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/bear/engagement-cpu"
CONDA_ENV="thesis"
SOURCE_PROCESSED="$WORKDIR/data/processed/runs/daisee709_sweep_20260525/openface709_compressed"
RUN_ID="stgcn_gpu_parallel_20260525"
RUN_ROOT="$WORKDIR/checkpoints/runs/$RUN_ID"
LOG="$WORKDIR/logs/${RUN_ID}.log"
RESULTS="$RUN_ROOT/results.jsonl"

DIMS="128 160 192 224 256 300 384"
BATCH_SIZE="192"
EPOCHS="24"
PATIENCE="6"
MIN_EPOCHS="8"
HIDDEN="192"
LAYERS="3"
DROPOUT="0.25"
CPU_THREADS="2"

mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
: > "$RESULTS"

run_py() {
  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" "$@"
}

append_result() {
  local name="$1" train_json="$2" eval_json="$3" manifest="$4"
  NAME="$name" TRAIN_PATH="$train_json" EVAL_PATH="$eval_json" MANIFEST_PATH="$manifest" run_py python - <<'PY' >> "$RESULTS"
import json, os
from pathlib import Path
train = Path(os.environ['TRAIN_PATH'])
evalp = Path(os.environ['EVAL_PATH'])
print(json.dumps({
  'kind':'rnn','model':'stgcn','name':os.environ['NAME'],'manifest':os.environ['MANIFEST_PATH'],
  'train': json.loads(train.read_text()) if train.exists() else {},
  'eval': json.loads(evalp.read_text()) if evalp.exists() else {},
}))
PY
}

echo "=== stgcn gpu parallel started $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader || true

for dim in $DIMS; do
  manifest="$SOURCE_PROCESSED/pca${dim}/feature_manifest.csv"
  out_dir="$RUN_ROOT/rnn_pca${dim}_stgcn"
  ckpt="$out_dir/engagement_stgcn.pt"
  train_json="${ckpt%.pt}.json"
  eval_json="$out_dir/eval_test.json"
  done_file="$out_dir/.done"

  mkdir -p "$out_dir"
  if [[ -f "$done_file" ]]; then
    echo "[skip] pca${dim} done"
    append_result "rnn_pca${dim}_stgcn" "$train_json" "$eval_json" "$manifest"
    continue
  fi

  echo "[start] pca${dim} stgcn"
  run_py python -u -m engagement_daisee.rnn.train \
    --manifest "$manifest" --output "$ckpt" --run-id "${RUN_ID}_pca${dim}_stgcn" \
    --model stgcn --device cuda --amp --cpu-threads "$CPU_THREADS" \
    --hidden-size "$HIDDEN" --num-layers "$LAYERS" --dropout "$DROPOUT" \
    --tcn-blocks 4 --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" \
    --patience "$PATIENCE" --min-epochs "$MIN_EPOCHS" \
    --threshold-objective balanced_accuracy --loss focal --train-sampler weighted

  run_py python -u -m engagement_daisee.rnn.evaluate \
    --manifest "$manifest" --checkpoint "$ckpt" --split test --batch-size "$BATCH_SIZE" \
    --aggregation video --output-json "$eval_json"

  touch "$done_file"
  append_result "rnn_pca${dim}_stgcn" "$train_json" "$eval_json" "$manifest"
  echo "[done] pca${dim} stgcn"
done

run_py python - <<'PY'
import json
from pathlib import Path
run_root = Path('/home/bear/engagement-cpu/checkpoints/runs/stgcn_gpu_parallel_20260525')
results = run_root / 'results.jsonl'
items = [json.loads(x) for x in results.read_text().splitlines() if x.strip()]
(run_root / 'summary.json').write_text(json.dumps({'run_id':'stgcn_gpu_parallel_20260525','items':items}, indent=2))
print('saved', run_root / 'summary.json')
PY

echo "=== stgcn gpu parallel finished $(date -Is) ==="
