#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/bear/engagement-cpu"
CONDA_ENV="thesis"
SOURCE_PROCESSED="$WORKDIR/data/processed/runs/daisee709_sweep_20260525/openface709_compressed"
RUN_ID="openface709_final_push_20260525"
RUN_ROOT="$WORKDIR/checkpoints/runs/$RUN_ID"
LOG="$WORKDIR/logs/${RUN_ID}.log"

# Final run: prioritize strongest settings while keeping memory safe.
DIMS="160 192 224 256"
MODELS_GPU="stgcn tcn tiny_transformer"
BATCH_SIZE="128"
HIDDEN_SIZE="192"
NUM_LAYERS="3"
DROPOUT="0.25"
EPOCHS="28"
PATIENCE="7"
MIN_EPOCHS="10"
CPU_THREADS="2"

mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

run_py() {
  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" "$@"
}

mem_snapshot() {
  echo "--- snapshot $(date -Is) ---"
  free -h
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader || true
  ps -eo pid,ppid,stat,%mem,%cpu,rss,etime,cmd --sort=-rss | head -12
}

evaluate_ckpt() {
  local manifest="$1"; local ckpt="$2"; local out_json="$3"
  run_py python -u -m engagement_daisee.rnn.evaluate \
    --manifest "$manifest" --checkpoint "$ckpt" --split test \
    --batch-size "$BATCH_SIZE" --aggregation video --output-json "$out_json"
}

echo "=== final push started $(date -Is) ==="
mem_snapshot

for dim in $DIMS; do
  manifest="$SOURCE_PROCESSED/pca${dim}/feature_manifest.csv"
  if [[ ! -f "$manifest" ]]; then
    echo "[skip] missing manifest $manifest"
    continue
  fi

  for model in $MODELS_GPU; do
    name="rnn_pca${dim}_${model}"
    out_dir="$RUN_ROOT/$name"
    mkdir -p "$out_dir"
    ckpt="$out_dir/engagement_${model}.pt"
    eval_json="$out_dir/eval_test.json"
    done_file="$out_dir/.done"

    if [[ -f "$done_file" ]]; then
      echo "[skip] $name already done"
      continue
    fi

    echo "[start] $name"
    mem_snapshot

    run_py python -u -m engagement_daisee.rnn.train \
      --manifest "$manifest" --output "$ckpt" --run-id "${RUN_ID}_${name}" \
      --model "$model" --device cuda --amp --cpu-threads "$CPU_THREADS" \
      --hidden-size "$HIDDEN_SIZE" --num-layers "$NUM_LAYERS" --dropout "$DROPOUT" \
      --tcn-blocks 4 --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --patience "$PATIENCE" \
      --min-epochs "$MIN_EPOCHS" --threshold-objective balanced_accuracy \
      --loss focal --train-sampler weighted

    evaluate_ckpt "$manifest" "$ckpt" "$eval_json"
    touch "$done_file"
    echo "[done] $name"
    mem_snapshot
  done
done

# Rank all results in this final run.
RUN_ROOT="$RUN_ROOT" run_py python - <<'PY'
import json, glob, os
from pathlib import Path
run_root = Path(os.environ['RUN_ROOT'])
rows = []
for p in glob.glob(str(run_root / '**/eval_test.json'), recursive=True):
    d = json.loads(Path(p).read_text())
    m = d.get('metrics') or d.get('video_metrics') or {}
    rows.append({
        'path': p,
        'balanced_accuracy': m.get('balanced_accuracy', -1),
        'accuracy': m.get('accuracy', -1),
        'f1_macro': m.get('f1_macro', -1),
        'recall_pos': m.get('recall_pos', -1),
        'recall_neg': m.get('recall_neg', -1),
    })
rows.sort(key=lambda x: x['balanced_accuracy'], reverse=True)
out = run_root / 'leaderboard.json'
out.write_text(json.dumps(rows, indent=2))
print('saved', out)
print('top3')
for r in rows[:3]:
    print(r)
PY

echo "=== final push finished $(date -Is) ==="
