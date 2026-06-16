#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."
ENV_NAME="thesis"
RUN_ID="${RUN_ID:-goal70_shift_calibrated_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$WORKDIR/checkpoints/runs/$RUN_ID"
LOG="$WORKDIR/logs/${RUN_ID}.log"

MANIFEST_BASE="$WORKDIR/data/processed/runs/openface709_pca_sweep/openface709_compressed"
DIMS="192 224"
MODELS="tiny_transformer stgcn"
SEEDS="42 52 62"

mkdir -p "$RUN_ROOT" "$WORKDIR/logs"
exec > >(tee -a "$LOG") 2>&1

run_py() {
  "$WORKDIR/scripts/lib/run_python.sh" --env "$ENV_NAME" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" "$@"
}

echo "=== $RUN_ID started $(date -Is) ==="

for dim in $DIMS; do
  manifest="$MANIFEST_BASE/pca${dim}/feature_manifest.csv"
  for model in $MODELS; do
    for seed in $SEEDS; do
      name="rnn_pca${dim}_${model}_s${seed}"
      out_dir="$RUN_ROOT/$name"
      mkdir -p "$out_dir"
      ckpt="$out_dir/engagement_${model}.pt"
      last_ckpt="$out_dir/engagement_${model}.last.pt"
      eval_json="$out_dir/eval_test.json"

      if [[ -f "$eval_json" ]]; then
        echo "[skip] $name (eval exists)"
        continue
      fi

      echo "[start] $name"
      resume_args=()
      if [[ -f "$last_ckpt" ]]; then
        resume_args=(--resume-from "$last_ckpt")
        echo "[resume] $name from $last_ckpt"
      fi
      run_py python -u -m engagement_daisee.rnn.train \
        --manifest "$manifest" --output "$ckpt" --run-id "${RUN_ID}_${name}" \
        --model "$model" --device cuda --amp --cpu-threads 2 \
        --hidden-size 192 --num-layers 3 --dropout 0.25 --tcn-blocks 4 \
        --batch-size 128 --epochs 36 --patience 9 --min-epochs 12 \
        --threshold-objective balanced_accuracy --loss focal --train-sampler weighted \
        --seed "$seed" --calibration-temperature-grid 0.75,0.9,1.0,1.1,1.25,1.5 \
        "${resume_args[@]}"

      run_py python -u -m engagement_daisee.rnn.evaluate \
        --manifest "$manifest" --checkpoint "$ckpt" --split test --batch-size 128 \
        --aggregation video --output-json "$eval_json"

      echo "[done] $name"
    done
  done
done

RUN_ROOT="$RUN_ROOT" run_py python - <<'PY'
import json,glob,os
from pathlib import Path
run_root=Path(os.environ['RUN_ROOT'])
rows=[]
for p in glob.glob(str(run_root/'**/eval_test.json'),recursive=True):
    d=json.loads(Path(p).read_text())
    m=d['metrics']
    rows.append({'path':p,'balanced_accuracy':m['balanced_accuracy'],'accuracy':m['accuracy'],'f1_macro':m['f1_macro'],'recall_pos':m['recall_pos'],'recall_neg':m['recall_neg']})
rows.sort(key=lambda x:x['balanced_accuracy'],reverse=True)
out=run_root/'leaderboard.json'
out.write_text(json.dumps(rows,indent=2))
print('saved',out)
print('top5')
for r in rows[:5]:
    print(r)
PY

echo "=== $RUN_ID finished $(date -Is) ==="
