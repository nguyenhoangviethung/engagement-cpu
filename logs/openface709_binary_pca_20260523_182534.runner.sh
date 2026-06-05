#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$RUN_ROOT" "$PROCESSED_ROOT" "$(dirname "$LOG")"
ln -sfn "$LOG" "$LATEST"
exec > >(tee -a "$LOG") 2>&1

echo "=== OpenFace709 pipeline started at $(date) ==="
echo "run_id=$RUN_ID"
echo "processed_root=$PROCESSED_ROOT"
echo "checkpoint_root=$RUN_ROOT"
echo "components=$COMPONENTS | rnn_dims=$RNN_DIMS | ml_dims=$ML_DIMS | rnn_models=$RNN_MODELS"

run_py() {
  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" "$@"
}

sample_args=()
if [[ "$SAMPLE" == "1" ]]; then
  sample_args=(--sample --sample-videos 12)
fi

echo "[1/5] Preprocess labels with official binary rule"
run_py python -m engagement_daisee.rnn.preprocess_labels
run_py python - <<'PY'
import pandas as pd
p='data/processed/engagement_only_labels.csv'
df=pd.read_csv(p)
print('label_rows=', len(df), 'label_counts=', df.engagement_binary.value_counts().sort_index().to_dict(), 'split_counts=', df.split.value_counts().to_dict())
assert set(df.engagement_binary.unique()) <= {0, 1}
PY

echo "[2/5] Extract exact OpenFace 709-D features"
run_py python -u -m engagement_daisee.openface.extract_openface709 "${sample_args[@]}" \
  --output-root "$PROCESSED_ROOT/openface709" \
  --max-frames "$MAX_FRAMES" \
  --sequence-length "$SEQUENCE_LENGTH" \
  --log-every 20
MANIFEST="$PROCESSED_ROOT/openface709/feature_manifest.csv"
SCHEMA="$PROCESSED_ROOT/openface709/openface709_schema.json"
test -f "$MANIFEST"
test -f "$SCHEMA"
MANIFEST="$MANIFEST" SCHEMA="$SCHEMA" run_py python - <<'PY'
import json, os, pandas as pd
m=os.environ['MANIFEST']
s=os.environ['SCHEMA']
df=pd.read_csv(m)
schema=json.load(open(s))
print('manifest_rows=', len(df), 'feature_dim_values=', sorted(df.feature_dim.unique().tolist()))
print('feature_columns=', len(schema['feature_columns']))
print('label_counts=', df.label.value_counts().sort_index().to_dict())
print('split_counts=', df.split.value_counts().to_dict())
assert len(schema['feature_columns']) == 709
assert set(df.label.unique()) <= {0, 1}
assert set(df.feature_dim.unique()) == {709}
PY

echo "[3/5] Fit train-only PCA compression and write compressed manifests"
run_py python -u -m engagement_daisee.openface.compress_openface_features \
  --manifest "$MANIFEST" \
  --output-root "$PROCESSED_ROOT/openface709_compressed" \
  --components "$COMPONENTS"

echo "[4/5] Train ML baselines: raw OpenFace copur + PCA tsfresh sweep"
RESULTS="$RUN_ROOT/results.jsonl"
: > "$RESULTS"
train_ml() {
  local name="$1" manifest="$2" feature_mode="$3"
  local out_dir="$RUN_ROOT/ml_$name"
  mkdir -p "$out_dir"
  local model_path="$out_dir/engagement_xgb.json"
  local summary_path="${model_path%.json}.summary.json"
  local eval_json="$out_dir/eval_test.json"
  echo "[ML] name=$name mode=$feature_mode manifest=$manifest"
  run_py python -u -m engagement_daisee.ml.train "${sample_args[@]}" \
    --manifest "$manifest" --output "$model_path" --run-id "${RUN_ID}_ml_$name" \
    --backend auto --feature-mode "$feature_mode" --cpu-workers "$ML_CPU_WORKERS" \
    --threshold-objective balanced_accuracy --oversample random
  if [[ ! -f "$model_path" ]]; then
    local alt="$WORKDIR/checkpoints/runs/trainml_${RUN_ID}_ml_$name/engagement_xgb.json"
    if [[ -f "$alt" ]]; then
      model_path="$alt"
      summary_path="${alt%.json}.summary.json"
    fi
  fi
  run_py python -u -m engagement_daisee.ml.evaluate \
    --manifest "$manifest" --model "$model_path" --summary-json "$summary_path" \
    --split test --feature-mode "$feature_mode" --aggregation video --output-json "$eval_json"
  NAME="$name" FEATURE_MODE="$feature_mode" MANIFEST_PATH="$manifest" SUMMARY_PATH="$summary_path" EVAL_PATH="$eval_json" run_py python - <<'PY' >> "$RESULTS"
import json, os
from pathlib import Path
train=Path(os.environ['SUMMARY_PATH']); evalp=Path(os.environ['EVAL_PATH'])
print(json.dumps({'kind':'ml','name':os.environ['NAME'],'feature_mode':os.environ['FEATURE_MODE'],'manifest':os.environ['MANIFEST_PATH'],'train':json.loads(train.read_text()) if train.exists() else {},'eval':json.loads(evalp.read_text()) if evalp.exists() else {}}))
PY
}
train_ml raw709_copur "$MANIFEST" copur
for dim in $ML_DIMS; do
  train_ml "pca${dim}_tsfresh" "$PROCESSED_ROOT/openface709_compressed/pca${dim}/feature_manifest.csv" tsfresh
done

echo "[5/5] Train RNN models on PCA dims"
train_rnn() {
  local dim="$1" model="$2"
  local manifest="$PROCESSED_ROOT/openface709_compressed/pca${dim}/feature_manifest.csv"
  local out_dir="$RUN_ROOT/rnn_pca${dim}_$model"
  mkdir -p "$out_dir"
  local ckpt="$out_dir/engagement_$model.pt"
  local eval_json="$out_dir/eval_test.json"
  echo "[RNN] dim=$dim model=$model manifest=$manifest"
  run_py python -u -m engagement_daisee.rnn.train "${sample_args[@]}" --amp \
    --manifest "$manifest" --output "$ckpt" --run-id "${RUN_ID}_rnn_pca${dim}_$model" \
    --model "$model" --device "$DEVICE" --cpu-threads "$CPU_THREADS" \
    --hidden-size "$RNN_HIDDEN_SIZE" --num-layers "$RNN_NUM_LAYERS" --dropout "$RNN_DROPOUT" \
    --batch-size "$RNN_BATCH_SIZE" --epochs "$RNN_EPOCHS" --patience "$RNN_PATIENCE" \
    --min-epochs "$RNN_MIN_EPOCHS" --threshold-objective balanced_accuracy --loss bce_weighted
  run_py python -u -m engagement_daisee.rnn.evaluate \
    --manifest "$manifest" --checkpoint "$ckpt" --split test --batch-size "$RNN_BATCH_SIZE" \
    --aggregation video --output-json "$eval_json"
  DIM="$dim" MODEL="$model" MANIFEST_PATH="$manifest" TRAIN_PATH="${ckpt%.pt}.json" EVAL_PATH="$eval_json" run_py python - <<'PY' >> "$RESULTS"
import json, os
from pathlib import Path
train=Path(os.environ['TRAIN_PATH']); evalp=Path(os.environ['EVAL_PATH'])
print(json.dumps({'kind':'rnn','dim':os.environ['DIM'],'model':os.environ['MODEL'],'manifest':os.environ['MANIFEST_PATH'],'train':json.loads(train.read_text()) if train.exists() else {},'eval':json.loads(evalp.read_text()) if evalp.exists() else {}}))
PY
}
for dim in $RNN_DIMS; do
  for model in $RNN_MODELS; do
    train_rnn "$dim" "$model"
  done
done

SUMMARY_OUT="$RUN_ROOT/openface709_summary.json" HISTORY_OUT="$WORKDIR/checkpoints/reports/openface709_history.jsonl" RESULTS="$RESULTS" RUN_ID="$RUN_ID" PROCESSED_ROOT="$PROCESSED_ROOT" RUN_ROOT="$RUN_ROOT" run_py python - <<'PY'
import json, os
from pathlib import Path
results=Path(os.environ['RESULTS'])
items=[json.loads(x) for x in results.read_text().splitlines() if x.strip()]
summary={'run_id':os.environ['RUN_ID'],'processed_root':os.environ['PROCESSED_ROOT'],'checkpoint_root':os.environ['RUN_ROOT'],'items':items}
out=Path(os.environ['SUMMARY_OUT'])
out.write_text(json.dumps(summary, indent=2))
hist=Path(os.environ['HISTORY_OUT'])
hist.parent.mkdir(parents=True, exist_ok=True)
with hist.open('a', encoding='utf-8') as f:
    f.write(json.dumps(summary) + '\n')
print('saved_summary=', out)
print('history=', hist)
PY

echo "=== OpenFace709 pipeline finished at $(date) ==="
