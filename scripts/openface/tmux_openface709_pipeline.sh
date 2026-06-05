#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-thesis}"
SESSION="${SESSION:-openface709_train}"
RUN_ID="${RUN_ID:-openface709_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
COMPONENTS="${COMPONENTS:-128,192,256,300,384}"
RNN_DIMS="${RNN_DIMS:-128 192 256 300 384}"
ML_DIMS="${ML_DIMS:-96 128 160 192 224 256 300 384}"
RNN_MODELS="${RNN_MODELS:-gru_basic tcn tiny_transformer bilstm simple_gru}"
MAX_FRAMES="${MAX_FRAMES:-120}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-30}"
RNN_EPOCHS="${RNN_EPOCHS:-60}"
RNN_PATIENCE="${RNN_PATIENCE:-12}"
RNN_MIN_EPOCHS="${RNN_MIN_EPOCHS:-12}"
RNN_HIDDEN_SIZE="${RNN_HIDDEN_SIZE:-192}"
RNN_NUM_LAYERS="${RNN_NUM_LAYERS:-3}"
RNN_DROPOUT="${RNN_DROPOUT:-0.20}"
RNN_BATCH_SIZE="${RNN_BATCH_SIZE:-128}"
CPU_THREADS="${CPU_THREADS:-2}"
ML_CPU_WORKERS="${ML_CPU_WORKERS:-2}"
SAMPLE="${SAMPLE:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
BASE_MANIFEST="${BASE_MANIFEST:-}"
BASE_SCHEMA="${BASE_SCHEMA:-}"
ML_THRESHOLD_OBJECTIVE="${ML_THRESHOLD_OBJECTIVE:-accuracy}"
RNN_THRESHOLD_OBJECTIVE="${RNN_THRESHOLD_OBJECTIVE:-accuracy}"
RNN_LOSS="${RNN_LOSS:-focal}"
RNN_SAMPLER="${RNN_SAMPLER:-weighted}"

RUN_ROOT="$WORKDIR/checkpoints/runs/$RUN_ID"
PROCESSED_ROOT="$WORKDIR/data/processed/runs/$RUN_ID"
LOG="$WORKDIR/logs/${RUN_ID}.log"
RUNNER="$WORKDIR/logs/${RUN_ID}.runner.sh"
LATEST="$WORKDIR/logs/openface709_latest.log"

usage() {
  cat <<USAGE
Usage: $0 {start|status|attach|stop}

Environment overrides:
  SESSION=$SESSION
  RUN_ID=$RUN_ID
  DEVICE=$DEVICE
  COMPONENTS=$COMPONENTS
  RNN_DIMS="$RNN_DIMS"
  ML_DIMS="$ML_DIMS"
  RNN_MODELS="$RNN_MODELS"
  SAMPLE=$SAMPLE
  SKIP_EXTRACT=$SKIP_EXTRACT
  BASE_MANIFEST=$BASE_MANIFEST
  BASE_SCHEMA=$BASE_SCHEMA
  ML_THRESHOLD_OBJECTIVE=$ML_THRESHOLD_OBJECTIVE
  RNN_THRESHOLD_OBJECTIVE=$RNN_THRESHOLD_OBJECTIVE
  RNN_LOSS=$RNN_LOSS
  RNN_SAMPLER=$RNN_SAMPLER
USAGE
}

cmd="${1:-}"
case "$cmd" in
  start)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "Session already exists: $SESSION"
      echo "Attach: tmux attach -t $SESSION"
      exit 1
    fi
    mkdir -p "$RUN_ROOT" "$PROCESSED_ROOT" "$WORKDIR/logs"
    cat > "$RUNNER" <<'EOF'
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
echo "skip_extract=$SKIP_EXTRACT | base_manifest=${BASE_MANIFEST:-<none>}"
echo "ml_threshold_objective=$ML_THRESHOLD_OBJECTIVE | rnn_threshold_objective=$RNN_THRESHOLD_OBJECTIVE | rnn_loss=$RNN_LOSS"

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

if [[ "$SKIP_EXTRACT" == "1" ]]; then
  echo "[2/5] Skip extraction and use existing OpenFace709 manifest"
  if [[ -z "${BASE_MANIFEST:-}" ]]; then
    echo "[error] SKIP_EXTRACT=1 requires BASE_MANIFEST"
    exit 1
  fi
  MANIFEST="$BASE_MANIFEST"
  if [[ -n "${BASE_SCHEMA:-}" ]]; then
    SCHEMA="$BASE_SCHEMA"
  else
    SCHEMA="$(dirname "$MANIFEST")/openface709_schema.json"
  fi
else
  echo "[2/5] Extract exact OpenFace 709-D features"
  run_py python -u -m engagement_daisee.openface.extract_openface709 "${sample_args[@]}" \
    --output-root "$PROCESSED_ROOT/openface709" \
    --max-frames "$MAX_FRAMES" \
    --sequence-length "$SEQUENCE_LENGTH" \
    --save-every 20 \
    --log-every 20
  MANIFEST="$PROCESSED_ROOT/openface709/feature_manifest.csv"
  SCHEMA="$PROCESSED_ROOT/openface709/openface709_schema.json"
fi
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
    --threshold-objective "$ML_THRESHOLD_OBJECTIVE" --oversample random
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
    --min-epochs "$RNN_MIN_EPOCHS" --threshold-objective "$RNN_THRESHOLD_OBJECTIVE" \
    --loss "$RNN_LOSS" --train-sampler "$RNN_SAMPLER"
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
EOF
    chmod +x "$RUNNER"
    tmux_cmd=(
      env
      "WORKDIR=$WORKDIR" "CONDA_ENV=$CONDA_ENV" "RUN_ID=$RUN_ID" "RUN_ROOT=$RUN_ROOT" "PROCESSED_ROOT=$PROCESSED_ROOT"
      "LOG=$LOG" "LATEST=$LATEST" "DEVICE=$DEVICE" "COMPONENTS=$COMPONENTS" "RNN_DIMS=$RNN_DIMS" "ML_DIMS=$ML_DIMS"
      "RNN_MODELS=$RNN_MODELS" "MAX_FRAMES=$MAX_FRAMES" "SEQUENCE_LENGTH=$SEQUENCE_LENGTH" "RNN_EPOCHS=$RNN_EPOCHS"
      "RNN_PATIENCE=$RNN_PATIENCE" "RNN_MIN_EPOCHS=$RNN_MIN_EPOCHS" "RNN_HIDDEN_SIZE=$RNN_HIDDEN_SIZE"
      "RNN_NUM_LAYERS=$RNN_NUM_LAYERS" "RNN_DROPOUT=$RNN_DROPOUT" "RNN_BATCH_SIZE=$RNN_BATCH_SIZE"
      "CPU_THREADS=$CPU_THREADS" "ML_CPU_WORKERS=$ML_CPU_WORKERS" "SAMPLE=$SAMPLE"
      "SKIP_EXTRACT=$SKIP_EXTRACT" "BASE_MANIFEST=$BASE_MANIFEST" "BASE_SCHEMA=$BASE_SCHEMA"
      "ML_THRESHOLD_OBJECTIVE=$ML_THRESHOLD_OBJECTIVE" "RNN_THRESHOLD_OBJECTIVE=$RNN_THRESHOLD_OBJECTIVE"
      "RNN_LOSS=$RNN_LOSS" "RNN_SAMPLER=$RNN_SAMPLER"
      bash "$RUNNER"
    )
    tmux new-session -d -s "$SESSION" "$(printf '%q ' "${tmux_cmd[@]}")"
    # Auto-open a monitor window so attaching always shows live logs.
    tmux new-window -t "$SESSION" -n monitor "cd '$WORKDIR' && tail -n 120 -f '$LOG'"
    tmux select-window -t "$SESSION:monitor"
    tmux setw -t "$SESSION" remain-on-exit on
    echo "Started tmux session: $SESSION"
    echo "Run id: $RUN_ID"
    echo "Attach: tmux attach -t $SESSION"
    echo "Watch log: tail -f $LOG"
    echo "Latest log: tail -f $LATEST"
    ;;
  status)
    tmux list-sessions 2>/dev/null | grep -F "$SESSION" || true
    echo "Latest log: $LATEST"
    [[ -f "$LATEST" ]] && tail -80 "$LATEST" || true
    ;;
  attach)
    exec tmux attach -t "$SESSION"
    ;;
  stop)
    tmux kill-session -t "$SESSION"
    echo "Stopped: $SESSION"
    ;;
  *)
    usage
    exit 1
    ;;
esac
