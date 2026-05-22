#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:?}"
CONDA_ENV="${CONDA_ENV:?}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:?}"
RNN_MANIFEST="${RNN_MANIFEST:?}"
ML_MANIFEST="${ML_MANIFEST:?}"
CNN_MANIFEST="${CNN_MANIFEST:?}"
RNN_MODELS="${RNN_MODELS:?}"
INCLUDE_ML="${INCLUDE_ML:?}"
INCLUDE_CNN="${INCLUDE_CNN:?}"
SAMPLE_MODE="${SAMPLE_MODE:?}"
DEVICE="${DEVICE:?}"
USE_AMP="${USE_AMP:?}"
RNN_CPU_THREADS="${RNN_CPU_THREADS:?}"
RNN_HIDDEN_SIZE="${RNN_HIDDEN_SIZE:-192}"
RNN_NUM_LAYERS="${RNN_NUM_LAYERS:-3}"
RNN_DROPOUT="${RNN_DROPOUT:-0.25}"
RNN_BATCH_SIZE="${RNN_BATCH_SIZE:-128}"
RNN_EPOCHS="${RNN_EPOCHS:-40}"
RNN_PATIENCE="${RNN_PATIENCE:-10}"
RNN_MIN_EPOCHS="${RNN_MIN_EPOCHS:-12}"
RNN_TCN_BLOCKS="${RNN_TCN_BLOCKS:-4}"
RNN_TCN_KERNEL_SIZE="${RNN_TCN_KERNEL_SIZE:-5}"
RNN_THRESHOLD_OBJECTIVE="${RNN_THRESHOLD_OBJECTIVE:-balanced_accuracy}"
RNN_LOSS="${RNN_LOSS:-bce_weighted}"
ML_CPU_WORKERS="${ML_CPU_WORKERS:?}"
CNN_MODEL="${CNN_MODEL:?}"
CNN_BATCH_SIZE="${CNN_BATCH_SIZE:?}"
CNN_EPOCHS="${CNN_EPOCHS:?}"
CNN_IMAGE_SIZE="${CNN_IMAGE_SIZE:?}"
RUN_ROOT="${RUN_ROOT:?}"
RUN_LOG="${RUN_LOG:?}"
SUMMARY_JSON="${SUMMARY_JSON:?}"
HISTORY_JSONL="${HISTORY_JSONL:?}"
LATEST_LOG_LINK="${LATEST_LOG_LINK:?}"
RUNS_CHECKPOINT_DIR="${RUNS_CHECKPOINT_DIR:?}"

mkdir -p "$RUN_ROOT"

echo "=== train_all started at $(date) ===" | tee -a "$RUN_LOG"
echo "run_root=$RUN_ROOT" | tee -a "$RUN_LOG"

sample_flag=""
[[ "$SAMPLE_MODE" == "1" ]] && sample_flag=" --sample"
amp_flag=""
[[ "$USE_AMP" == "1" ]] && amp_flag=" --amp"

localize_manifest() {
  local src_manifest="$1"
  local local_feature_root="$2"
  local out_manifest="$3"

  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" python - "$src_manifest" "$local_feature_root" "$out_manifest" <<'PY'
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
feat_root = Path(sys.argv[2])
out = Path(sys.argv[3])

if not src.exists():
    raise SystemExit(f"manifest not found: {src}")
out.parent.mkdir(parents=True, exist_ok=True)
with src.open(newline="", encoding="utf-8") as f, out.open("w", newline="", encoding="utf-8") as g:
    r = csv.DictReader(f)
    if not r.fieldnames or "feature_path" not in r.fieldnames:
        raise SystemExit("manifest must contain 'feature_path' column")
    w = csv.DictWriter(g, fieldnames=r.fieldnames)
    w.writeheader()
    for row in r:
        row["feature_path"] = str(feat_root / Path(row["feature_path"]).name)
        w.writerow(row)
print(str(out))
PY
}

ensure_local_manifest() {
  local src_manifest="$1"
  local feature_root="$2"
  local out_manifest="$3"

  [[ -f "$src_manifest" ]] || { echo "[FATAL] manifest not found: $src_manifest" | tee -a "$RUN_LOG" >&2; exit 1; }

  local first_feature
  first_feature="$($WORKDIR/scripts/lib/run_python.sh --env "$CONDA_ENV" --workdir "$WORKDIR" python - "$src_manifest" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
with manifest.open(newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    row = next(r, None)
    print((row or {}).get("feature_path", ""))
PY
)"

  if [[ -n "$first_feature" && -f "$first_feature" ]]; then
    echo "$src_manifest"
    return
  fi

  echo "[INFO] Localizing manifest paths: $src_manifest -> $out_manifest" | tee -a "$RUN_LOG" >&2
  localize_manifest "$src_manifest" "$feature_root" "$out_manifest" >/dev/null
  echo "$out_manifest"
}

RNN_MANIFEST_LOCAL="$(ensure_local_manifest "$RNN_MANIFEST" "$WORKDIR/data/processed/runs/pipeline_2/features" "$RUN_ROOT/rnn_manifest.local.csv")"
ML_MANIFEST_LOCAL="$(ensure_local_manifest "$ML_MANIFEST" "$WORKDIR/data/processed/runs/pipeline_2/features" "$RUN_ROOT/ml_manifest.local.csv")"
echo "rnn_manifest_local=$RNN_MANIFEST_LOCAL" | tee -a "$RUN_LOG"
echo "ml_manifest_local=$ML_MANIFEST_LOCAL" | tee -a "$RUN_LOG"

results_jsonl="$RUN_ROOT/results.jsonl"
: > "$results_jsonl"

if [[ "$INCLUDE_CNN" == "1" ]]; then
  [[ -f "$CNN_MANIFEST" ]] || { echo "[FATAL][CNN] manifest not found at $CNN_MANIFEST" | tee -a "$RUN_LOG"; exit 1; }
  rid="${RUN_ID_PREFIX}_cnn"
  out_dir="$RUN_ROOT/cnn"
  mkdir -p "$out_dir"
  ckpt="$out_dir/engagement_cnn.pt"
  eval_json="$out_dir/eval_test.json"
  agg_json="$out_dir/train_eval_summary.json"

  echo "[CNN] rid=$rid" | tee -a "$RUN_LOG"
  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.train$sample_flag \
    --manifest "$CNN_MANIFEST" --output "$ckpt" --run-id "$rid" --model "$CNN_MODEL" \
    --image-size "$CNN_IMAGE_SIZE" --batch-size "$CNN_BATCH_SIZE" --epochs "$CNN_EPOCHS" 2>&1 | tee -a "$RUN_LOG"

  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.evaluate \
    --manifest "$CNN_MANIFEST" --checkpoint "$ckpt" --split test --batch-size 256 --output-json "$eval_json" 2>&1 | tee -a "$RUN_LOG"

  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" python - "$out_dir/engagement_cnn.json" "$eval_json" "$agg_json" "$rid" "$CNN_MODEL" <<'PY' | tee -a "$RUN_LOG" >> "$results_jsonl"
import json, sys
from pathlib import Path
train_path, eval_path, agg_path, rid, model = [Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]]
train = json.loads(train_path.read_text()) if train_path.exists() else {}
evalp = json.loads(eval_path.read_text()) if eval_path.exists() else {}
payload = {"run_id": rid, "module": "cnn", "model": model, "train": train, "eval": evalp}
agg_path.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload))
PY
fi

for model in $RNN_MODELS; do
  rid="${RUN_ID_PREFIX}_rnn_${model}"
  out_dir="$RUN_ROOT/rnn_${model}"
  mkdir -p "$out_dir"
  ckpt="$out_dir/engagement_${model}.pt"
  eval_json="$out_dir/eval_test.json"
  agg_json="$out_dir/train_eval_summary.json"

  echo "[RNN] model=$model rid=$rid" | tee -a "$RUN_LOG"
  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.train$sample_flag$amp_flag \
    --manifest "$RNN_MANIFEST_LOCAL" --output "$ckpt" --run-id "$rid" --model "$model" --device "$DEVICE" \
    --cpu-threads "$RNN_CPU_THREADS" --hidden-size "$RNN_HIDDEN_SIZE" --num-layers "$RNN_NUM_LAYERS" \
    --dropout "$RNN_DROPOUT" --batch-size "$RNN_BATCH_SIZE" --epochs "$RNN_EPOCHS" \
    --patience "$RNN_PATIENCE" --min-epochs "$RNN_MIN_EPOCHS" --tcn-blocks "$RNN_TCN_BLOCKS" \
    --tcn-kernel-size "$RNN_TCN_KERNEL_SIZE" --threshold-objective "$RNN_THRESHOLD_OBJECTIVE" \
    --loss "$RNN_LOSS" 2>&1 | tee -a "$RUN_LOG"

  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.evaluate \
    --manifest "$RNN_MANIFEST_LOCAL" --checkpoint "$ckpt" --split test --batch-size "$RNN_BATCH_SIZE" --output-json "$eval_json" 2>&1 | tee -a "$RUN_LOG"

  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" python - "${ckpt%.pt}.json" "$eval_json" "$agg_json" "$rid" "$model" <<'PY' | tee -a "$RUN_LOG" >> "$results_jsonl"
import json, sys
from pathlib import Path
train_path, eval_path, agg_path, rid, model = [Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]]
train = json.loads(train_path.read_text()) if train_path.exists() else {}
evalp = json.loads(eval_path.read_text()) if eval_path.exists() else {}
payload = {"run_id": rid, "module": "rnn", "model": model, "train": train, "eval": evalp}
agg_path.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload))
PY
done

if [[ "$INCLUDE_ML" == "1" ]]; then
  rid="${RUN_ID_PREFIX}_ml"
  out_dir="$RUN_ROOT/ml"
  mkdir -p "$out_dir"
  model_path="$out_dir/engagement_xgb.json"
  eval_json="$out_dir/eval_test.json"
  agg_json="$out_dir/train_eval_summary.json"

  echo "[ML] rid=$rid" | tee -a "$RUN_LOG"
  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.train$sample_flag \
    --manifest "$ML_MANIFEST_LOCAL" --output "$model_path" --run-id "$rid" --backend auto --feature-mode tsfresh --cpu-workers "$ML_CPU_WORKERS" 2>&1 | tee -a "$RUN_LOG"

  model_path_effective="$model_path"
  runml_path="$RUNS_CHECKPOINT_DIR/trainml_${rid}/$(basename "$model_path")"
  [[ -f "$runml_path" ]] && model_path_effective="$runml_path"

  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.evaluate \
    --manifest "$ML_MANIFEST_LOCAL" --model "$model_path_effective" --split test --feature-mode tsfresh \
    --summary-json "$out_dir/engagement_xgb.summary.json" --output-json "$eval_json" 2>&1 | tee -a "$RUN_LOG"

  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" python - "$out_dir/engagement_xgb.summary.json" "$eval_json" "$agg_json" "$rid" <<'PY' | tee -a "$RUN_LOG" >> "$results_jsonl"
import json, sys
from pathlib import Path
train_path, eval_path, agg_path, rid = [Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]]
train = json.loads(train_path.read_text()) if train_path.exists() else {}
evalp = json.loads(eval_path.read_text()) if eval_path.exists() else {}
payload = {"run_id": rid, "module": "ml", "train": train, "eval": evalp}
agg_path.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload))
PY
fi

"$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" python - "$results_jsonl" "$SUMMARY_JSON" "$HISTORY_JSONL" "$RUN_ID_PREFIX" <<'PY' | tee -a "$RUN_LOG"
import json, sys
from pathlib import Path
results_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
history_path = Path(sys.argv[3])
prefix = sys.argv[4]
items = [json.loads(line.strip()) for line in results_path.read_text().splitlines() if line.strip()]
payload = {"run_id_prefix": prefix, "items": items}
summary_path.write_text(json.dumps(payload, indent=2))
with history_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=True) + "\n")
print(json.dumps({"saved": str(summary_path), "history": str(history_path), "count": len(items)}, indent=2))
PY

echo "=== train_all finished at $(date) ===" | tee -a "$RUN_LOG"
ln -sfn "$RUN_LOG" "$LATEST_LOG_LINK"
