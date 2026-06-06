#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:?}"
CONDA_ENV="${CONDA_ENV:?}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:?}"
RNN_MANIFEST="${RNN_MANIFEST:?}"
ML_MANIFEST="${ML_MANIFEST:?}"
CNN_MANIFEST="${CNN_MANIFEST:-}"
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
RNN_LEARNING_RATE="${RNN_LEARNING_RATE:-3e-4}"
RNN_WEIGHT_DECAY="${RNN_WEIGHT_DECAY:-1e-4}"
RNN_SCHEDULER="${RNN_SCHEDULER:-plateau}"
RNN_FREEZE_FEATURE_EPOCHS="${RNN_FREEZE_FEATURE_EPOCHS:-0}"
RNN_TCN_BLOCKS="${RNN_TCN_BLOCKS:-4}"
RNN_TCN_KERNEL_SIZE="${RNN_TCN_KERNEL_SIZE:-5}"
RNN_THRESHOLD_OBJECTIVE="${RNN_THRESHOLD_OBJECTIVE:-balanced_accuracy}"
RNN_LOSS="${RNN_LOSS:-bce_weighted}"
ML_CPU_WORKERS="${ML_CPU_WORKERS:?}"
ML_THRESHOLD_OBJECTIVE="${ML_THRESHOLD_OBJECTIVE:-accuracy}"
ML_DIM_REDUCTION="${ML_DIM_REDUCTION:-none}"
ML_DIM_COMPONENTS="${ML_DIM_COMPONENTS:-128}"
ML_OVERSAMPLE="${ML_OVERSAMPLE:-none}"
ML_FEATURE_MODE="${ML_FEATURE_MODE:-tsfresh}"
CNN_MODEL="${CNN_MODEL:-mobilenet_v3_small}"
CNN_BATCH_SIZE="${CNN_BATCH_SIZE:-64}"
CNN_EPOCHS="${CNN_EPOCHS:-12}"
CNN_IMAGE_SIZE="${CNN_IMAGE_SIZE:-112}"
CNN_FRAMES_PER_VIDEO="${CNN_FRAMES_PER_VIDEO:-8}"
CNN_PRETRAINED="${CNN_PRETRAINED:-1}"
CNN_FREEZE_BACKBONE="${CNN_FREEZE_BACKBONE:-0}"
CNN_FORCE_EXTRACT="${CNN_FORCE_EXTRACT:-0}"
CNN_LEARNING_RATE="${CNN_LEARNING_RATE:-3e-4}"
CNN_WEIGHT_DECAY="${CNN_WEIGHT_DECAY:-1e-4}"
CNN_PATIENCE="${CNN_PATIENCE:-6}"
CNN_TRAIN_SAMPLER="${CNN_TRAIN_SAMPLER:-weighted}"
CNN_THRESHOLD_OBJECTIVE="${CNN_THRESHOLD_OBJECTIVE:-accuracy}"
EVAL_AGGREGATION="${EVAL_AGGREGATION:-video}"
RUN_ROOT="${RUN_ROOT:?}"
RUN_LOG="${RUN_LOG:?}"
SUMMARY_JSON="${SUMMARY_JSON:?}"
HISTORY_JSONL="${HISTORY_JSONL:?}"
LATEST_LOG_LINK="${LATEST_LOG_LINK:?}"
RUNS_CHECKPOINT_DIR="${RUNS_CHECKPOINT_DIR:?}"
RUN_PROCESSED_ROOT="${RUN_PROCESSED_ROOT:-$WORKDIR/data/processed/runs/train_all_$RUN_ID_PREFIX}"

mkdir -p "$RUN_ROOT"
mkdir -p "$RUN_PROCESSED_ROOT"

echo "=== train_all started at $(date) ===" | tee -a "$RUN_LOG"
echo "run_root=$RUN_ROOT" | tee -a "$RUN_LOG"

sample_flag=""
[[ "$SAMPLE_MODE" == "1" ]] && sample_flag=" --sample"
amp_flag=""
[[ "$USE_AMP" == "1" ]] && amp_flag=" --amp"
cnn_pretrained_flag=""
[[ "$CNN_PRETRAINED" == "1" ]] && cnn_pretrained_flag=" --pretrained"
cnn_freeze_flag=""
[[ "$CNN_FREEZE_BACKBONE" == "1" ]] && cnn_freeze_flag=" --freeze-backbone"

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

RNN_MANIFEST_LOCAL="$(ensure_local_manifest "$RNN_MANIFEST" "$WORKDIR/data/processed/runs/baseline_pipeline_features/features" "$RUN_ROOT/rnn_manifest.local.csv")"
ML_MANIFEST_LOCAL="$(ensure_local_manifest "$ML_MANIFEST" "$WORKDIR/data/processed/runs/baseline_pipeline_features/features" "$RUN_ROOT/ml_manifest.local.csv")"
echo "rnn_manifest_local=$RNN_MANIFEST_LOCAL" | tee -a "$RUN_LOG"
echo "ml_manifest_local=$ML_MANIFEST_LOCAL" | tee -a "$RUN_LOG"

results_jsonl="$RUN_ROOT/results.jsonl"
: > "$results_jsonl"

if [[ "$INCLUDE_CNN" == "1" ]]; then
  CNN_MANIFEST_EFFECTIVE="$CNN_MANIFEST"
  if [[ "$CNN_FORCE_EXTRACT" != "1" && ! -f "$CNN_MANIFEST_EFFECTIVE" ]]; then
    latest_existing_cnn_manifest="$(
      find "$WORKDIR/data/processed/runs" -maxdepth 2 -name "cnn_frame_manifest.csv" -type f -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"
    if [[ -n "$latest_existing_cnn_manifest" && -f "$latest_existing_cnn_manifest" ]]; then
      CNN_MANIFEST_EFFECTIVE="$latest_existing_cnn_manifest"
      echo "[CNN] manifest not found at $CNN_MANIFEST; reusing existing frame manifest: $CNN_MANIFEST_EFFECTIVE" | tee -a "$RUN_LOG"
    fi
  fi
  if [[ "$CNN_FORCE_EXTRACT" == "1" || ! -f "$CNN_MANIFEST_EFFECTIVE" ]]; then
    CNN_MANIFEST_EFFECTIVE="$RUN_PROCESSED_ROOT/cnn_frame_manifest.csv"
    cnn_frames_dir="$RUN_PROCESSED_ROOT/cnn_frames"
    echo "[CNN] manifest not found; generating frame manifest at $CNN_MANIFEST_EFFECTIVE" | tee -a "$RUN_LOG"
    "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.preprocess_labels 2>&1 | tee -a "$RUN_LOG"
    "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.cnn.extract_frames$sample_flag \
      --output-dir "$cnn_frames_dir" --manifest "$CNN_MANIFEST_EFFECTIVE" \
      --frame-size "$CNN_IMAGE_SIZE" --frames-per-video "$CNN_FRAMES_PER_VIDEO" 2>&1 | tee -a "$RUN_LOG"
  fi

  [[ -f "$CNN_MANIFEST_EFFECTIVE" ]] || { echo "[FATAL][CNN] manifest not found at $CNN_MANIFEST_EFFECTIVE" | tee -a "$RUN_LOG"; exit 1; }
  rid="${RUN_ID_PREFIX}_cnn"
  out_dir="$RUN_ROOT/cnn"
  mkdir -p "$out_dir"
  ckpt="$out_dir/engagement_cnn.pt"
  eval_json="$out_dir/eval_test.json"
  agg_json="$out_dir/train_eval_summary.json"

  echo "[CNN] rid=$rid" | tee -a "$RUN_LOG"
  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.train$sample_flag \
    --manifest "$CNN_MANIFEST_EFFECTIVE" --output "$ckpt" --run-id "$rid" --model "$CNN_MODEL" \
    --image-size "$CNN_IMAGE_SIZE" --batch-size "$CNN_BATCH_SIZE" --epochs "$CNN_EPOCHS" --device "$DEVICE" \
    --lr "$CNN_LEARNING_RATE" --weight-decay "$CNN_WEIGHT_DECAY" --patience "$CNN_PATIENCE" \
    --train-sampler "$CNN_TRAIN_SAMPLER" --threshold-objective "$CNN_THRESHOLD_OBJECTIVE"$cnn_pretrained_flag$cnn_freeze_flag 2>&1 | tee -a "$RUN_LOG"

  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.cnn.evaluate \
    --manifest "$CNN_MANIFEST_EFFECTIVE" --checkpoint "$ckpt" --split test --batch-size 256 --device "$DEVICE" \
    --aggregation "$EVAL_AGGREGATION" --output-json "$eval_json" 2>&1 | tee -a "$RUN_LOG"

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
    --patience "$RNN_PATIENCE" --min-epochs "$RNN_MIN_EPOCHS" --lr "$RNN_LEARNING_RATE" \
    --weight-decay "$RNN_WEIGHT_DECAY" --scheduler "$RNN_SCHEDULER" \
    --freeze-feature-epochs "$RNN_FREEZE_FEATURE_EPOCHS" --tcn-blocks "$RNN_TCN_BLOCKS" \
    --tcn-kernel-size "$RNN_TCN_KERNEL_SIZE" --threshold-objective "$RNN_THRESHOLD_OBJECTIVE" \
    --loss "$RNN_LOSS" 2>&1 | tee -a "$RUN_LOG"

  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.rnn.evaluate \
    --manifest "$RNN_MANIFEST_LOCAL" --checkpoint "$ckpt" --split test --batch-size "$RNN_BATCH_SIZE" \
    --aggregation "$EVAL_AGGREGATION" --output-json "$eval_json" 2>&1 | tee -a "$RUN_LOG"

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
    --manifest "$ML_MANIFEST_LOCAL" --output "$model_path" --run-id "$rid" --backend auto --feature-mode "$ML_FEATURE_MODE" \
    --cpu-workers "$ML_CPU_WORKERS" --threshold-objective "$ML_THRESHOLD_OBJECTIVE" \
    --dim-reduction "$ML_DIM_REDUCTION" --dim-components "$ML_DIM_COMPONENTS" --oversample "$ML_OVERSAMPLE" 2>&1 | tee -a "$RUN_LOG"

  model_path_effective="$model_path"
  summary_path_effective="$out_dir/engagement_xgb.summary.json"
  runml_path="$RUNS_CHECKPOINT_DIR/trainml_${rid}/$(basename "$model_path")"
  runml_summary_path="${runml_path%.json}.summary.json"
  if [[ -f "$runml_path" ]]; then
    model_path_effective="$runml_path"
  fi
  if [[ -f "$runml_summary_path" ]]; then
    summary_path_effective="$runml_summary_path"
  fi

  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" env PYTHONPATH="$WORKDIR/src" python -m engagement_daisee.ml.evaluate \
    --manifest "$ML_MANIFEST_LOCAL" --model "$model_path_effective" --split test --feature-mode "$ML_FEATURE_MODE" \
    --summary-json "$summary_path_effective" --aggregation "$EVAL_AGGREGATION" --output-json "$eval_json" 2>&1 | tee -a "$RUN_LOG"

  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" python - "$summary_path_effective" "$eval_json" "$agg_json" "$rid" <<'PY' | tee -a "$RUN_LOG" >> "$results_jsonl"
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
