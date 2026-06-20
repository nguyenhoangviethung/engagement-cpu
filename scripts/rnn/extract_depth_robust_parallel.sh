#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${1:?Usage: extract_depth_robust_parallel.sh RUN_ID [WORKERS]}"
WORKERS="${2:-5}"
LABELS_CSV="$WORKDIR/data/processed/runs/daisee_4class_final_dataset/video_labels_4class.csv"
VIDEOS_DIR="$WORKDIR/data/raw/daisee/DAiSEE/DataSet"
RUN_DIR="$WORKDIR/data/processed/runs/extract_${RUN_ID}"
LOG_DIR="$WORKDIR/logs"
GLOBAL_LOG="$LOG_DIR/extract_${RUN_ID}.log"
FINAL_MANIFEST="$RUN_DIR/feature_manifest.csv"

if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer, got: $WORKERS" >&2
  exit 2
fi
if [[ ! -f "$LABELS_CSV" ]]; then
  echo "Missing labels CSV: $LABELS_CSV" >&2
  exit 1
fi
if [[ ! -d "$VIDEOS_DIR" ]]; then
  echo "Missing videos directory: $VIDEOS_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"
ln -sfn "$GLOBAL_LOG" "$LOG_DIR/latest_extract_depth_robust_parallel.log"
exec > >(tee -a "$GLOBAL_LOG") 2>&1

pids=()
stop_workers() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap stop_workers EXIT INT TERM HUP

echo "=== Parallel depth-robust extraction started at $(date) ==="
echo "Run ID: $RUN_ID"
echo "Workers: $WORKERS"
echo "Output: $RUN_DIR"

for ((worker=0; worker<WORKERS; worker++)); do
  worker_dir="$RUN_DIR/shard_${worker}"
  worker_log="$LOG_DIR/extract_${RUN_ID}_worker${worker}.log"
  mkdir -p "$worker_dir/features"

  (
    set -o pipefail
    PYTHONPATH="$WORKDIR/src" "$WORKDIR/scripts/lib/run_python.sh" \
      --env thesis \
      --workdir "$WORKDIR" \
      python -u -m engagement_daisee.rnn.extract_features \
      --labels "$LABELS_CSV" \
      --videos "$VIDEOS_DIR" \
      --features-dir "$worker_dir/features" \
      --manifest "$worker_dir/feature_manifest.csv" \
      --log-every 10 \
      --frame-stride 1 \
      --max-frames 0 \
      --resize-width 0 \
      --feature-set depth_robust_v2 \
      --temporal-enrichment velocity_std \
      --label-mode four_class \
      --num-shards "$WORKERS" \
      --shard-index "$worker" 2>&1 \
      | tee "$worker_log" \
      | sed -u "s/^/[worker-$worker] /"
  ) &
  pids+=("$!")
done

failed=0
for ((worker=0; worker<WORKERS; worker++)); do
  if ! wait "${pids[$worker]}"; then
    echo "Worker $worker failed; see $LOG_DIR/extract_${RUN_ID}_worker${worker}.log"
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "Extraction failed; manifests were not merged."
  exit 1
fi

manifests=()
for ((worker=0; worker<WORKERS; worker++)); do
  manifests+=("$RUN_DIR/shard_${worker}/feature_manifest.csv")
done
awk 'FNR == 1 && NR != 1 { next } { print }' "${manifests[@]}" > "$FINAL_MANIFEST"

echo "Merged manifest: $FINAL_MANIFEST"
echo "Manifest rows: $(( $(wc -l < "$FINAL_MANIFEST") - 1 ))"
echo "=== Parallel depth-robust extraction finished at $(date) ==="
