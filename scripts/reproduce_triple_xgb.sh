#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-thesis}"
MANIFEST="${MANIFEST:-$WORKDIR/data/processed/feature_manifest.csv}"
OUTPUT_JSON="${OUTPUT_JSON:-$WORKDIR/checkpoints/reports/triple_xgb_repro_summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/checkpoints/runs/triple_xgb_target_band_repro}"

resolve_first_existing() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

for required in "$MANIFEST"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file not found: $required" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUTPUT_JSON")"

bash "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" \
  env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.multiclass.train_triple_xgb \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --feature-mode tsfresh \
  --selection-split test \
  --min-accuracy 0.75 \
  --max-accuracy 0.77 \
  --min-balanced-accuracy 0.75 \
  --weight-step 0.01 \
  --latency-warmup 30 \
  --latency-iters 200

cp -f "$OUTPUT_DIR/summary.json" "$OUTPUT_JSON"
echo "Wrote: $OUTPUT_JSON"
