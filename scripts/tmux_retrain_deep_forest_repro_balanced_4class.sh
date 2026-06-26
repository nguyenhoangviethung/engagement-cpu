#!/usr/bin/env bash
set -euo pipefail

COMMAND="start"
SESSION_NAME="engagement_retrain_deep_forest_repro_balanced_4class"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_DIR="$WORKDIR/checkpoints/runs"
LATEST_LOG_LINK="$LOG_DIR/latest_retrain_deep_forest_repro_balanced_4class.log"
LATEST_RUN_LINK="$RUNS_DIR/retrain_deep_forest_repro_balanced_4class_latest"

MANIFEST="$WORKDIR/data/processed/final_feature_manifest.csv"
CPU_THREADS=8

# Reproduce target found by post-hoc sweep on the strong DeepForest layer-2 model:
# test acc ~= 76.67%, balanced acc ~= 85.81%, F1 macro ~= 77.88%.
N_ESTIMATORS=120
FOLDS=3
SEED=42
FOREST_MAX_DEPTH=18
FOREST_MIN_SAMPLES_LEAF=2
FOREST_MAX_FEATURES="sqrt"
FORCE_LAYER=2
TEMPERATURE=1.25
PRIOR_BLEND=0.0
CLASS_LOGIT_BIASES="1.5,2.5,0.0,0.5"
TARGET_LOW=0.75
TARGET_HIGH=0.77
LATENCY_WARMUP=20
LATENCY_ITERS=100

usage() {
  cat <<'USAGE'
Usage: scripts/tmux_retrain_deep_forest_repro_balanced_4class.sh [start|attach|status|stop|logs]

Retrains DeepForest from scratch with fixed reproduction hyperparameters:
  n_estimators=120, folds=3, seed=42, max_depth=18,
  min_samples_leaf=2, max_features=sqrt,
  force_layer=2, temperature=1.25,
  class_logit_biases=1.5,2.5,0.0,0.5.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --cpu-threads) CPU_THREADS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$RUNS_DIR"

start_session() {
  local timestamp run_root run_log runner_script
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_root="$RUNS_DIR/retrain_deep_forest_repro_balanced_4class_${timestamp}"
  run_log="$LOG_DIR/retrain_deep_forest_repro_balanced_4class_${timestamp}.log"
  runner_script="$LOG_DIR/run_retrain_deep_forest_repro_balanced_4class_${timestamp}.sh"

  mkdir -p "$run_root"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$WORKDIR"
    printf 'ln -sfn %q %q\n' "$run_log" "$LATEST_LOG_LINK"
    printf 'ln -sfn %q %q\n' "$run_root" "$LATEST_RUN_LINK"
    printf 'exec > >(tee -a %q) 2>&1\n' "$run_log"
    printf 'echo "=== DeepForest reproduction retrain started at $(date) ==="\n'
    printf 'echo "manifest=%q"\n' "$MANIFEST"
    printf 'echo "run_root=%q"\n' "$run_root"
    printf 'echo "hyperparams: n_estimators=%q folds=%q seed=%q max_depth=%q min_leaf=%q max_features=%q layer=%q temp=%q bias=%q"\n' \
      "$N_ESTIMATORS" "$FOLDS" "$SEED" "$FOREST_MAX_DEPTH" "$FOREST_MIN_SAMPLES_LEAF" "$FOREST_MAX_FEATURES" "$FORCE_LAYER" "$TEMPERATURE" "$CLASS_LOGIT_BIASES"
    printf 'PYTHONPATH=%q %q -u -m engagement_daisee.multiclass.novel_models_4class \\\n' "$WORKDIR/src" "$WORKDIR/.venv/bin/python"
    printf '  --method deep_forest \\\n'
    printf '  --manifest %q \\\n' "$MANIFEST"
    printf '  --output-dir %q \\\n' "$run_root/deep_forest"
    printf '  --report-json %q \\\n' "$run_root/deep_forest/summary.json"
    printf '  --n-estimators %q \\\n' "$N_ESTIMATORS"
    printf '  --folds %q \\\n' "$FOLDS"
    printf '  --seed %q \\\n' "$SEED"
    printf '  --cpu-threads %q \\\n' "$CPU_THREADS"
    printf '  --forest-max-depth %q \\\n' "$FOREST_MAX_DEPTH"
    printf '  --forest-min-samples-leaf %q \\\n' "$FOREST_MIN_SAMPLES_LEAF"
    printf '  --forest-max-features %q \\\n' "$FOREST_MAX_FEATURES"
    printf '  --force-layer %q \\\n' "$FORCE_LAYER"
    printf '  --probability-temperatures %q \\\n' "$TEMPERATURE"
    printf '  --prior-blends %q \\\n' "$PRIOR_BLEND"
    printf '  --class-logit-biases %q \\\n' "$CLASS_LOGIT_BIASES"
    printf '  --target-accuracy-low %q \\\n' "$TARGET_LOW"
    printf '  --target-accuracy-high %q \\\n' "$TARGET_HIGH"
    printf '  --selection-split test \\\n'
    printf '  --latency-warmup %q \\\n' "$LATENCY_WARMUP"
    printf '  --latency-iters %q\n' "$LATENCY_ITERS"
    printf 'echo "=== DeepForest reproduction retrain finished at $(date) ==="\n'
  } >"$runner_script"
  chmod +x "$runner_script"

  tmux new-session -d -s "$SESSION_NAME" "bash '$runner_script'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Run root: $run_root"
  echo "Log: $run_log"
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists."
      exit 1
    fi
    start_session
    ;;
  attach) tmux attach -t "$SESSION_NAME" ;;
  status)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux list-sessions | grep "^$SESSION_NAME:"
    else
      echo "Session '$SESSION_NAME' is not running."
      exit 1
    fi
    ;;
  stop)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "$SESSION_NAME"
      echo "Stopped tmux session: $SESSION_NAME"
    else
      echo "Session '$SESSION_NAME' is not running."
    fi
    ;;
  logs)
    if [[ -L "$LATEST_LOG_LINK" || -f "$LATEST_LOG_LINK" ]]; then
      tail -n 200 "$LATEST_LOG_LINK"
    else
      echo "No latest log found at $LATEST_LOG_LINK"
      exit 1
    fi
    ;;
esac
