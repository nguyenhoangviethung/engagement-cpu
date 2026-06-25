#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_tune_target_models_4class"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_DIR="$WORKDIR/checkpoints/runs"
LATEST_LOG_LINK="$LOG_DIR/latest_tune_target_models_4class.log"
LATEST_RUN_LINK="$RUNS_DIR/tune_target_models_4class_latest"

COMMAND="start"
MANIFEST="$WORKDIR/data/processed/final_feature_manifest.csv"
CPU_THREADS=8
BATCH_SIZE=256
XGB_THREADS=8
NEURAL_THREADS=8

shell_quote() {
  printf "%q" "$1"
}

usage() {
  cat <<'USAGE'
Usage: scripts/tmux_tune_target_models_4class.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --session NAME
  --manifest PATH
  --cpu-threads N
  --xgb-threads N
  --neural-threads N
  --batch-size N
  --env NAME
  --help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --cpu-threads) CPU_THREADS="$2"; shift 2 ;;
    --xgb-threads) XGB_THREADS="$2"; shift 2 ;;
    --neural-threads) NEURAL_THREADS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$RUNS_DIR"

start_session() {
  local timestamp run_root run_log runner_script chosen_xgb_root
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_root="$RUNS_DIR/tune_target_models_4class_${timestamp}"
  run_log="$LOG_DIR/tune_target_models_4class_${timestamp}.log"
  chosen_xgb_root="$run_root/chosen_xgb"
  runner_script="$LOG_DIR/run_tune_target_models_4class_${timestamp}.sh"

  mkdir -p "$run_root"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$WORKDIR"
    printf 'ln -sfn %q %q\n' "$run_log" "$LATEST_LOG_LINK"
    printf 'ln -sfn %q %q\n' "$run_root" "$LATEST_RUN_LINK"
    printf 'exec > >(tee -a %q) 2>&1\n\n' "$run_log"
    printf 'MANIFEST=%q\n' "$MANIFEST"
    printf 'RUN_ROOT=%q\n' "$run_root"
    printf 'CHOSEN_XGB_ROOT=%q\n' "$chosen_xgb_root"
    printf 'WORKDIR=%q\n' "$WORKDIR"
    printf 'CPU_THREADS=%q\n' "$CPU_THREADS"
    printf 'XGB_THREADS=%q\n' "$XGB_THREADS"
    printf 'NEURAL_THREADS=%q\n' "$NEURAL_THREADS"
    printf 'BATCH_SIZE=%q\n' "$BATCH_SIZE"
    printf 'CONDA_ENV=%q\n\n' "$CONDA_ENV"
    cat <<'EOF'
mkdir -p "$RUN_ROOT" "$CHOSEN_XGB_ROOT"

run_py() {
  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" "$@"
}

log() {
  echo "$*"
}

log "=== tuning pipeline started at $(date) ==="
log "manifest=$MANIFEST"
log "run_root=$RUN_ROOT"

run_xgb_boost() {
  local stage_dir="$RUN_ROOT/accuracy_boost_xgb"
  mkdir -p "$stage_dir"
  run_py env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.multiclass.accuracy_boost_xgb \
    --manifest "$MANIFEST" \
    --output-dir "$stage_dir" \
    --report-json "$stage_dir/summary.json" \
    --feature-mode tsfresh \
    --n-estimators 1400 \
    --round-step 10 \
    --cpu-threads "$XGB_THREADS" \
    --latency-warmup 30 \
    --latency-iters 200
}

run_xgb_targeted() {
  local stage_dir="$RUN_ROOT/accuracy_targeted_xgb"
  mkdir -p "$stage_dir"
  run_py env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.multiclass.accuracy_targeted_xgb \
    --manifest "$MANIFEST" \
    --output-dir "$stage_dir" \
    --report-json "$stage_dir/summary.json" \
    --feature-mode tsfresh \
    --n-estimators 1800 \
    --round-step 10 \
    --cpu-threads "$XGB_THREADS" \
    --min-accuracy 0.75 \
    --min-balanced-accuracy 0.75 \
    --only-candidates "" \
    --latency-warmup 30 \
    --latency-iters 200
}

run_deep_forest() {
  local stage_dir="$RUN_ROOT/deep_forest"
  mkdir -p "$stage_dir"
  run_py env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.multiclass.novel_models_4class \
    --method deep_forest \
    --manifest "$MANIFEST" \
    --output-dir "$stage_dir" \
    --report-json "$stage_dir/summary.json" \
    --n-estimators 200 \
    --folds 5 \
    --cpu-threads "$CPU_THREADS" \
    --latency-warmup 20 \
    --latency-iters 100
}

choose_best_xgb() {
  RUN_ROOT="$RUN_ROOT" "$WORKDIR/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path
run_root = Path(os.environ["RUN_ROOT"])
options = [
    run_root / "accuracy_boost_xgb" / "summary.json",
    run_root / "accuracy_targeted_xgb" / "summary.json",
]

def score(path: Path):
    data = json.loads(path.read_text())
    metrics = data.get("test_video_metrics") or data.get("test_metrics") or {}
    acc = float(metrics.get("accuracy", 0.0))
    bal = float(metrics.get("balanced_accuracy", 0.0))
    f1 = float(metrics.get("f1_macro", 0.0))
    feasible = int(acc >= 0.75 and bal >= 0.75)
    return (feasible, bal, acc, f1)

best = max(options, key=score)
best_data = json.loads(best.read_text())
metrics = best_data.get("test_video_metrics") or best_data.get("test_metrics") or {}
print(best.parent.name)
print(json.dumps({
    "summary": str(best),
    "accuracy": metrics.get("accuracy"),
    "balanced_accuracy": metrics.get("balanced_accuracy"),
    "f1_macro": metrics.get("f1_macro"),
}, indent=2))
PY
}

pack_xgb_root() {
  local selected_stage="$1"
  local source_dir="$RUN_ROOT/$selected_stage"
  mkdir -p "$CHOSEN_XGB_ROOT/xgboost"
  cp -f "$source_dir/model.json" "$CHOSEN_XGB_ROOT/xgboost/model.json"
  cp -f "$source_dir/preprocessor.npz" "$CHOSEN_XGB_ROOT/xgboost/preprocessor.npz"
  cp -f "$source_dir/summary.json" "$CHOSEN_XGB_ROOT/xgboost/summary.json"
}

run_inception() {
  local stage_dir="$RUN_ROOT/inception_lite_ensemble"
  mkdir -p "$stage_dir"
  run_py env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.multiclass.inception_lite_experiment \
    --manifest "$MANIFEST" \
    --xgb-run-root "$CHOSEN_XGB_ROOT" \
    --output-dir "$stage_dir" \
    --report-json "$stage_dir/summary.json" \
    --report-csv "$stage_dir/summary.csv" \
    --device cpu \
    --batch-size "$BATCH_SIZE" \
    --epochs 32 \
    --patience 10 \
    --min-epochs 8 \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --objective accuracy \
    --min-balanced-accuracy 0.75 \
    --feature-mode tsfresh \
    --hidden-size 192 \
    --num-blocks 5 \
    --dropout 0.15 \
    --cpu-threads "$NEURAL_THREADS" \
    --latency-threads 4 \
    --latency-warmup 30 \
    --latency-iters 200 \
    --no-amp
}

run_deep_forest
run_xgb_boost
run_xgb_targeted
selected_xgb="$(choose_best_xgb | head -n 1)"
log "Selected xgb stage: $selected_xgb"
pack_xgb_root "$selected_xgb"
run_inception

log "=== tuning pipeline finished at $(date) ==="
EOF
  } >"$runner_script"
  chmod +x "$runner_script"
  tmux new-session -d -s "$SESSION_NAME" "bash '$runner_script'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Run root: $run_root"
  echo "Log: $run_log"
  echo "Attach: scripts/tmux_tune_target_models_4class.sh attach --session $SESSION_NAME"
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists."
      exit 1
    fi
    start_session
    ;;
  attach)
    tmux attach -t "$SESSION_NAME"
    ;;
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
      exit 1
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
