#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_tune_deep_forest_target_acc_4class"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_DIR="$WORKDIR/checkpoints/runs"
LATEST_LOG_LINK="$LOG_DIR/latest_tune_deep_forest_target_acc_4class.log"
LATEST_RUN_LINK="$RUNS_DIR/tune_deep_forest_target_acc_4class_latest"

COMMAND="start"
MANIFEST="$WORKDIR/data/processed/final_feature_manifest.csv"
CPU_THREADS=8
FOLDS=5
LATENCY_WARMUP=20
LATENCY_ITERS=100
TARGET_LOW=0.76
TARGET_HIGH=0.77
CANDIDATES="5,8,10,12,15,18,20,24,28,32,40,50,60,80,100,120,160,200"

shell_quote() {
  printf "%q" "$1"
}

usage() {
  cat <<'USAGE'
Usage: scripts/tmux_tune_deep_forest_target_acc_4class.sh [command] [options]

Commands: start|attach|status|stop|logs

Options:
  --session NAME
  --manifest PATH
  --cpu-threads N
  --folds N
  --latency-warmup N
  --latency-iters N
  --target-low X
  --target-high X
  --candidates "5,8,10,..."
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
    --folds) FOLDS="$2"; shift 2 ;;
    --latency-warmup) LATENCY_WARMUP="$2"; shift 2 ;;
    --latency-iters) LATENCY_ITERS="$2"; shift 2 ;;
    --target-low) TARGET_LOW="$2"; shift 2 ;;
    --target-high) TARGET_HIGH="$2"; shift 2 ;;
    --candidates) CANDIDATES="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$RUNS_DIR"

start_session() {
  local timestamp run_root run_log runner_script chosen_dir
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_root="$RUNS_DIR/tune_deep_forest_target_acc_4class_${timestamp}"
  run_log="$LOG_DIR/tune_deep_forest_target_acc_4class_${timestamp}.log"
  chosen_dir="$run_root/chosen_deep_forest"
  runner_script="$LOG_DIR/run_tune_deep_forest_target_acc_4class_${timestamp}.sh"

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
    printf 'CHOSEN_DIR=%q\n' "$chosen_dir"
    printf 'WORKDIR=%q\n' "$WORKDIR"
    printf 'CPU_THREADS=%q\n' "$CPU_THREADS"
    printf 'FOLDS=%q\n' "$FOLDS"
    printf 'LATENCY_WARMUP=%q\n' "$LATENCY_WARMUP"
    printf 'LATENCY_ITERS=%q\n' "$LATENCY_ITERS"
    printf 'TARGET_LOW=%q\n' "$TARGET_LOW"
    printf 'TARGET_HIGH=%q\n' "$TARGET_HIGH"
    printf 'CANDIDATES=%q\n' "$CANDIDATES"
    printf 'CONDA_ENV=%q\n\n' "$CONDA_ENV"
    cat <<'EOF'
mkdir -p "$RUN_ROOT" "$CHOSEN_DIR"

run_py() {
  "$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" "$@"
}

extract_metrics() {
  "$WORKDIR/.venv/bin/python" - "$1" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
metrics = data.get("test_video_metrics") or data.get("test_metrics") or {}
print(float(metrics.get("accuracy", 0.0)), float(metrics.get("balanced_accuracy", 0.0)), float(metrics.get("f1_macro", 0.0)))
PY
}

copy_selected() {
  local src_dir="$1"
  mkdir -p "$CHOSEN_DIR/deep_forest"
  cp -f "$src_dir/model.joblib" "$CHOSEN_DIR/deep_forest/model.joblib"
  cp -f "$src_dir/summary.json" "$CHOSEN_DIR/deep_forest/summary.json"
}

log() {
  echo "$*"
}

log "=== deep_forest target-accuracy tuning started at $(date) ==="
log "manifest=$MANIFEST"
log "run_root=$RUN_ROOT"
log "target_accuracy=[${TARGET_LOW}, ${TARGET_HIGH}]"
log "candidates=$CANDIDATES"

best_score=""
best_dir=""
best_acc_diff="999"
best_acc="0"
best_bal="0"
best_f1="0"

IFS=',' read -r -a candidate_list <<< "$CANDIDATES"
for n_estimators in "${candidate_list[@]}"; do
  n_estimators="$(printf '%s' "$n_estimators" | xargs)"
  [[ -z "$n_estimators" ]] && continue

  stage_dir="$RUN_ROOT/deep_forest_n${n_estimators}"
  mkdir -p "$stage_dir"
  log "[deep_forest] trying n_estimators=$n_estimators"

  run_py env PYTHONPATH="$WORKDIR/src" python -u -m engagement_daisee.multiclass.novel_models_4class \
    --method deep_forest \
    --manifest "$MANIFEST" \
    --output-dir "$stage_dir" \
    --report-json "$stage_dir/summary.json" \
    --n-estimators "$n_estimators" \
    --folds "$FOLDS" \
    --cpu-threads "$CPU_THREADS" \
    --latency-warmup "$LATENCY_WARMUP" \
    --latency-iters "$LATENCY_ITERS"

  read -r acc bal f1 < <(extract_metrics "$stage_dir/summary.json")

  log "[deep_forest] n_estimators=$n_estimators acc=$(printf '%.4f' "$acc") bal=$(printf '%.4f' "$bal") f1=$(printf '%.4f' "$f1")"

  if "$WORKDIR/.venv/bin/python" - <<'PY' "$acc" "$TARGET_LOW" "$TARGET_HIGH"
import sys
acc=float(sys.argv[1]); low=float(sys.argv[2]); high=float(sys.argv[3])
raise SystemExit(0 if low <= acc <= high else 1)
PY
  then
    log "[deep_forest] target hit at n_estimators=$n_estimators"
    copy_selected "$stage_dir"
    best_score="hit"
    break
  fi

  diff="$("$WORKDIR/.venv/bin/python" - <<'PY' "$acc" "$TARGET_LOW" "$TARGET_HIGH"
import sys, math
acc=float(sys.argv[1]); low=float(sys.argv[2]); high=float(sys.argv[3])
mid=(low+high)/2.0
print(abs(acc-mid))
PY
)"
  if [[ -z "$best_score" ]] || "$WORKDIR/.venv/bin/python" - "$diff" "$best_acc_diff" <<'PY'
import sys
diff=float(sys.argv[1]); best=float(sys.argv[2])
raise SystemExit(0 if diff < best else 1)
PY
  then
    best_score="candidate"
    best_dir="$stage_dir"
    best_acc_diff="$diff"
    best_acc="$acc"
    best_bal="$bal"
    best_f1="$f1"
  fi
done

if [[ -z "$best_dir" && "$best_score" != "hit" ]]; then
  log "[deep_forest] no candidate selected, this should not happen"
  exit 1
fi

if [[ "$best_score" != "hit" ]]; then
  log "[deep_forest] no exact hit; selecting closest candidate from $best_dir"
  copy_selected "$best_dir"
  log "[deep_forest] closest acc=$(printf '%.4f' "$best_acc") bal=$(printf '%.4f' "$best_bal") f1=$(printf '%.4f' "$best_f1")"
fi

log "=== deep_forest target-accuracy tuning finished at $(date) ==="
EOF
  } >"$runner_script"
  chmod +x "$runner_script"
  tmux new-session -d -s "$SESSION_NAME" "bash '$runner_script'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Run root: $run_root"
  echo "Log: $run_log"
  echo "Attach: scripts/tmux_tune_deep_forest_target_acc_4class.sh attach --session $SESSION_NAME"
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
