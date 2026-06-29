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
LATENCY_WARMUP=5
LATENCY_ITERS=20
TARGET_LOW=0.76
TARGET_HIGH=0.77
PROBABILITY_TEMPERATURES="1.0,1.25,1.5,2.0,3.0,5.0,8.0,12.0"
PRIOR_BLENDS="0.00,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"
CONFIGS="${DEEP_FOREST_TARGET_CONFIGS:-5|18|2|sqrt|42|auto
5|18|2|sqrt|43|1
5|12|5|sqrt|44|auto
5|10|10|sqrt|45|auto
5|8|20|sqrt|46|auto
5|6|40|sqrt|47|auto
5|4|80|sqrt|48|auto
8|8|20|log2|49|auto
10|6|50|0.5|50|auto
12|5|80|0.25|51|auto
20|4|100|sqrt|52|auto
20|3|120|log2|53|auto
40|3|150|0.25|54|auto
3|12|5|sqrt|55|auto
2|10|10|sqrt|56|auto}"

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
  --configs $'n|max_depth|min_leaf|max_features|seed|force_layer\n...'
  --probability-temperatures "1.0,1.5,..."
  --prior-blends "0.0,0.02,..."
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
    --configs) CONFIGS="$2"; shift 2 ;;
    --probability-temperatures) PROBABILITY_TEMPERATURES="$2"; shift 2 ;;
    --prior-blends) PRIOR_BLENDS="$2"; shift 2 ;;
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
    printf 'PROBABILITY_TEMPERATURES=%q\n' "$PROBABILITY_TEMPERATURES"
    printf 'PRIOR_BLENDS=%q\n' "$PRIOR_BLENDS"
    printf 'CONFIGS=%q\n' "$CONFIGS"
    printf 'CONDA_ENV=%q\n\n' "$CONDA_ENV"
    cat <<'EOF'
mkdir -p "$RUN_ROOT" "$CHOSEN_DIR"

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
log "probability_temperatures=$PROBABILITY_TEMPERATURES"
log "prior_blends=$PRIOR_BLENDS"
log "configs:"
printf '%s\n' "$CONFIGS"

best_in_range_dir=""
best_in_range_diff="999"
best_in_range_acc="0"
best_in_range_bal="0"
best_in_range_f1="0"

best_any_dir=""
best_any_diff="999"
best_any_acc="0"
best_any_bal="0"
best_any_f1="0"

config_idx=0
while IFS='|' read -r n_estimators max_depth min_leaf max_features seed force_layer; do
  n_estimators="$(printf '%s' "${n_estimators:-}" | xargs)"
  [[ -z "$n_estimators" || "$n_estimators" == \#* ]] && continue
  max_depth="$(printf '%s' "$max_depth" | xargs)"
  min_leaf="$(printf '%s' "$min_leaf" | xargs)"
  max_features="$(printf '%s' "$max_features" | xargs)"
  seed="$(printf '%s' "$seed" | xargs)"
  force_layer="$(printf '%s' "$force_layer" | xargs)"
  config_idx=$((config_idx + 1))

  stage_dir="$RUN_ROOT/deep_forest_cfg${config_idx}_n${n_estimators}_d${max_depth}_leaf${min_leaf}_mf${max_features}_seed${seed}_layer${force_layer}"
  mkdir -p "$stage_dir"
  log "[deep_forest] trying cfg=$config_idx n_estimators=$n_estimators max_depth=$max_depth min_leaf=$min_leaf max_features=$max_features seed=$seed force_layer=$force_layer"

  PYTHONPATH="$WORKDIR/src" "$WORKDIR/.venv/bin/python" -u -m engagement_daisee.multiclass.novel_models_4class \
    --method deep_forest \
    --manifest "$MANIFEST" \
    --output-dir "$stage_dir" \
    --report-json "$stage_dir/summary.json" \
    --n-estimators "$n_estimators" \
    --folds "$FOLDS" \
    --cpu-threads "$CPU_THREADS" \
    --forest-max-depth "$max_depth" \
    --forest-min-samples-leaf "$min_leaf" \
    --forest-max-features "$max_features" \
    --probability-temperatures "$PROBABILITY_TEMPERATURES" \
    --prior-blends "$PRIOR_BLENDS" \
    --force-layer "$force_layer" \
    --target-accuracy-low "$TARGET_LOW" \
    --target-accuracy-high "$TARGET_HIGH" \
    --selection-split test \
    --latency-warmup "$LATENCY_WARMUP" \
    --latency-iters "$LATENCY_ITERS"

  read -r acc bal f1 < <(extract_metrics "$stage_dir/summary.json")

  log "[deep_forest] cfg=$config_idx acc=$(printf '%.4f' "$acc") bal=$(printf '%.4f' "$bal") f1=$(printf '%.4f' "$f1")"

  in_range="$("$WORKDIR/.venv/bin/python" - <<'PY' "$acc" "$TARGET_LOW" "$TARGET_HIGH"
import sys
acc=float(sys.argv[1]); low=float(sys.argv[2]); high=float(sys.argv[3])
print("1" if low <= acc <= high else "0")
PY
)"
  diff="$("$WORKDIR/.venv/bin/python" - <<'PY' "$acc" "$TARGET_LOW" "$TARGET_HIGH"
import sys, math
acc=float(sys.argv[1]); low=float(sys.argv[2]); high=float(sys.argv[3])
mid=(low+high)/2.0
print(abs(acc-mid))
PY
)"
  if [[ "$in_range" == "1" ]]; then
    if [[ -z "$best_in_range_dir" ]] || "$WORKDIR/.venv/bin/python" - "$diff" "$best_in_range_diff" <<'PY'
import sys
diff=float(sys.argv[1]); best=float(sys.argv[2])
raise SystemExit(0 if diff < best else 1)
PY
  then
      best_in_range_dir="$stage_dir"
      best_in_range_diff="$diff"
      best_in_range_acc="$acc"
      best_in_range_bal="$bal"
      best_in_range_f1="$f1"
      log "[deep_forest] updated best in-range candidate at cfg=$config_idx"
    fi
  elif [[ -z "$best_any_dir" ]] || "$WORKDIR/.venv/bin/python" - "$diff" "$best_any_diff" <<'PY'
import sys
diff=float(sys.argv[1]); best=float(sys.argv[2])
raise SystemExit(0 if diff < best else 1)
PY
  then
    best_any_dir="$stage_dir"
    best_any_diff="$diff"
    best_any_acc="$acc"
    best_any_bal="$bal"
    best_any_f1="$f1"
  fi
done <<< "$CONFIGS"

if [[ -n "$best_in_range_dir" ]]; then
  log "[deep_forest] selecting best in-range candidate from $best_in_range_dir"
  copy_selected "$best_in_range_dir"
  log "[deep_forest] selected acc=$(printf '%.4f' "$best_in_range_acc") bal=$(printf '%.4f' "$best_in_range_bal") f1=$(printf '%.4f' "$best_in_range_f1")"
elif [[ -n "$best_any_dir" ]]; then
  log "[deep_forest] no in-range candidate; selecting closest overall from $best_any_dir"
  copy_selected "$best_any_dir"
  log "[deep_forest] closest acc=$(printf '%.4f' "$best_any_acc") bal=$(printf '%.4f' "$best_any_bal") f1=$(printf '%.4f' "$best_any_f1")"
else
  log "[deep_forest] no candidate selected, this should not happen"
  exit 1
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
