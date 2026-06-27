#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_tune_triple_xgb_4class"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_DIR="$WORKDIR/checkpoints/runs"
LATEST_LOG_LINK="$LOG_DIR/latest_tune_triple_xgb_4class.log"
COMMAND="start"

MANIFEST="$WORKDIR/data/processed/final_feature_manifest.csv"
FEATURE_MODE="tsfresh"
CPU_THREADS=8
XGB_THREADS=8
N_ESTIMATORS_BOOST=900
N_ESTIMATORS_TARGETED=1400
ROUND_STEP=25
WEIGHT_STEP=0.01
RUN_ID_PREFIX="triple_xgb_depth_robust_tune"

shell_quote() {
  printf "%q" "$1"
}

usage() {
  cat <<EOF
Usage:
  $0 start [options]
  $0 status|logs|attach|stop

Options:
  --session NAME
  --manifest PATH                  default: data/processed/final_feature_manifest.csv
  --feature-mode MODE              default: tsfresh
  --cpu-threads N                  default: 8
  --xgb-threads N                  default: 8
  --boost-estimators N             default: 900
  --targeted-estimators N          default: 1400
  --round-step N                   default: 25
  --weight-step FLOAT              default: 0.01
  --run-id-prefix ID               default: triple_xgb_depth_robust_tune
  --env NAME                       default: thesis
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --cpu-threads) CPU_THREADS="$2"; shift 2 ;;
    --xgb-threads) XGB_THREADS="$2"; shift 2 ;;
    --boost-estimators) N_ESTIMATORS_BOOST="$2"; shift 2 ;;
    --targeted-estimators) N_ESTIMATORS_TARGETED="$2"; shift 2 ;;
    --round-step) ROUND_STEP="$2"; shift 2 ;;
    --weight-step) WEIGHT_STEP="$2"; shift 2 ;;
    --run-id-prefix) RUN_ID_PREFIX="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$RUNS_DIR"

start_session() {
  local timestamp run_root run_log runner_script
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_root="$RUNS_DIR/${RUN_ID_PREFIX}_${timestamp}"
  run_log="$LOG_DIR/${RUN_ID_PREFIX}_${timestamp}.log"
  runner_script="$LOG_DIR/run_${RUN_ID_PREFIX}_${timestamp}.sh"

  cat >"$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd $(shell_quote "$WORKDIR")
mkdir -p $(shell_quote "$run_root")
ln -sfn $(shell_quote "$run_log") $(shell_quote "$LATEST_LOG_LINK")

export OMP_NUM_THREADS=$(shell_quote "$CPU_THREADS")
export OPENBLAS_NUM_THREADS=$(shell_quote "$CPU_THREADS")
export MKL_NUM_THREADS=$(shell_quote "$CPU_THREADS")
export NUMEXPR_NUM_THREADS=$(shell_quote "$CPU_THREADS")

trap 'status=\$?; if [[ \$status -ne 0 ]]; then echo "=== triple xgb tune failed with exit code \$status at \$(date) ===" | tee -a $(shell_quote "$run_log"); fi' EXIT

echo "=== triple xgb tune started at \$(date) ===" | tee -a $(shell_quote "$run_log")
echo "Run root: $(shell_quote "$run_root")" | tee -a $(shell_quote "$run_log")
echo "Manifest: $(shell_quote "$MANIFEST")" | tee -a $(shell_quote "$run_log")

BASE_DIR=$(shell_quote "$run_root/base_xgb")
BOOST_DIR=$(shell_quote "$run_root/boost_xgb")
TARGETED_DIR=$(shell_quote "$run_root/targeted_xgb")
FUSION_MAXACC=$(shell_quote "$run_root/fusion_maxacc_bal75/summary.json")
FUSION_BAND=$(shell_quote "$run_root/fusion_acc75_77_bal75/summary.json")

echo "=== [1/5] train base xgboost ===" | tee -a $(shell_quote "$run_log")
bash $(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") \
  env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.multiclass.train_all \
  --manifest $(shell_quote "$MANIFEST") \
  --output-dir "\$BASE_DIR" \
  --models xgboost \
  --device cpu \
  --feature-mode $(shell_quote "$FEATURE_MODE") \
  --objective balanced_accuracy \
  --cpu-threads $(shell_quote "$CPU_THREADS") \
  --xgb-threads $(shell_quote "$XGB_THREADS") \
  --latency-threads 1 \
  --latency-warmup 20 \
  --latency-iters 120 2>&1 | tee -a $(shell_quote "$run_log")

echo "=== [2/5] train accuracy boost xgboost ===" | tee -a $(shell_quote "$run_log")
bash $(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") \
  env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.multiclass.accuracy_boost_xgb \
  --manifest $(shell_quote "$MANIFEST") \
  --output-dir "\$BOOST_DIR" \
  --report-json "\$BOOST_DIR/summary.json" \
  --feature-mode $(shell_quote "$FEATURE_MODE") \
  --n-estimators $(shell_quote "$N_ESTIMATORS_BOOST") \
  --round-step $(shell_quote "$ROUND_STEP") \
  --cpu-threads $(shell_quote "$XGB_THREADS") \
  --latency-warmup 20 \
  --latency-iters 120 2>&1 | tee -a $(shell_quote "$run_log")

echo "=== [3/5] train balanced targeted xgboost ===" | tee -a $(shell_quote "$run_log")
bash $(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") \
  env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.multiclass.accuracy_targeted_xgb \
  --manifest $(shell_quote "$MANIFEST") \
  --output-dir "\$TARGETED_DIR" \
  --report-json "\$TARGETED_DIR/summary.json" \
  --feature-mode $(shell_quote "$FEATURE_MODE") \
  --n-estimators $(shell_quote "$N_ESTIMATORS_TARGETED") \
  --round-step $(shell_quote "$ROUND_STEP") \
  --cpu-threads $(shell_quote "$XGB_THREADS") \
  --min-accuracy 0.75 \
  --min-balanced-accuracy 0.75 \
  --only-candidates "" \
  --latency-warmup 20 \
  --latency-iters 120 2>&1 | tee -a $(shell_quote "$run_log")

echo "=== [4/5] fusion target: max accuracy with balanced >= 75% ===" | tee -a $(shell_quote "$run_log")
bash $(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") \
  env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.multiclass.fusion_sweep_xgb \
  --manifest $(shell_quote "$MANIFEST") \
  --output-json "\$FUSION_MAXACC" \
  --final-xgb-model "\$BASE_DIR/xgboost/model.json" \
  --final-xgb-preprocessor "\$BASE_DIR/xgboost/preprocessor.npz" \
  --boost-xgb-model "\$BOOST_DIR/model.json" \
  --boost-xgb-preprocessor "\$BOOST_DIR/preprocessor.npz" \
  --targeted-xgb-model "\$TARGETED_DIR/model.json" \
  --targeted-xgb-preprocessor "\$TARGETED_DIR/preprocessor.npz" \
  --feature-mode $(shell_quote "$FEATURE_MODE") \
  --weight-step $(shell_quote "$WEIGHT_STEP") \
  --min-accuracy 0.0 \
  --min-balanced-accuracy 0.75 \
  --max-accuracy 1.0 \
  --max-balanced-accuracy 1.0 \
  --latency-warmup 20 \
  --latency-iters 120 2>&1 | tee -a $(shell_quote "$run_log")

echo "=== [5/5] fusion target: accuracy 75-77% with balanced >= 75% ===" | tee -a $(shell_quote "$run_log")
bash $(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") \
  env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.multiclass.fusion_sweep_xgb \
  --manifest $(shell_quote "$MANIFEST") \
  --output-json "\$FUSION_BAND" \
  --final-xgb-model "\$BASE_DIR/xgboost/model.json" \
  --final-xgb-preprocessor "\$BASE_DIR/xgboost/preprocessor.npz" \
  --boost-xgb-model "\$BOOST_DIR/model.json" \
  --boost-xgb-preprocessor "\$BOOST_DIR/preprocessor.npz" \
  --targeted-xgb-model "\$TARGETED_DIR/model.json" \
  --targeted-xgb-preprocessor "\$TARGETED_DIR/preprocessor.npz" \
  --feature-mode $(shell_quote "$FEATURE_MODE") \
  --weight-step $(shell_quote "$WEIGHT_STEP") \
  --min-accuracy 0.75 \
  --min-balanced-accuracy 0.75 \
  --max-accuracy 0.77 \
  --max-balanced-accuracy 1.0 \
  --latency-warmup 20 \
  --latency-iters 120 2>&1 | tee -a $(shell_quote "$run_log")

python - <<'PY' "\$BASE_DIR/xgboost/summary.json" "\$BOOST_DIR/summary.json" "\$TARGETED_DIR/summary.json" "\$FUSION_MAXACC" "\$FUSION_BAND" "$(shell_quote "$run_root/selection_summary.json")" | tee -a $(shell_quote "$run_log")
import json
import sys
from pathlib import Path

base, boost, targeted, maxacc, band, out = map(Path, sys.argv[1:])

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def metrics(payload):
    return payload.get("test_video_metrics") or payload.get("test_metrics") or {}

payload = {
    "run_root": str(out.parent),
    "criteria": {
        "maxacc_bal75": "highest validation-selected Triple-XGB fusion candidate with balanced_accuracy >= 0.75",
        "acc75_77_bal75": "validation-selected Triple-XGB fusion candidate with 0.75 <= accuracy <= 0.77 and balanced_accuracy >= 0.75",
    },
    "components": {
        "base_xgb": {"summary": str(base), "test_metrics": metrics(load(base))},
        "boost_xgb": {"summary": str(boost), "test_metrics": metrics(load(boost))},
        "targeted_xgb": {"summary": str(targeted), "test_metrics": metrics(load(targeted))},
    },
    "selected_fusions": {
        "maxacc_bal75": {"summary": str(maxacc), "test_metrics": metrics(load(maxacc)), "selected": load(maxacc).get("selected")},
        "acc75_77_bal75": {"summary": str(band), "test_metrics": metrics(load(band)), "selected": load(band).get("selected")},
    },
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload["selected_fusions"], indent=2))
PY

echo "Selection summary: $(shell_quote "$run_root/selection_summary.json")" | tee -a $(shell_quote "$run_log")
echo "=== triple xgb tune finished at \$(date) ===" | tee -a $(shell_quote "$run_log")
EOF

  chmod +x "$runner_script"
  tmux new-session -d -s "$SESSION_NAME" "bash '$runner_script'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Log: $run_log"
  echo "Run root: $run_root"
  echo "Attach: $0 attach --session $SESSION_NAME"
  echo "Status: $0 status --session $SESSION_NAME"
  echo "Logs: $0 logs --session $SESSION_NAME"
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
    tmux list-sessions 2>/dev/null | grep "^$SESSION_NAME:" || { echo "Session '$SESSION_NAME' is not running."; exit 1; }
    ;;
  stop)
    tmux kill-session -t "$SESSION_NAME" && echo "Stopped tmux session: $SESSION_NAME" || { echo "Session '$SESSION_NAME' is not running."; exit 1; }
    ;;
  logs)
    if [[ -L "$LATEST_LOG_LINK" || -f "$LATEST_LOG_LINK" ]]; then
      tail -n 160 "$LATEST_LOG_LINK"
    else
      echo "No latest log yet: $LATEST_LOG_LINK"
      exit 1
    fi
    ;;
esac
