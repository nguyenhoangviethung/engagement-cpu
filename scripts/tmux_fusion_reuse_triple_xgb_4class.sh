#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_fusion_reuse_triple_xgb_4class"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
RUNS_DIR="$WORKDIR/checkpoints/runs"
LATEST_LOG_LINK="$LOG_DIR/latest_fusion_reuse_triple_xgb_4class.log"
COMMAND="start"

MANIFEST="$WORKDIR/data/processed/final_feature_manifest.csv"
FEATURE_MODE="tsfresh"
WEIGHT_STEP=0.01
RUN_ID_PREFIX="triple_xgb_reuse_fusion"

FINAL_XGB_MODEL="$WORKDIR/checkpoints/runs/full_train_4class_20260621_053823/train_all/xgboost/model.json"
FINAL_XGB_PREPROCESSOR="$WORKDIR/checkpoints/runs/full_train_4class_20260621_053823/train_all/xgboost/preprocessor.npz"
BOOST_XGB_MODEL="$WORKDIR/checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_boost_xgb/model.json"
BOOST_XGB_PREPROCESSOR="$WORKDIR/checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_boost_xgb/preprocessor.npz"
TARGETED_XGB_MODEL="$WORKDIR/checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_targeted_xgb/model.json"
TARGETED_XGB_PREPROCESSOR="$WORKDIR/checkpoints/runs/full_train_4class_20260621_053823/strong_followups/accuracy_targeted_xgb/preprocessor.npz"

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
  --manifest PATH
  --feature-mode MODE
  --weight-step FLOAT
  --run-id-prefix ID
  --final-xgb-model PATH
  --final-xgb-preprocessor PATH
  --boost-xgb-model PATH
  --boost-xgb-preprocessor PATH
  --targeted-xgb-model PATH
  --targeted-xgb-preprocessor PATH
  --env NAME
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs) COMMAND="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --feature-mode) FEATURE_MODE="$2"; shift 2 ;;
    --weight-step) WEIGHT_STEP="$2"; shift 2 ;;
    --run-id-prefix) RUN_ID_PREFIX="$2"; shift 2 ;;
    --final-xgb-model) FINAL_XGB_MODEL="$2"; shift 2 ;;
    --final-xgb-preprocessor) FINAL_XGB_PREPROCESSOR="$2"; shift 2 ;;
    --boost-xgb-model) BOOST_XGB_MODEL="$2"; shift 2 ;;
    --boost-xgb-preprocessor) BOOST_XGB_PREPROCESSOR="$2"; shift 2 ;;
    --targeted-xgb-model) TARGETED_XGB_MODEL="$2"; shift 2 ;;
    --targeted-xgb-preprocessor) TARGETED_XGB_PREPROCESSOR="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

for required in "$MANIFEST" "$FINAL_XGB_MODEL" "$FINAL_XGB_PREPROCESSOR" "$BOOST_XGB_MODEL" "$BOOST_XGB_PREPROCESSOR" "$TARGETED_XGB_MODEL" "$TARGETED_XGB_PREPROCESSOR"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file not found: $required" >&2
    exit 1
  fi
done

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
trap 'status=\$?; if [[ \$status -ne 0 ]]; then echo "=== reuse triple xgb fusion failed with exit code \$status at \$(date) ===" | tee -a $(shell_quote "$run_log"); fi' EXIT

echo "=== reuse triple xgb fusion started at \$(date) ===" | tee -a $(shell_quote "$run_log")
echo "Run root: $(shell_quote "$run_root")" | tee -a $(shell_quote "$run_log")
echo "Manifest: $(shell_quote "$MANIFEST")" | tee -a $(shell_quote "$run_log")

MAXACC_JSON=$(shell_quote "$run_root/fusion_maxacc_bal75/summary.json")
BAND_JSON=$(shell_quote "$run_root/fusion_acc75_77_bal75/summary.json")

echo "=== [1/2] max accuracy with balanced >= 75% ===" | tee -a $(shell_quote "$run_log")
bash $(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") \
  env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.multiclass.fusion_sweep_xgb \
  --manifest $(shell_quote "$MANIFEST") \
  --output-json "\$MAXACC_JSON" \
  --final-xgb-model $(shell_quote "$FINAL_XGB_MODEL") \
  --final-xgb-preprocessor $(shell_quote "$FINAL_XGB_PREPROCESSOR") \
  --boost-xgb-model $(shell_quote "$BOOST_XGB_MODEL") \
  --boost-xgb-preprocessor $(shell_quote "$BOOST_XGB_PREPROCESSOR") \
  --targeted-xgb-model $(shell_quote "$TARGETED_XGB_MODEL") \
  --targeted-xgb-preprocessor $(shell_quote "$TARGETED_XGB_PREPROCESSOR") \
  --feature-mode $(shell_quote "$FEATURE_MODE") \
  --weight-step $(shell_quote "$WEIGHT_STEP") \
  --selection-mode max_accuracy \
  --min-accuracy 0.0 \
  --min-balanced-accuracy 0.75 \
  --max-accuracy 1.0 \
  --max-balanced-accuracy 1.0 \
  --latency-warmup 30 \
  --latency-iters 200 2>&1 | tee -a $(shell_quote "$run_log")

echo "=== [2/2] accuracy 75-77% with balanced >= 75% ===" | tee -a $(shell_quote "$run_log")
bash $(shell_quote "$WORKDIR/scripts/lib/run_python.sh") --env $(shell_quote "$CONDA_ENV") --workdir $(shell_quote "$WORKDIR") \
  env PYTHONPATH=$(shell_quote "$WORKDIR/src") python -u -m engagement_daisee.multiclass.fusion_sweep_xgb \
  --manifest $(shell_quote "$MANIFEST") \
  --output-json "\$BAND_JSON" \
  --final-xgb-model $(shell_quote "$FINAL_XGB_MODEL") \
  --final-xgb-preprocessor $(shell_quote "$FINAL_XGB_PREPROCESSOR") \
  --boost-xgb-model $(shell_quote "$BOOST_XGB_MODEL") \
  --boost-xgb-preprocessor $(shell_quote "$BOOST_XGB_PREPROCESSOR") \
  --targeted-xgb-model $(shell_quote "$TARGETED_XGB_MODEL") \
  --targeted-xgb-preprocessor $(shell_quote "$TARGETED_XGB_PREPROCESSOR") \
  --feature-mode $(shell_quote "$FEATURE_MODE") \
  --weight-step $(shell_quote "$WEIGHT_STEP") \
  --selection-mode target_band \
  --min-accuracy 0.75 \
  --min-balanced-accuracy 0.75 \
  --max-accuracy 0.77 \
  --max-balanced-accuracy 1.0 \
  --latency-warmup 30 \
  --latency-iters 200 2>&1 | tee -a $(shell_quote "$run_log")

python - <<'PY' "\$MAXACC_JSON" "\$BAND_JSON" "$(shell_quote "$run_root/selection_summary.json")" | tee -a $(shell_quote "$run_log")
import json
import sys
from pathlib import Path

maxacc, band, out = map(Path, sys.argv[1:])

def pick(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "summary": str(path),
        "selection_mode": payload.get("selection_mode"),
        "selected": payload.get("selected"),
        "test_metrics": payload.get("test_metrics"),
        "latency": payload.get("latency"),
        "sources": payload.get("sources"),
    }

summary = {
    "run_root": str(out.parent),
    "model_family": "reused_depth_robust_triple_xgb_fusion",
    "maxacc_bal75": pick(maxacc),
    "acc75_77_bal75": pick(band),
}
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "Selection summary: $(shell_quote "$run_root/selection_summary.json")" | tee -a $(shell_quote "$run_log")
echo "=== reuse triple xgb fusion finished at \$(date) ===" | tee -a $(shell_quote "$run_log")
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
  attach) tmux attach -t "$SESSION_NAME" ;;
  status) tmux list-sessions 2>/dev/null | grep "^$SESSION_NAME:" || { echo "Session '$SESSION_NAME' is not running."; exit 1; } ;;
  stop) tmux kill-session -t "$SESSION_NAME" && echo "Stopped tmux session: $SESSION_NAME" || { echo "Session '$SESSION_NAME' is not running."; exit 1; } ;;
  logs)
    if [[ -L "$LATEST_LOG_LINK" || -f "$LATEST_LOG_LINK" ]]; then
      tail -n 160 "$LATEST_LOG_LINK"
    else
      echo "No latest log yet: $LATEST_LOG_LINK"
      exit 1
    fi
    ;;
esac
