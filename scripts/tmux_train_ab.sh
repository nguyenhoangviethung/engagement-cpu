#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="engagement_train_ab"
CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$WORKDIR/logs"
RUNS_CHECKPOINT_DIR="$WORKDIR/checkpoints/runs"
DEFAULT_MANIFEST="$WORKDIR/data/processed/runs/pipeline_2/feature_manifest.csv"

LOG_EVERY=1
SAMPLE_MODE=0
AB_RUN_ID=""
MANIFEST_PATH="$DEFAULT_MANIFEST"
FOCAL_ALPHA="0.30"
FOCAL_GAMMA="2.0"
COMMAND="start"

usage() {
  cat <<'EOF'
Usage: scripts/tmux_train_ab.sh [command] [options]

Commands:
  start       Start A/B train in tmux (default)
  attach      Attach to active tmux session
  status      Show tmux session status
  stop        Stop the tmux session
  logs        Show latest A/B train log (tail)

Options:
  --manifest PATH      Feature manifest path
  --sample             Run sample training for both A and B
  --session NAME       tmux session name (default: engagement_train_ab)
  --env NAME           conda environment name (default: thesis)
  --log-every N        Log frequency for train.py
  --run-id ID          Custom run id (default: timestamp)
  --focal-alpha V      Focal alpha for config B (default: 0.30)
  --focal-gamma V      Focal gamma for config B (default: 2.0)
  --help               Show this help

A config:
  --threshold-objective balanced_accuracy
  --loss bce_weighted
  --train-sampler weighted

B config:
  --threshold-objective balanced_accuracy
  --loss focal --focal-alpha <value> --focal-gamma <value>
  --train-sampler weighted
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs)
      COMMAND="$1"
      shift
      ;;
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --sample)
      SAMPLE_MODE=1
      shift
      ;;
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --log-every)
      LOG_EVERY="$2"
      shift 2
      ;;
    --run-id)
      AB_RUN_ID="$2"
      shift 2
      ;;
    --focal-alpha)
      FOCAL_ALPHA="$2"
      shift 2
      ;;
    --focal-gamma)
      FOCAL_GAMMA="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR"
LATEST_LOG_LINK="$LOG_DIR/latest_train_ab.log"

build_command() {
  local sample_flag=""
  if [[ "$SAMPLE_MODE" -eq 1 ]]; then
    sample_flag=" --sample"
  fi

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local active_run_id="$AB_RUN_ID"
  if [[ -z "$active_run_id" ]]; then
    active_run_id="$timestamp"
  fi

  local run_a_dir="$RUNS_CHECKPOINT_DIR/train_${active_run_id}_A"
  local run_b_dir="$RUNS_CHECKPOINT_DIR/train_${active_run_id}_B"
  local ckpt_a="$run_a_dir/engagement_gru.pt"
  local ckpt_b="$run_b_dir/engagement_gru.pt"
  local json_a="$run_a_dir/engagement_gru.json"
  local json_b="$run_b_dir/engagement_gru.json"
  local run_log="$LOG_DIR/train_ab_${active_run_id}_${timestamp}.log"
  local compare_md="$LOG_DIR/train_ab_${active_run_id}_comparison.md"
  local compare_csv="$LOG_DIR/train_ab_${active_run_id}_comparison.csv"

  if [[ -e "$ckpt_a" || -e "$ckpt_b" || -e "$compare_md" || -e "$compare_csv" ]]; then
    echo "Output for run-id '$active_run_id' already exists. Use another --run-id."
    exit 1
  fi

  cat <<EOF
cd "$WORKDIR"
set -euo pipefail

echo "=== A/B train started at \$(date) ===" | tee -a "$run_log"
echo "Run ID: $active_run_id" | tee -a "$run_log"
echo "Manifest: $MANIFEST_PATH" | tee -a "$run_log"
echo "A checkpoint: $ckpt_a" | tee -a "$run_log"
echo "B checkpoint: $ckpt_b" | tee -a "$run_log"
echo "Sample mode: $SAMPLE_MODE" | tee -a "$run_log"
echo "Config A: objective=balanced_accuracy, loss=bce_weighted, sampler=weighted" | tee -a "$run_log"
echo "Config B: objective=balanced_accuracy, loss=focal(alpha=$FOCAL_ALPHA,gamma=$FOCAL_GAMMA), sampler=weighted" | tee -a "$run_log"

mkdir -p "$run_a_dir" "$run_b_dir"

echo "[A/2] Training config A..." | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" python -u train.py$sample_flag \\
  --manifest "$MANIFEST_PATH" \\
  --output "$ckpt_a" \\
  --log-every "$LOG_EVERY" \\
  --threshold-objective "balanced_accuracy" \\
  --loss "bce_weighted" \\
  --train-sampler "weighted" 2>&1 | tee -a "$run_log"

echo "[B/2] Training config B..." | tee -a "$run_log"
conda run --no-capture-output -n "$CONDA_ENV" python -u train.py$sample_flag \\
  --manifest "$MANIFEST_PATH" \\
  --output "$ckpt_b" \\
  --log-every "$LOG_EVERY" \\
  --threshold-objective "balanced_accuracy" \\
  --loss "focal" \\
  --focal-alpha "$FOCAL_ALPHA" \\
  --focal-gamma "$FOCAL_GAMMA" \\
  --train-sampler "weighted" 2>&1 | tee -a "$run_log"

echo "[Compare] Building summary table..." | tee -a "$run_log"
python - "$json_a" "$json_b" "$compare_md" "$compare_csv" <<'PY' | tee -a "$run_log"
import csv
import json
import sys
from pathlib import Path

json_a = Path(sys.argv[1])
json_b = Path(sys.argv[2])
compare_md = Path(sys.argv[3])
compare_csv = Path(sys.argv[4])

def load_row(name: str, path: Path) -> dict:
    data = json.loads(path.read_text())
    return {
        "config": name,
        "best_val_balanced_accuracy": float(data.get("best_val_balanced_accuracy", 0.0)),
        "test_balanced_accuracy": float(data.get("test_balanced_accuracy", 0.0)),
        "test_recall_neg": float(data.get("test_recall_neg", 0.0)),
        "test_recall_pos": float(data.get("test_recall_pos", 0.0)),
        "best_threshold": float(data.get("best_threshold", 0.5)),
    }

rows = [
    load_row("A_bce_weighted", json_a),
    load_row("B_focal", json_b),
]
rows.sort(key=lambda r: r["test_balanced_accuracy"], reverse=True)

header = [
    "config",
    "best_val_balanced_accuracy",
    "test_balanced_accuracy",
    "test_recall_neg",
    "test_recall_pos",
    "best_threshold",
]

compare_csv.parent.mkdir(parents=True, exist_ok=True)
with compare_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

lines = []
lines.append("| config | val_bal_acc | test_bal_acc | test_recall_neg | test_recall_pos | threshold |")
lines.append("|---|---:|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| {row['config']} | {row['best_val_balanced_accuracy']:.4f} | "
        f"{row['test_balanced_accuracy']:.4f} | {row['test_recall_neg']:.4f} | "
        f"{row['test_recall_pos']:.4f} | {row['best_threshold']:.2f} |"
    )

best = rows[0]
lines.append("")
lines.append(f"Best by test_balanced_accuracy: **{best['config']}** ({best['test_balanced_accuracy']:.4f})")
markdown = "\n".join(lines)
compare_md.write_text(markdown)
print(markdown)
print(f"Saved markdown: {compare_md}")
print(f"Saved csv: {compare_csv}")
PY

echo "=== A/B train finished at \$(date) ===" | tee -a "$run_log"
ln -sfn "$run_log" "$LATEST_LOG_LINK"
EOF
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists. Use attach/status/stop first."
      exit 1
    fi
    RUN_SCRIPT="$(build_command)"
    tmux new-session -d -s "$SESSION_NAME" "bash -lc '$RUN_SCRIPT'"
    echo "Started tmux session: $SESSION_NAME"
    echo "Attach: scripts/tmux_train_ab.sh attach --session $SESSION_NAME"
    ;;
  attach)
    tmux attach -t "$SESSION_NAME"
    ;;
  status)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' is running."
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
      tail -n 160 "$LATEST_LOG_LINK"
    else
      echo "No latest log found at $LATEST_LOG_LINK"
      exit 1
    fi
    ;;
  *)
    usage
    exit 1
    ;;
esac

