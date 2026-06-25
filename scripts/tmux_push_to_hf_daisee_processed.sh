#!/usr/bin/env bash
set -euo pipefail

COMMAND="start"
SESSION_NAME="engagement_push_hf_daisee_processed"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
LATEST_LOG="$LOG_DIR/latest_hf_push_daisee_processed.log"
REPO_ID="Hnug/daisee-processed"

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs)
      COMMAND="$1"
      shift
      ;;
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --help|-h)
      cat <<USAGE
Usage: scripts/tmux_push_to_hf_daisee_processed.sh [start|attach|status|stop|logs] [--session NAME] [--repo-id ID]
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR"

start_push() {
  local timestamp log_file runner_script
  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOG_DIR/hf_push_daisee_processed_${timestamp}.log"
  runner_script="$LOG_DIR/run_hf_push_daisee_processed_${timestamp}.sh"

  cat >"$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$WORKDIR")
ln -sfn $(printf '%q' "$log_file") $(printf '%q' "$LATEST_LOG")
export HF_XET_HIGH_PERFORMANCE=1
export HF_TOKEN="$(tr -d '\n\r' < "$WORKDIR/.env")"
REPO_ID=$(printf '%q' "$REPO_ID") WORKDIR=$(printf '%q' "$WORKDIR") $(printf '%q' "$WORKDIR/.venv/bin/python") -u - <<'PY' 2>&1 | tee -a $(printf '%q' "$log_file")
from pathlib import Path
from huggingface_hub import HfApi
import os

repo_id = os.environ["REPO_ID"]
token = os.environ["HF_TOKEN"]
workdir = Path(os.environ["WORKDIR"])
api = HfApi(token=token)

print(f"Creating/checking dataset repo: {repo_id}", flush=True)
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

processed_root = workdir / "data" / "processed"
models_root = workdir / "checkpoints" / "runs" / "full_train_4class_20260621_053823"

print("Uploading processed dataset artifacts...", flush=True)
api.upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=str(processed_root),
    allow_patterns=[
        "final_feature_manifest.csv",
        "feature_manifest.csv",
        "engagement_only_labels.csv",
        "runs/extract_depth_robust_5w_20260620_130850/**",
    ],
    num_workers=8,
    print_report=True,
    print_report_every=60,
)

print("Uploading trained models and summaries...", flush=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=str(models_root),
    path_in_repo="models/full_train_4class_20260621_053823",
    commit_message="Upload latest trained 4-class models",
)

print("HF upload finished successfully.", flush=True)
PY
EOF
  chmod +x "$runner_script"
  tmux new-session -d -s "$SESSION_NAME" "bash '$runner_script'"
  echo "Started HF upload session: $SESSION_NAME"
  echo "Log: $log_file"
  echo "Attach: $0 attach --session $SESSION_NAME"
}

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' already exists."
      exit 1
    fi
    start_push
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
    if [[ -L "$LATEST_LOG" || -f "$LATEST_LOG" ]]; then
      tail -n 200 "$LATEST_LOG"
    else
      echo "No latest log found at $LATEST_LOG"
      exit 1
    fi
    ;;
esac
