#!/usr/bin/env bash
set -euo pipefail

# Sequential runner: for each compression level, start a tmux session
# to run scripts/tmux_train_all.sh against the level's manifest and wait
# for the session to finish before proceeding.

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$WORKDIR/.venv/bin/activate"

read -r -a levels <<< "${LEVELS:-pca96 pca128 pca160 pca192 pca224 pca256 pca300 pca384}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
RNN_CPU_THREADS="${RNN_CPU_THREADS:-2}"
ML_CPU_WORKERS="${ML_CPU_WORKERS:-2}"

for lvl in "${levels[@]}"; do
  MANIFEST="$WORKDIR/data/processed/runs/openface709_pca_manifests/${lvl}/feature_manifest.csv"
  if [ ! -f "$MANIFEST" ]; then
    echo "[run_all] Missing manifest for ${lvl}, skipping"
    continue
  fi

  SESSION="train_manifest_${lvl}"
  RUN_PREFIX="${lvl}"

  # ensure previous session with same name removed
  tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true

  scripts/tmux_train_all.sh start \
    --session "$SESSION" \
    --rnn-manifest "$MANIFEST" \
    --ml-manifest "$MANIFEST" \
    --device "$TRAIN_DEVICE" \
    --rnn-cpu-threads "$RNN_CPU_THREADS" \
    --ml-cpu-workers "$ML_CPU_WORKERS" \
    --run-id-prefix "$RUN_PREFIX"

  echo "[run_all] Started tmux session $SESSION for level $lvl"

  # wait until session disappears (train finished or session killed)
  while tmux has-session -t "$SESSION" 2>/dev/null; do
    sleep 30
  done

  echo "[run_all] Session $SESSION finished"
done

echo "[run_all] All levels processed"
