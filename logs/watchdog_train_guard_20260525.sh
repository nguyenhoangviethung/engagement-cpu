#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/bear/engagement-cpu"
TRAIN_SESSION="openface709_safe_train_safe"
TRAIN_SCRIPT="$WORKDIR/logs/openface709_safe_train_20260525.runner.sh"
TRAIN_LOG="$WORKDIR/logs/openface709_safe_train_20260525.log"
WATCHDOG_LOG="$WORKDIR/logs/watchdog_train_guard_20260525.log"
DURATION_SEC=3600
INTERVAL_SEC=15
DISABLE_TMUX_AUTOSTART="${DISABLE_TMUX_AUTOSTART:-0}"

# thresholds (conservative for 15 GiB RAM, no swap)
MIN_AVAIL_MEM_MB=1800
MAX_MEM_PERCENT=92
MAX_LOAD_PER_CPU=3.2

start_ts=$(date +%s)
end_ts=$((start_ts + DURATION_SEC))

log() {
  echo "[$(date -Is)] $*" | tee -a "$WATCHDOG_LOG"
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

gpu_snapshot() {
  if have_cmd nvidia-smi; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1
  else
    echo "no-gpu"
  fi
}

mem_snapshot() {
  free -m | awk 'NR==2{printf "%d %d %d\n", $2, $3, $7}'
}

load_per_cpu() {
  local l c
  l=$(awk '{print $1}' /proc/loadavg)
  c=$(nproc)
  awk -v l="$l" -v c="$c" 'BEGIN{printf "%.2f", l/c}'
}

ensure_train_session() {
  if [[ "$DISABLE_TMUX_AUTOSTART" == "1" ]]; then
    return
  fi
  if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    return
  fi
  log "train session missing -> start new tmux session $TRAIN_SESSION"
  tmux new-session -d -s "$TRAIN_SESSION" "cd $WORKDIR && exec $TRAIN_SCRIPT"
}

scale_down_runner() {
  if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    log "ERROR: train script missing: $TRAIN_SCRIPT"
    return 1
  fi

  # Decrease RNN batch size by half (min 8), and reduce cpu threads to 1.
  local current_bs new_bs
  current_bs=$(awk -F'"' '/^RNN_BATCH_SIZE=/{print $2}' "$TRAIN_SCRIPT" | head -n1)
  if [[ -z "$current_bs" ]]; then current_bs=32; fi
  new_bs=$(( current_bs / 2 ))
  if (( new_bs < 8 )); then new_bs=8; fi

  sed -i -E "s/^RNN_BATCH_SIZE=\"[0-9]+\"/RNN_BATCH_SIZE=\"${new_bs}\"/" "$TRAIN_SCRIPT"
  sed -i -E 's/^CPU_THREADS="[0-9]+"/CPU_THREADS="1"/' "$TRAIN_SCRIPT"
  sed -i -E 's/^ML_CPU_WORKERS="[0-9]+"/ML_CPU_WORKERS="1"/' "$TRAIN_SCRIPT"

  log "scaled runner config: RNN_BATCH_SIZE ${current_bs} -> ${new_bs}, CPU_THREADS=1, ML_CPU_WORKERS=1"
}

restart_train_safely() {
  if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    log "sending Ctrl-C to $TRAIN_SESSION"
    tmux send-keys -t "$TRAIN_SESSION" C-c
    sleep 3
    if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
      tmux kill-session -t "$TRAIN_SESSION" || true
    fi
  fi
  log "restarting train session: $TRAIN_SESSION"
  tmux new-session -d -s "$TRAIN_SESSION" "cd $WORKDIR && exec $TRAIN_SCRIPT"
}

trap 'log "watchdog aborted unexpectedly at line $LINENO"' ERR

log "watchdog started | duration=${DURATION_SEC}s interval=${INTERVAL_SEC}s"
log "guarding session=$TRAIN_SESSION script=$TRAIN_SCRIPT"

anomaly_count=0
while (( $(date +%s) < end_ts )); do
  ensure_train_session

  read -r mem_total mem_used mem_avail < <(mem_snapshot)
  mem_pct=$(awk -v u="$mem_used" -v t="$mem_total" 'BEGIN{printf "%d", (u*100)/t}')
  lpc=$(load_per_cpu)
  gpu=$(gpu_snapshot)

  log "snapshot mem_avail_mb=${mem_avail} mem_pct=${mem_pct} load_per_cpu=${lpc} gpu=${gpu}"

  abnormal=0
  reason=""

  if (( mem_avail < MIN_AVAIL_MEM_MB )); then
    abnormal=1
    reason="low_available_mem"
  fi
  if (( mem_pct > MAX_MEM_PERCENT )); then
    abnormal=1
    reason="high_mem_percent"
  fi
  awk -v x="$lpc" -v th="$MAX_LOAD_PER_CPU" 'BEGIN{exit !(x>th)}' && {
    abnormal=1
    reason="high_cpu_load"
  }

  if (( abnormal == 1 )); then
    anomaly_count=$((anomaly_count + 1))
    log "ANOMALY #${anomaly_count}: ${reason}"

    # Two consecutive abnormal checks -> mitigate immediately.
    if (( anomaly_count >= 2 )); then
      log "trigger mitigation: scale-down + restart train"
      scale_down_runner || true
      restart_train_safely
      anomaly_count=0
      sleep 20
    fi
  else
    anomaly_count=0
  fi

  sleep "$INTERVAL_SEC"
done

log "watchdog finished after 1 hour"
