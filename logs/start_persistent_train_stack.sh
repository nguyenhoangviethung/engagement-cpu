#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/bear/engagement-cpu"
TRAIN_SCRIPT="$WORKDIR/logs/openface709_safe_train_20260525.runner.sh"
WATCHDOG_SCRIPT="$WORKDIR/logs/watchdog_train_guard_20260525.sh"
TRAIN_NOHUP_LOG="$WORKDIR/logs/persistent_train.nohup.log"
WATCHDOG_NOHUP_LOG="$WORKDIR/logs/persistent_watchdog.nohup.log"
PID_DIR="$WORKDIR/logs/.pids"
mkdir -p "$PID_DIR"

start_one() {
  local name="$1" cmd="$2" pidfile="$3" logfile="$4"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "[$name] already running pid=$(cat "$pidfile")"
    return
  fi
  echo "[$name] starting..."
  # setsid + nohup detaches from VS Code/SSH terminal lifecycle.
  setsid nohup bash -lc "$cmd" >> "$logfile" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$pidfile"
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    echo "[$name] started pid=$pid"
  else
    echo "[$name] failed to start; check $logfile"
    exit 1
  fi
}

start_one "train" "cd '$WORKDIR' && exec '$TRAIN_SCRIPT'" "$PID_DIR/train.pid" "$TRAIN_NOHUP_LOG"
start_one "watchdog" "cd '$WORKDIR' && exec '$WATCHDOG_SCRIPT'" "$PID_DIR/watchdog.pid" "$WATCHDOG_NOHUP_LOG"

echo "stack status:"
ps -fp "$(cat "$PID_DIR/train.pid")" || true
ps -fp "$(cat "$PID_DIR/watchdog.pid")" || true
