#!/usr/bin/env bash
set -euo pipefail
WORKDIR="/home/bear/engagement-cpu"
PID_DIR="$WORKDIR/logs/.pids"
for name in train watchdog; do
  pidfile="$PID_DIR/$name.pid"
  if [[ -f "$pidfile" ]]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name: running pid=$pid"
      ps -fp "$pid" | sed -n '2p'
    else
      echo "$name: not running (stale pid $pid)"
    fi
  else
    echo "$name: no pidfile"
  fi
done
