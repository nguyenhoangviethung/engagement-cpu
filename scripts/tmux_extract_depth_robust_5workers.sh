#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
SESSION_NAME="engagement_extract_depth_robust_5w"
LATEST_LOG="$WORKDIR/logs/latest_extract_depth_robust_parallel.log"
COMMAND="${1:-start}"

case "$COMMAND" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session '$SESSION_NAME' is already running."
      exit 1
    fi
    run_id="depth_robust_5w_$(date +%Y%m%d_%H%M%S)"
    tmux new-session -d -s "$SESSION_NAME" \
      "bash '$WORKDIR/scripts/rnn/extract_depth_robust_parallel.sh' '$run_id' 5"
    echo "Started 5-worker session: $SESSION_NAME"
    echo "Attach: tmux attach -t $SESSION_NAME"
    echo "Logs: $0 logs"
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
  logs)
    if [[ -e "$LATEST_LOG" || -L "$LATEST_LOG" ]]; then
      tail -n 150 "$LATEST_LOG"
    else
      echo "No parallel extraction log found."
      exit 1
    fi
    ;;
  stop)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "$SESSION_NAME"
      echo "Stopped session: $SESSION_NAME"
    else
      echo "Session '$SESSION_NAME' is not running."
    fi
    ;;
  *)
    echo "Usage: $0 {start|attach|status|logs|stop}" >&2
    exit 2
    ;;
esac
