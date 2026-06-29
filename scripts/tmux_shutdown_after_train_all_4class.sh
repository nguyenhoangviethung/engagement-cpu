#!/usr/bin/env bash
set -euo pipefail

COMMAND="start"
SESSION_TO_WATCH=""
WATCH_SESSION_NAME="engagement_shutdown_after_train_all_4class"
POLL_SECONDS="${POLL_SECONDS:-60}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="$WORKDIR/logs"
LATEST_LOG="$LOG_DIR/latest_shutdown_after_train_all_4class.log"

mkdir -p "$LOG_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|attach|status|stop|logs)
      COMMAND="$1"
      shift
      ;;
    --session|--watch-session)
      SESSION_TO_WATCH="$2"
      shift 2
      ;;
    --watcher-session|--session-name)
      WATCH_SESSION_NAME="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --help|-h)
  echo "Usage: $0 [start|attach|status|stop|logs] [--session NAME] [--watcher-session NAME] [--poll-seconds N]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

start_watch() {
  local timestamp watch_log runner_script
  timestamp="$(date +%Y%m%d_%H%M%S)"
  watch_log="$LOG_DIR/shutdown_after_train_all_4class_${timestamp}.log"
  runner_script="$LOG_DIR/run_shutdown_after_train_all_4class_${timestamp}.sh"

  cat >"$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf "%q" "$WORKDIR")
ln -sfn $(printf "%q" "$watch_log") $(printf "%q" "$LATEST_LOG")
echo "=== shutdown watcher started at \$(date) ===" | tee -a $(printf "%q" "$watch_log")
if [[ -n $(printf "%q" "$SESSION_TO_WATCH") ]]; then
  echo "Watching session: $(printf "%q" "$SESSION_TO_WATCH")" | tee -a $(printf "%q" "$watch_log")
else
  echo "Watching all tmux sessions except this watcher" | tee -a $(printf "%q" "$watch_log")
fi
poweroff_now() {
  if sudo -n shutdown -h now; then
    return 0
  fi
  if sudo -n systemctl poweroff; then
    return 0
  fi
  if sudo -n poweroff; then
    return 0
  fi
  echo "Unable to power off with sudo -n; check sudoers permissions." | tee -a $(printf "%q" "$watch_log")
  return 1
}
watch_any() {
  local sessions other_sessions
  while true; do
    sessions="\$(tmux list-sessions -F '#S' 2>/dev/null || true)"
    if [[ -n $(printf "%q" "$SESSION_TO_WATCH") ]]; then
      if ! tmux has-session -t $(printf "%q" "$SESSION_TO_WATCH") 2>/dev/null; then
        echo "Session $(printf "%q" "$SESSION_TO_WATCH") ended at \$(date)" | tee -a $(printf "%q" "$watch_log")
        break
      fi
    else
      other_sessions="\$(printf '%s\n' "\$sessions" | grep -vx $(printf "%q" "$WATCH_SESSION_NAME") || true)"
      if [[ -z "\$other_sessions" ]]; then
        echo "No tmux sessions remain besides watcher at \$(date)" | tee -a $(printf "%q" "$watch_log")
        break
      fi
    fi
    sleep $(printf "%q" "$POLL_SECONDS")
  done
}
watch_any
echo "Powering off now..." | tee -a $(printf "%q" "$watch_log")
sync
poweroff_now
EOF
  chmod +x "$runner_script"
  tmux new-session -d -s "$WATCH_SESSION_NAME" "bash '$runner_script'"
  echo "Started shutdown watcher: $WATCH_SESSION_NAME"
  echo "Watching: $SESSION_TO_WATCH"
  echo "Log: $watch_log"
}

case "$COMMAND" in
  start) start_watch ;;
  attach) tmux attach -t "$WATCH_SESSION_NAME" ;;
  status)
    if tmux has-session -t "$WATCH_SESSION_NAME" 2>/dev/null; then
      tmux list-sessions | grep "^$WATCH_SESSION_NAME:"
    else
      echo "Watcher session '$WATCH_SESSION_NAME' is not running."
      exit 1
    fi
    ;;
  stop)
    if tmux has-session -t "$WATCH_SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "$WATCH_SESSION_NAME"
      echo "Stopped watcher: $WATCH_SESSION_NAME"
    else
      echo "Watcher session '$WATCH_SESSION_NAME' is not running."
    fi
    ;;
  logs)
    if [[ -L "$LATEST_LOG" || -f "$LATEST_LOG" ]]; then
      tail -n 120 "$LATEST_LOG"
    else
      echo "No shutdown watcher log found."
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [start|attach|status|stop|logs] [session_to_watch] [watch_session_name]" >&2
    exit 2
    ;;
esac
