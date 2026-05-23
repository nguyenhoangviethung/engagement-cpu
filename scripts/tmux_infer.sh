#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/tmux_infer.sh <rnn|cnn|ml> [args...]"
  exit 1
fi

MODEL="$1"
shift

case "$MODEL" in
  rnn|cnn|ml) exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$MODEL/tmux_infer.sh" "$@" ;;
  *) echo "Unsupported model: $MODEL (expected: rnn|cnn|ml)"; exit 1 ;;
esac
