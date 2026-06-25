#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$WORKDIR/scripts/tmux_extract_depth_robust_5workers.sh" "$@"
