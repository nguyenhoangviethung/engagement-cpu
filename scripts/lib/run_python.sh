#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-thesis}"
WORKDIR=""

usage() {
  cat <<'USAGE'
Usage: run_python.sh [--env NAME] [--workdir PATH] -- <command...>
       run_python.sh [--env NAME] [--workdir PATH] <command...>

Runs a command with auto runtime selection:
1) conda env (if conda exists and env is available)
2) .venv fallback at WORKDIR/.venv
3) currently active python fallback
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(pwd)"
fi

if command -v conda >/dev/null 2>&1; then
  if conda env list 2>/dev/null | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
    exec conda run --no-capture-output -n "$CONDA_ENV" "$@"
  fi
fi

VENV_PY="$WORKDIR/.venv/bin/python"
if [[ -x "$VENV_PY" ]]; then
  # MediaPipe Tasks needs EGL/GLES. Keep a local runtime in the user's home so
  # execution stays tied to this repo/user environment.
  LOCAL_MEDIAPIPE_LIB="$HOME/.local/lib/engagement-cpu"
  USING_LOCAL_MEDIAPIPE_LIB=0
  if [[ ! -e /usr/lib/x86_64-linux-gnu/libGLESv2.so.2 && -e "$LOCAL_MEDIAPIPE_LIB/libGLESv2.so.2" ]]; then
    export LD_LIBRARY_PATH="$LOCAL_MEDIAPIPE_LIB"
    USING_LOCAL_MEDIAPIPE_LIB=1
  fi
  export PATH="$WORKDIR/.venv/bin:$PATH"
  # Avoid picking system CUDA/cuDNN first (can mismatch with PyTorch wheel bundled libs).
  if [[ "$USING_LOCAL_MEDIAPIPE_LIB" -eq 0 ]]; then
    unset LD_LIBRARY_PATH || true
  fi
  if [[ "$1" == "python" ]]; then
    shift
    exec "$VENV_PY" "$@"
  fi
  exec "$@"
fi

if command -v python >/dev/null 2>&1; then
  if [[ "$1" == "python" ]]; then
    exec "$@"
  fi
  exec "$@"
fi

echo "ERROR: No usable runtime found."
echo "- Conda env '$CONDA_ENV' not available"
echo "- Venv python not found at: $VENV_PY"
echo "- No active 'python' command found on PATH"
echo "Create one with: python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt"
exit 1
