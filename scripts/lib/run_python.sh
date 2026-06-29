#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="thesis"
WORKDIR=""

usage() {
  cat <<'USAGE'
Usage: run_python.sh [--env NAME] [--workdir PATH] -- <command...>
       run_python.sh [--env NAME] [--workdir PATH] <command...>

Runs a command with auto runtime selection:
1) conda env (if conda exists and env is available)
2) .venv fallback at WORKDIR/.venv
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
  # MediaPipe Tasks needs EGL/GLES. Rescue instances keep a private runtime in
  # the user's home so execution does not depend on /mnt/rescue system paths.
  LOCAL_MEDIAPIPE_LIB="$HOME/.local/lib/engagement-cpu"
  USING_LOCAL_MEDIAPIPE_LIB=0
  if [[ ! -e /usr/lib/x86_64-linux-gnu/libGLESv2.so.2 && -e "$LOCAL_MEDIAPIPE_LIB/libGLESv2.so.2" ]]; then
    export LD_LIBRARY_PATH="$LOCAL_MEDIAPIPE_LIB"
    USING_LOCAL_MEDIAPIPE_LIB=1
  fi
  # A mounted rescue filesystem may contain a venv created by an older Python
  # than the rescue OS. Prefer the matching interpreter from /mnt/rescue when
  # the venv's original major.minor runtime is available there.
  VENV_SITE="$(find "$WORKDIR/.venv/lib" -maxdepth 1 -type d -name 'python[0-9]*.[0-9]*' -print -quit 2>/dev/null || true)"
  if [[ -n "$VENV_SITE" ]]; then
    VENV_VERSION="${VENV_SITE##*/python}"
    RESCUE_PY="/mnt/rescue/usr/bin/python$VENV_VERSION"
    ACTIVE_VERSION="$($VENV_PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [[ "$ACTIVE_VERSION" != "$VENV_VERSION" && -x "$RESCUE_PY" ]]; then
      export VIRTUAL_ENV="$WORKDIR/.venv"
      export PATH="$WORKDIR/.venv/bin:$PATH"
      export PYTHONPATH="$VENV_SITE/site-packages${PYTHONPATH:+:$PYTHONPATH}"
      if [[ "$1" == "python" ]]; then
        shift
        exec "$RESCUE_PY" "$@"
      fi
    fi
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

echo "ERROR: No usable runtime found."
echo "- Conda env '$CONDA_ENV' not available"
echo "- Venv python not found at: $VENV_PY"
echo "Run: ./scripts/init_project.sh"
exit 1
