#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="thesis"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_ROOT="$WORKDIR/data/raw/daisee"
TARGET_DIR="$TARGET_ROOT/DAiSEE"
FORCE_CLEAN=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/data/download_daisee.sh [options]

Download DAiSEE from KaggleHub and place it at:
  data/raw/daisee/DAiSEE

Options:
  --env NAME         Conda env name (default: thesis)
  --target-dir PATH  Destination DAiSEE directory (default: data/raw/daisee/DAiSEE)
  --force-clean      Remove existing target before copying
  --dry-run          Print actions only (no download/copy)
  --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --force-clean)
      FORCE_CLEAN=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DRY-RUN] Conda env: $CONDA_ENV"
  echo "[DRY-RUN] Target dir: $TARGET_DIR"
  echo "[DRY-RUN] Force clean: $FORCE_CLEAN"
  exit 0
fi

if [[ ! -f "$HOME/.kaggle/kaggle.json" && ! -f "$HOME/.kaggle/access_token" ]]; then
  echo "ERROR: Missing Kaggle credentials."
  echo "Expected either ~/.kaggle/kaggle.json or ~/.kaggle/access_token."
  exit 1
fi

mkdir -p "$(dirname "$TARGET_DIR")"

echo "[1/4] Downloading dataset with kagglehub..."
DOWNLOAD_OUTPUT="$("$WORKDIR/scripts/lib/run_python.sh" --env "$CONDA_ENV" --workdir "$WORKDIR" python - <<'PY'
from pathlib import Path
import kagglehub

path = Path(kagglehub.dataset_download("olgaparfenova/daisee")).resolve()
print(path)
PY
)"
DOWNLOAD_PATH="$(printf '%s\n' "$DOWNLOAD_OUTPUT" | awk '/^\// { path=$0 } END { print path }')"

if [[ -z "${DOWNLOAD_PATH:-}" ]]; then
  echo "ERROR: Could not resolve Kaggle download path."
  exit 1
fi

SOURCE_DIR=""
if [[ -d "$DOWNLOAD_PATH/DAiSEE" ]]; then
  SOURCE_DIR="$DOWNLOAD_PATH/DAiSEE"
elif [[ -d "$DOWNLOAD_PATH/DataSet" && -d "$DOWNLOAD_PATH/Labels" ]]; then
  SOURCE_DIR="$DOWNLOAD_PATH"
else
  CANDIDATE="$(find "$DOWNLOAD_PATH" -maxdepth 4 -type d -name 'DAiSEE' | head -n 1 || true)"
  if [[ -n "$CANDIDATE" ]]; then
    SOURCE_DIR="$CANDIDATE"
  fi
fi

if [[ -z "$SOURCE_DIR" || ! -d "$SOURCE_DIR" ]]; then
  echo "ERROR: Could not locate DAiSEE root inside: $DOWNLOAD_PATH"
  exit 1
fi

if [[ "$FORCE_CLEAN" -eq 1 && -d "$TARGET_DIR" ]]; then
  echo "[2/4] Cleaning existing target: $TARGET_DIR"
  rm -rf "$TARGET_DIR"
fi

echo "[3/4] Copying dataset to: $TARGET_DIR"
mkdir -p "$TARGET_DIR"
rsync -a "$SOURCE_DIR/" "$TARGET_DIR/"

if [[ ! -d "$TARGET_DIR/DataSet" || ! -d "$TARGET_DIR/Labels" ]]; then
  echo "ERROR: Dataset structure check failed after copy."
  echo "Expected at least: DataSet/ and Labels/ under $TARGET_DIR"
  exit 1
fi

echo "[4/4] Done. Dataset is ready at: $TARGET_DIR"
echo "You can now run re-extraction if needed."
