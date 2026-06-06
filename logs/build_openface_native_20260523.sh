#!/usr/bin/env bash
set -euo pipefail
WORKDIR="/home/bear/engagement-cpu"
OPENFACE_DIR="$WORKDIR/external/openface/OpenFace"
LOG="$WORKDIR/logs/build_openface_native_20260523.log"
cd "$WORKDIR"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== OpenFace native build started at $(date) ==="
echo "OpenFace dir: $OPENFACE_DIR"
echo "OS: $(lsb_release -ds 2>/dev/null || cat /etc/os-release | head -1)"
echo "Compiler before install: $(g++ --version | head -1 || true)"

sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config wget unzip curl git \
  libopenblas-dev liblapack-dev libgtk2.0-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev \
  libopencv-dev libdlib-dev

cd "$OPENFACE_DIR"
echo "=== Downloading OpenFace models ==="
bash ./download_models.sh

echo "=== Configuring OpenFace ==="
rm -rf build
mkdir -p build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -G Ninja ..

echo "=== Building OpenFace ==="
ninja -j1

echo "=== Build finished at $(date) ==="
ls -lh bin/FeatureExtraction || true
./bin/FeatureExtraction -help | head -80 || true
