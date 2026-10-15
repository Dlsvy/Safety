#!/usr/bin/env bash
# 우분투 24.04 (클라우드/리눅스 PC). COLMAP 은 pycolmap, OpenMVS 는 소스 빌드(CPU, 약 30분).
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
sudo apt-get update
sudo apt-get install -y ffmpeg cmake build-essential git libcgal-dev libboost-all-dev libopencv-dev \
  libeigen3-dev libjpeg-dev libpng-dev libtiff-dev libglu1-mesa-dev libglew-dev libglfw3-dev
pip install -r "$ROOT/requirements.txt"
mkdir -p "$ROOT/third_party" && cd "$ROOT/third_party"
[ -d vcglib ] || git clone --depth 1 https://github.com/cdcseacave/VCG.git vcglib
[ -d openMVS ] || git clone --depth 1 -b v2.3.0 https://github.com/cdcseacave/openMVS.git openMVS
# v2.3.0 버그: formatTime 의 snprintf 크기 인자 오류 → glibc fortify 가 DensifyPointCloud 를 강제 종료
sed -i 's/_sntprintf(buf+len, 128,/_sntprintf(buf+len, 128-len,/' openMVS/libs/Common/Util.h
cmake -S openMVS -B openMVS/b -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT="$PWD/vcglib" \
  -DOpenMVS_USE_CUDA=OFF -DOpenMVS_USE_PYTHON=OFF -DCMAKE_INSTALL_PREFIX="$ROOT/third_party/openmvs"
cmake --build openMVS/b -j"$(nproc)" && cmake --install openMVS/b
echo "export OPENMVS_BIN=$ROOT/third_party/openmvs/bin/OpenMVS"
