#!/usr/bin/env bash
# 클라우드/CPU 에서 가우시안 스플래팅 (OpenSplat, AGPL-3.0 — 사내 사용 OK, 수정·배포 시 소스 공개 의무)
# 사용: scripts/splat_cpu.sh <photo3d 출력폴더> [반복수=3000] [축소배율=4]
# 결과: <출력폴더>/splat/site_splat.ply (좌표 = site_model_zup.obj 와 같음, m)
# 빌드: OpenSplat 을 -DGPU_RUNTIME=CPU 로 빌드(README 참고). pip torch 가 CUDA 빌드면
#       -DCMAKE_PROJECT_INCLUDE_BEFORE=scripts/cuda_stub.cmake 와 Caffe2Config 의 CUDA 검사 해제가 필요했다.
set -e
OUT=$1; N=${2:-3000}; D=${3:-4}
OPENSPLAT=${OPENSPLAT:-/home/user/build/OpenSplat/build/opensplat}
TORCH_LIB=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH
DATA=$OUT/splat/data
mkdir -p "$DATA/sparse/0"
ln -sfn "$(realpath "$OUT/work/dense/images")" "$DATA/images"   # OpenSplat 은 images/ + sparse/0/ 구조를 기대
cp "$OUT"/work/dense/sparse/*.bin "$DATA/sparse/0/"
"$OPENSPLAT" "$DATA" -n "$N" -d "$D" --cpu -s 1000 -o "$OUT/splat/site_splat.ply"
echo "저장: $OUT/splat/site_splat.ply"
