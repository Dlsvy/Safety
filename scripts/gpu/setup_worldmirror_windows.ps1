# GPU PC(NVIDIA, Windows) — Tencent HunyuanWorld-Mirror(WorldMirror) 설치
#   Set-ExecutionPolicy -Scope Process Bypass; .\scripts\gpu\setup_worldmirror_windows.ps1
# 사진·영상 → 카메라·깊이·법선·점군·3D 가우시안을 한 번에(ATLAS 의 복원 기능과 비슷한 공개 모델).
# 라이선스: tencent-hunyuanworld-mirror-community(EU 제외, 사용자 확인: 비상업). photo3d 는 별도 프로그램으로 실행만 한다.
# 공식 안내는 torch 2.4 + CUDA 12.4 지만 RTX 50(sm_120)은 CUDA 12.8 이 필요 → 다른 도구와 같은 torch 2.7.1 cu128.
# 설치 위치: third_party\worldmirror\HunyuanWorld-Mirror + third_party\worldmirror\venv
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Dir = Join-Path $Root "third_party\worldmirror"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
if (-not (Test-Path "$Dir\HunyuanWorld-Mirror\infer.py")) {
  git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror "$Dir\HunyuanWorld-Mirror"
}
$Venv = Join-Path $Dir "venv"
if (-not (Test-Path "$Venv\Scripts\python.exe")) {
  if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv $Venv } else { python -m venv $Venv }
}
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install --upgrade pip wheel "setuptools<82"
& $Py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# requirements.txt 에서 torch 고정·gsplat(리눅스 휠)은 빼고 설치
$req = Get-Content "$Dir\HunyuanWorld-Mirror\requirements.txt" | Where-Object { $_ -notmatch '^\s*(torch|torchvision|gsplat)\b' }
$req | Set-Content "$Dir\requirements_win.txt" -Encoding ascii
& $Py -m pip install -r "$Dir\requirements_win.txt"
& $Py -c "import torch; print('WorldMirror 준비됨(torch', torch.__version__, '), GPU:', torch.cuda.get_device_name(0))"
& $Py -m pip install onnxruntime  # infer.py 가 하늘 분리에 씀
& $Py -m pip install "pycolmap==3.12.6"  # 3.10 은 이 PC 에서 DLL 초기화 오류
