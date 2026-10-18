# GPU PC(NVIDIA, Windows) — PGSR(평면 가우시안 표면 복원, 형태 정확도 최상) 설치
#   Set-ExecutionPolicy -Scope Process Bypass; .\scripts\gpu\setup_pgsr_windows.ps1
# 라이선스: PGSR 은 연구·비상업 용도만(사용자 확인: 비상업). photo3d 는 별도 프로그램으로 실행만 한다.
# 필요한 것(먼저 설치):
#   1) NVIDIA 드라이버 최신
#   2) CUDA Toolkit 12.8 (RTX 50 시리즈는 12.8 이상 필수) — https://developer.nvidia.com/cuda-downloads
#   3) Visual Studio 2022 Build Tools 의 'C++ 데스크톱 개발' — CUDA 확장(diff-plane-rasterization·simple-knn) 컴파일용
#   4) Python 3.10 또는 3.11, git
# 설치 위치: third_party\pgsr\PGSR + third_party\pgsr\venv  (photo3d 가 자동으로 찾음)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Dir = Join-Path $Root "third_party\pgsr"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
if (-not (Test-Path "$Dir\PGSR\train.py")) {
  git clone --recursive https://github.com/zju3dv/PGSR.git "$Dir\PGSR"
}
$Venv = Join-Path $Dir "venv"
if (-not (Test-Path "$Venv\Scripts\python.exe")) {
  if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv $Venv } else { python -m venv $Venv }
}
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install --upgrade pip wheel "setuptools<82" ninja
# RTX 50(Blackwell, sm_120)은 PyTorch 2.7+ cu128 이 있어야 돈다. 그 아래 카드도 cu128 로 문제없음
# torch 2.7.1 고정: 2.8+ 헤더(compiled_autograd.h)는 이미 ::std:: 로 써도 CUDA 12.8 + VS 17.14 에서 C2872 가 남(사용자 PC 2.11 실패).
#   MeshSplatting 과 같은 2.7.1 은 torch_win_fix.py 로 빌드 성공 확인(2026-10-17). 이미 받은 파일은 pip 캐시에서 재사용
& $Py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# pytorch3d 는 윈도우 설치가 어려워 빼고(photo3d 가 실행 전에 그 함수 하나를 직접 넣어 줌) 나머지만
# open3d 0.20.0(윈도우)은 TSDF 메쉬가 비어 나온다(평평한 벽 시험도 삼각형 0개, 2026-10-18) → 0.19.0 고정
& $Py -m pip install "open3d==0.19.0" plyfile opencv-python lpips trimesh tqdm
# CUDA 확장 컴파일: 'x64 Native Tools' 환경 변수가 필요 → vcvars64 를 불러 같은 창에서 빌드
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
if (-not $vs) { throw "Visual Studio Build Tools(C++)가 없습니다. 설치 후 다시 실행하세요." }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
if (-not $env:CUDA_PATH -or -not (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
  throw "CUDA Toolkit(nvcc)이 없습니다. CUDA 12.8 설치 후 PowerShell 을 새로 열고 다시 실행하세요."
}
Write-Host "CUDA: $env:CUDA_PATH"
# PyTorch 2.6+ 윈도우 버그: nvcc 로 .cu 를 빌드하면 compiled_autograd.h 에서 "'std': 모호한 기호"(C2872) →
#   헤더의 std:: 를 ::std:: 로 바꿔 우회(scripts/gpu/torch_win_fix.py)
& $Py (Join-Path $PSScriptRoot "torch_win_fix.py")
$env:TORCH_CUDA_ARCH_LIST = "7.5;8.6;8.9;12.0"
$env:DISTUTILS_USE_SDK = "1"
foreach ($m in @("diff-plane-rasterization", "simple-knn")) {
  $log = Join-Path $Dir "build_$m.log"
  Remove-Item -Recurse -Force "$Dir\PGSR\submodules\$m\build" -ErrorAction SilentlyContinue  # 다른 torch 로 만든 중간 파일 제거
  cmd /c "`"$vcvars`" && `"$Py`" -m pip install --no-build-isolation `"$Dir\PGSR\submodules\$m`" > `"$log`" 2>&1"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "---- 컴파일 오류 줄 (전체 기록: $log) ----"
    Select-String -Path $log -Pattern "error C[0-9]+|error:|FAILED:" | Select-Object -First 25 | ForEach-Object { $_.Line }
    throw "$m 컴파일 실패 — CUDA Toolkit 12.8 과 VS Build Tools 를 확인하세요." }
}
& $Py -c "import torch, diff_plane_rasterization, simple_knn; print('PGSR 준비됨, GPU:', torch.cuda.get_device_name(0))"
Write-Host "완료. 통합 탭에서 자동으로 씁니다(없으면 Spirula 메쉬로 대신)."
