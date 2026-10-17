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
& $Py -m pip install --upgrade pip wheel setuptools ninja
# RTX 50(Blackwell, sm_120)은 PyTorch 2.7+ cu128 이 있어야 돈다. 그 아래 카드도 cu128 로 문제없음
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# pytorch3d 는 윈도우 설치가 어려워 빼고(photo3d 가 실행 전에 그 함수 하나를 직접 넣어 줌) 나머지만
& $Py -m pip install open3d plyfile opencv-python lpips trimesh tqdm
# CUDA 확장 컴파일: 'x64 Native Tools' 환경 변수가 필요 → vcvars64 를 불러 같은 창에서 빌드
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
if (-not $vs) { throw "Visual Studio Build Tools(C++)가 없습니다. 설치 후 다시 실행하세요." }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
if (-not $env:CUDA_PATH -or -not (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
  throw "CUDA Toolkit(nvcc)이 없습니다. CUDA 12.8 설치 후 PowerShell 을 새로 열고 다시 실행하세요."
}
Write-Host "CUDA: $env:CUDA_PATH"
$env:TORCH_CUDA_ARCH_LIST = "7.5;8.6;8.9;12.0"
$env:DISTUTILS_USE_SDK = "1"
foreach ($m in @("diff-plane-rasterization", "simple-knn")) {
  cmd /c "`"$vcvars`" && `"$Py`" -m pip install --no-build-isolation `"$Dir\PGSR\submodules\$m`""
  if ($LASTEXITCODE -ne 0) { throw "$m 컴파일 실패 — CUDA Toolkit 12.8 과 VS Build Tools 를 확인하세요." }
}
& $Py -c "import torch, diff_plane_rasterization, simple_knn; print('PGSR 준비됨, GPU:', torch.cuda.get_device_name(0))"
Write-Host "완료. 통합 탭에서 자동으로 씁니다(없으면 Spirula 메쉬로 대신)."
