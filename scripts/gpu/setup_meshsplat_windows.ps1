# GPU PC(NVIDIA, Windows) — MeshSplatting(불투명 삼각형 메쉬, CVPR 2026) 설치
#   Set-ExecutionPolicy -Scope Process Bypass; .\scripts\gpu\setup_meshsplat_windows.ps1
# 라이선스: 자체 코드 Apache-2.0, 바탕 3DGS 코드는 Inria 비상업(사용자 확인: 비상업). photo3d 는 별도 프로그램으로 실행만 한다.
# 필요한 것: PGSR 과 같음 — CUDA Toolkit 12.8, Visual Studio 2022 Build Tools(C++), Python 3.11, git
# 설치 위치: third_party\meshsplat\mesh-splatting + third_party\meshsplat\venv  (photo3d 가 자동으로 찾음)
# 저장소 README 와 다른 점: 저장소의 compile.bat 은 옛 폴더 이름(diff-convex-rasterization)을 가리켜 그대로는 안 됨 →
#   세 확장(diff-triangle-mesh-rasterization, simple-knn, effrdel)을 여기서 직접 빌드. mmcv·mmengine·timm 은 코드에서 안 써서 뺌.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Dir = Join-Path $Root "third_party\meshsplat"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
if (-not (Test-Path "$Dir\mesh-splatting\train.py")) {
  git clone --recursive https://github.com/meshsplatting/mesh-splatting.git "$Dir\mesh-splatting"
}
git -C "$Dir\mesh-splatting" submodule update --init --recursive
$Venv = Join-Path $Dir "venv"
if (-not (Test-Path "$Venv\Scripts\python.exe")) {
  if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv $Venv } else { python -m venv $Venv }
}
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install --upgrade pip wheel "setuptools<82" ninja pybind11 numpy
# 저장소 기준 torch 2.7.1 — RTX 50(sm_120)은 cu128 빌드가 필요
& $Py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& $Py -m pip install lpips plyfile mediapy opencv-python matplotlib tqdm trimesh scipy gstaichi libigl
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
& $Py (Join-Path $PSScriptRoot "meshsplat_win_fix.py") "$Dir\mesh-splatting"   # 리눅스 전용 문법(or·uint·sqrt(int)) 고침
$env:TORCH_CUDA_ARCH_LIST = "7.5;8.6;8.9;12.0"
$env:DISTUTILS_USE_SDK = "1"
# effrdel 은 src 바로 아래 모듈(tetutils·intersect)을 쓰므로 README 대로 -e(편집 설치)여야 import 된다
foreach ($m in @("diff-triangle-mesh-rasterization", "simple-knn", "effrdel")) {
  $log = Join-Path $Dir "build_$m.log"
  $src = "$Dir\mesh-splatting\submodules\$m"
  Remove-Item -Recurse -Force "$src\build" -ErrorAction SilentlyContinue
  $e = if ($m -eq "effrdel") { "-e " } else { "" }
  cmd /c "`"$vcvars`" && `"$Py`" -m pip install --no-build-isolation $e`"$src`" > `"$log`" 2>&1"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "---- 컴파일 오류 줄 (전체 기록: $log) ----"
    Select-String -Path $log -Pattern "error C[0-9]+|error:|FAILED:" | Select-Object -First 25 | ForEach-Object { $_.Line }
    throw "$m 컴파일 실패 — CUDA Toolkit 12.8 과 VS Build Tools 를 확인하세요." }
}
# 편집 설치가 src 폴더를 경로에 안 넣는 setuptools 버전도 있어 직접 추가(tetutils·intersect import 보장)
$site = & $Py -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
Set-Content -Path (Join-Path $site "photo3d_effrdel.pth") -Value "$Dir\mesh-splatting\submodules\effrdel\src" -Encoding ascii
# 선택: 학습 속도↑(fused-ssim). 실패해도 계속(없으면 저장소가 기본 SSIM 을 씀)
cmd /c "`"$vcvars`" && `"$Py`" -m pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim/"
& $Py -c "import torch, diff_triangle_rasterization, simple_knn, rdel; print('MeshSplatting 준비됨, GPU:', torch.cuda.get_device_name(0))"
Write-Host "완료. 통합 탭(기본·최고 품질)에서 자동으로 '불투명 모델'을 만듭니다."
