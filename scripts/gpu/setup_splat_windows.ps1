# GPU PC(NVIDIA, Windows) — 가우시안 스플래팅(보기용 레이어) 환경 설치
#   Set-ExecutionPolicy -Scope Process Bypass; .\scripts\gpu\setup_splat_windows.ps1
# gsplat: Apache-2.0 (상업 사용 가능). CUDA 12.x 드라이버 필요.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Venv = Join-Path $Root ".venv-splat"
if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv $Venv } else { python -m venv $Venv }
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install --upgrade pip
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $Py -m pip install gsplat
$Tp = Join-Path $Root "third_party"
New-Item -ItemType Directory -Force -Path $Tp | Out-Null
if (-not (Test-Path "$Tp\gsplat")) { git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git "$Tp\gsplat" }
& $Py -m pip install -r "$Tp\gsplat\examples\requirements.txt"
Write-Host "완료. 학습: $Py scripts\gpu\train_splat.py <photo3d 출력폴더>"
