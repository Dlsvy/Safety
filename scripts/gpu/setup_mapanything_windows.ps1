# GPU PC(NVIDIA, Windows) — MapAnything(사진 2~15장에서 카메라 자세·깊이를 한 번에 추정) 설치
#   Set-ExecutionPolicy -Scope Process Bypass; .\scripts\gpu\setup_mapanything_windows.ps1
# 라이선스: 코드 Apache-2.0, 가중치는 facebook/map-anything-apache(Apache-2.0)를 쓴다.
# 쓰임: 사진이 적어(10장 안팎) COLMAP 이 카메라를 못 잡을 때 photo3d 가 자동으로 대신 실행 → COLMAP 형식으로 이어받음.
# CUDA 확장 빌드가 없어 VS Build Tools 는 필요 없다. 필요: NVIDIA 드라이버, Python 3.10/3.11, git
# 설치 위치: third_party\mapanything\map-anything + third_party\mapanything\venv  (photo3d 가 자동으로 찾음)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Dir = Join-Path $Root "third_party\mapanything"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
if (-not (Test-Path "$Dir\map-anything\pyproject.toml")) {
  git clone https://github.com/facebookresearch/map-anything.git "$Dir\map-anything"
}
$Venv = Join-Path $Dir "venv"
if (-not (Test-Path "$Venv\Scripts\python.exe")) {
  if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv $Venv } else { python -m venv $Venv }
}
$Py = Join-Path $Venv "Scripts\python.exe"
& $Py -m pip install --upgrade pip wheel "setuptools<82"
# PGSR·MeshSplatting 과 같은 torch(2.7.1 cu128, RTX 50 지원) — pip 캐시에서 재사용
& $Py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& $Py -m pip install -e "$Dir\map-anything[colmap]"
if ($LASTEXITCODE -ne 0) { & $Py -m pip install -e "$Dir\map-anything" }
& $Py -m pip install pycolmap trimesh
& $Py -c "import torch, mapanything; print('MapAnything 준비됨, GPU:', torch.cuda.get_device_name(0))"
Write-Host "완료. 사진이 적거나 COLMAP 이 카메라를 못 잡으면 자동으로 씁니다(첫 실행 때 가중치 내려받기)."
