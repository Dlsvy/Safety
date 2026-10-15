# 빠른 미리보기(GPU): 사진/동영상 → 3D 를 3~4분 안에.
# 사용: powershell -ExecutionPolicy Bypass -File scripts\run_fast.ps1 <사진폴더|영상> [출력폴더]
# - 특징점·조밀복원: COLMAP CUDA(GPU)
# - 메쉬·텍스처: OpenMVS 일반판(CPU) — OpenMVS v2.4.0 CUDA판 ReconstructMesh 가 RTX 50(SM 12.0)에서 비정상 종료해서
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Src,
    [Parameter(Position = 1)][string]$Out = ""
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = Split-Path -Parent $PSScriptRoot
$Tp = Join-Path $Root "third_party"
New-Item -ItemType Directory -Force $Tp | Out-Null

# COLMAP (CUDA)
$colmap = Get-ChildItem "$Tp\colmap" -Recurse -Filter COLMAP.bat -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $colmap) {
    Write-Host "COLMAP(GPU) 내려받는 중..."
    Invoke-WebRequest https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip -OutFile "$Tp\colmap.zip"
    Expand-Archive "$Tp\colmap.zip" "$Tp\colmap" -Force
    $colmap = Get-ChildItem "$Tp\colmap" -Recurse -Filter COLMAP.bat | Select-Object -First 1
}

# OpenMVS 일반판(CPU)
$rm = Get-ChildItem "$Tp\openmvs_cpu" -Recurse -Filter ReconstructMesh.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $rm) {
    Write-Host "OpenMVS(CPU) 내려받는 중..."
    Invoke-WebRequest https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip -OutFile "$Tp\openmvs_cpu.zip"
    Expand-Archive "$Tp\openmvs_cpu.zip" "$Tp\openmvs_cpu" -Force
    $rm = Get-ChildItem "$Tp\openmvs_cpu" -Recurse -Filter ReconstructMesh.exe | Select-Object -First 1
}
$env:OPENMVS_BIN = $rm.DirectoryName

if (-not $Out) { $Out = Join-Path $Root ("out\" + (Split-Path -Leaf $Src) + "_fast") }
Write-Host "입력: $Src"
Write-Host "출력: $Out"
$ErrorActionPreference = "Continue"  # 파이썬·COLMAP 의 stderr 로그로 멈추지 않게
$t0 = Get-Date
Push-Location $Root
& "$Root\.venv\Scripts\python.exe" -m photo3d run $Src --profile fast --colmap-bin $colmap.FullName --out $Out 2>&1 |
    Select-String -Pattern "^\[|Error|Traceback|실패"
Pop-Location
Write-Host ("총 소요: {0:N1} 분" -f ((Get-Date) - $t0).TotalMinutes)
$glb = Join-Path $Out "site_model.glb"
if (Test-Path $glb) {
    Write-Host "결과: $glb"
    try { Start-Process $glb } catch { Write-Host "기본 3D 뷰어가 없습니다. https://gltf-viewer.donmccurdy.com 에 이 파일을 끌어다 놓으세요." }
} else {
    Write-Host "결과 파일이 없습니다 — 위 오류 줄과 $Out\work\openmvs.log 마지막 부분을 보내주세요."
}
