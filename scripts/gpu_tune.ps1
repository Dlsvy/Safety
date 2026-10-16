# Photo3D GPU 최적 설정 (NVIDIA + Windows) — 관리자 권한으로 실행됨. 바꾸는 것:
#  1) GPU 시간 제한(TDR) 2초 → 60초: 화면 출력 GPU 에서 긴 CUDA 계산(조밀복원)이 윈도우에 의해 강제 종료되는 것 방지
#     HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers  TdrDelay / TdrDdiDelay (DWORD, 초)  ※ 재부팅 후 적용
#  2) 윈도우 전원 계획 '고성능'
#  3) 이 앱이 쓰는 실행파일(python·COLMAP·Brush·OpenMVS)을 '고성능 GPU' 로 지정(HKCU UserGpuPreferences)
# 되돌리기: gpu_tune.ps1 -Undo
param([switch]$Undo)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$gfx = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
$pref = "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences"
$log = Join-Path $root "app_data\gpu_tune.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Say($m) { Write-Host $m; Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format s), $m) -Encoding UTF8 }

if ($Undo) {
  Remove-ItemProperty -Path $gfx -Name TdrDelay -ErrorAction SilentlyContinue
  Remove-ItemProperty -Path $gfx -Name TdrDdiDelay -ErrorAction SilentlyContinue
  powercfg /setactive SCHEME_BALANCED | Out-Null
  Say "되돌림: TDR 기본값(2초), 전원 계획 균형"
} else {
  New-ItemProperty -Path $gfx -Name TdrDelay -PropertyType DWord -Value 60 -Force | Out-Null
  New-ItemProperty -Path $gfx -Name TdrDdiDelay -PropertyType DWord -Value 60 -Force | Out-Null
  Say "TDR: TdrDelay=60, TdrDdiDelay=60 (재부팅 후 적용)"
  powercfg /setactive SCHEME_MIN | Out-Null
  Say ("전원 계획: " + (powercfg /getactivescheme))
  New-Item -Path $pref -Force | Out-Null
  $exes = @()
  $exes += Get-ChildItem -Path (Join-Path $root ".venv\Scripts") -Filter "python*.exe" -ErrorAction SilentlyContinue
  $exes += Get-ChildItem -Path (Join-Path $root "third_party") -Recurse -Include "colmap.exe","brush_app.exe","brush.exe","DensifyPointCloud.exe","ReconstructMesh.exe","TextureMesh.exe" -ErrorAction SilentlyContinue
  foreach ($e in $exes) {
    New-ItemProperty -Path $pref -Name $e.FullName -PropertyType String -Value "GpuPreference=2;" -Force | Out-Null
    Say ("고성능 GPU 지정: " + $e.FullName)
  }
}
Say "완료. TDR 변경은 PC 를 한 번 다시 시작해야 적용됩니다."
Start-Sleep -Seconds 4
