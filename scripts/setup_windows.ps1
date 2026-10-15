# photo3d 윈도우 설치 (NVIDIA GPU 권장). PowerShell 에서:
#   Set-ExecutionPolicy -Scope Process Bypass; .\scripts\setup_windows.ps1
# 설치 내용: Python 가상환경 + COLMAP(CUDA) + OpenMVS 공식 릴리스 바이너리 → .\third_party\
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$TP = Join-Path $Root "third_party"
New-Item -ItemType Directory -Force -Path $TP | Out-Null

function Get-Asset($repo, $pattern) {
    $rels = Invoke-RestMethod "https://api.github.com/repos/$repo/releases" -Headers @{ "User-Agent" = "photo3d" }
    foreach ($r in $rels) {
        foreach ($a in $r.assets) { if ($a.name -match $pattern) { return $a } }
    }
    throw "$repo 에서 '$pattern' 릴리스 파일을 찾지 못했습니다. 수동 설치하세요."
}

function Install-Archive($repo, $pattern, $dest) {
    if (Test-Path $dest) { Write-Host "  이미 설치됨: $dest"; return }
    $a = Get-Asset $repo $pattern
    $file = Join-Path $TP $a.name
    Write-Host "  다운로드: $($a.name)"
    Invoke-WebRequest $a.browser_download_url -OutFile $file
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    tar -xf $file -C $dest   # Windows 10+ 내장 tar 는 zip/7z 를 푼다
}

Write-Host "[1/4] GPU 확인"
try { nvidia-smi --query-gpu=name,memory.total --format=csv,noheader } catch { Write-Warning "NVIDIA GPU 없음 — CPU 로 동작(느림)" }

Write-Host "[2/4] Python 가상환경"
if (Get-Command py -ErrorAction SilentlyContinue) { $py = "py -3" } else { $py = "python" }
Invoke-Expression "$py -m venv `"$Root\.venv`""
& "$Root\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Root\.venv\Scripts\python.exe" -m pip install -r "$Root\requirements.txt"

Write-Host "[3/4] COLMAP (CUDA)"
Install-Archive "colmap/colmap" "windows.*cuda.*\.zip$" (Join-Path $TP "colmap")
$colmap = Get-ChildItem (Join-Path $TP "colmap") -Recurse -Include "COLMAP.bat","colmap.exe" | Select-Object -First 1

Write-Host "[4/4] OpenMVS"
Install-Archive "cdcseacave/openMVS" "(?i)win.*\.(zip|7z)$" (Join-Path $TP "openmvs")
$dens = Get-ChildItem (Join-Path $TP "openmvs") -Recurse -Filter "DensifyPointCloud.exe" | Select-Object -First 1

@"
`$env:COLMAP_BIN = "$($colmap.FullName)"
`$env:OPENMVS_BIN = "$($dens.DirectoryName)"
"@ | Set-Content (Join-Path $Root "scripts\env.ps1") -Encoding UTF8
Write-Host "완료. COLMAP=$($colmap.FullName)  OpenMVS=$($dens.DirectoryName)"
Write-Host "실행 예: .\scripts\run_windows.ps1 D:\현장\영상.mp4 -MarkerSize 0.25 -Out D:\현장\out"
