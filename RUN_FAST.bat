@echo off
rem 더블클릭: 시험 사진(분수대)으로 3D / 사진 폴더나 영상을 이 파일 위에 끌어다 놓기: 그 입력으로 3D
cd /d "%~dp0"
git pull -q
set "SRC=%~1"
if "%SRC%"=="" set "SRC=%~dp0data\strecha\Benchmarking_Camera_Calibration_2008\fountain-P11\images"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_fast.ps1" "%SRC%"
pause
