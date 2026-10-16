@echo off
chcp 65001 >nul
rem NVIDIA GPU 최적 설정 — 관리자 권한 확인 창이 뜨면 '예'. 되돌리기: GPU_TUNE.bat undo
cd /d "%~dp0"
if /i "%1"=="undo" (set ARG=-Undo) else (set ARG=)
powershell -NoProfile -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"%~dp0scripts\gpu_tune.ps1\" %ARG%'"
