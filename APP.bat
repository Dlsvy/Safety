@echo off
chcp 65001 >nul
cd /d "%~dp0"
git pull -q
echo Photo3D app: http://localhost:8765  (keep this window open)
".venv\Scripts\python.exe" -m photo3d.app
pause
