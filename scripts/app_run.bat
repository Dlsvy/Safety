@echo off
rem Called by APP.bat after git pull. Restarts the app if it stops (close window to quit).
cd /d "%~dp0.."
:loop
echo Photo3D app: http://localhost:8765  (keep this window open)
".venv\Scripts\python.exe" -m photo3d.app
echo App stopped. Restarting in 3 seconds... (close this window to quit)
timeout /t 3 >nul
set PHOTO3D_NO_BROWSER=1
goto loop
