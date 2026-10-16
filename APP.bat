@echo off
chcp 65001 >nul
cd /d "%~dp0"
git pull -q
call "%~dp0scripts\app_run.bat"
exit /b
