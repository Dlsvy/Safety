@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem 이미 켜져 있는 예전 앱(포트 8765)을 끈다 — 창을 여러 번 열면 예전 버전이 계속 응답하는 문제 방지
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8765" ^| findstr "LISTENING"') do taskkill /F /T /PID %%p >nul 2>&1
echo 최신 버전 받는 중...
git pull
if errorlevel 1 (
  echo.
  echo [주의] 최신 버전을 받지 못했습니다. 인터넷 연결을 확인하거나 이 창 내용을 캡처해 보내주세요.
  echo        지금 있는 버전으로 계속 실행합니다.
  echo.
)
for /f %%v in ('git rev-parse --short HEAD') do echo 현재 버전: %%v
call "%~dp0scripts\app_run.bat"
exit /b
