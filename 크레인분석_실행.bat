@echo off
chcp 65001 > nul
title 크레인 양중분석 프로그램

echo ======================================================================
echo                    크레인 양중분석 프로그램
echo ======================================================================
echo.

REM Python이 설치되어 있는지 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo.
    echo Python 설치 방법:
    echo 1. https://www.python.org/downloads/ 접속
    echo 2. "Download Python" 클릭
    echo 3. 설치 시 "Add Python to PATH" 체크 필수!
    echo.
    pause
    exit /b
)

REM 현재 배치 파일이 있는 디렉토리로 이동
cd /d "%~dp0"

REM 프로그램 실행
python example_usage.py

echo.
echo ======================================================================
echo                        프로그램 실행 완료
echo ======================================================================
echo.
pause
