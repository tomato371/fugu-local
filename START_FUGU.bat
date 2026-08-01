@echo off
REM ==============================================================
REM  Fugu Local - double-click entry point.
REM  This only launches fugu_launcher.py (stdlib only); the menu
REM  itself, in Japanese, lives there.
REM  NOTE: keep this file ASCII + CRLF so cmd.exe parses it safely.
REM ==============================================================
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
cd /d "%~dp0"

set "FUGU_PY="
REM Prefer the python on PATH (that is where optional deps like gradio live),
REM and fall back to the py launcher.
python --version >nul 2>&1 && set "FUGU_PY=python"
if not defined FUGU_PY (
    py -3 --version >nul 2>&1 && set "FUGU_PY=py -3"
)
if not defined FUGU_PY (
    echo.
    echo  Python not found.
    echo  Install Python 3.10+ from https://www.python.org/downloads/
    echo  and tick "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

%FUGU_PY% fugu_launcher.py %*
set "FUGU_RC=%ERRORLEVEL%"

echo.
if not "%FUGU_RC%"=="0" echo (exit code %FUGU_RC%)
pause
exit /b %FUGU_RC%
