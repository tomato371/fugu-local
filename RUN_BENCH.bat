@echo off
REM ==============================================================
REM  fugu - accuracy re-check, one click.
REM  Runs bench_recheck.py (AIME 2026, 30 problems, config sc@nim)
REM  and compares the result with the 2026-08-03 baseline.
REM
REM  Safe to double-click any time. Every problem is appended to
REM  JSONL as soon as it is graded, so closing this window or a
REM  power cut loses at most the problem in flight - just run it
REM  again and it resumes from where it stopped.
REM
REM  Pass arguments through, for example:
REM      RUN_BENCH.bat --dry-run
REM      RUN_BENCH.bat --datasets aime26,aime24,aime25 --config sc@nim
REM      RUN_BENCH.bat --report-only
REM
REM  The API key is never stored here. bench_recheck.py reads the
REM  user environment variable NVIDIA_API_KEY and prints a full
REM  Japanese explanation if it is missing.
REM
REM  NOTE: keep this file ASCII + CRLF so cmd.exe parses it safely
REM  (same rule as OPEN_FUGU.bat). All Japanese wording lives in
REM  bench_recheck.py, which prints it at runtime.
REM ==============================================================
chcp 65001 >nul
setlocal
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
cd /d "%~dp0"
title fugu accuracy re-check

set "FUGU_PY="
python --version >nul 2>&1 && set "FUGU_PY=python"
if not defined FUGU_PY (
    py -3 --version >nul 2>&1 && set "FUGU_PY=py -3"
)
if not defined FUGU_PY (
    echo.
    echo  Python not found. Install Python 3.10+ and tick
    echo  "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

set "ARGS=%*"
if "%ARGS%"=="" set "ARGS=--datasets aime26 --config sc@nim"

echo.
echo  %FUGU_PY% bench_recheck.py %ARGS%
echo.
%FUGU_PY% bench_recheck.py %ARGS%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
    echo  (exit code %RC%^)
    echo  Re-run this file to resume from the last graded problem.
)
echo.
pause
endlocal
exit /b %RC%
