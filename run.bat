@echo off
cd /d "%~dp0"

echo ========================================================
echo Starting Enterprise Contract Intelligence Platform...
echo ========================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" start.py
) else (
    python start.py
)

pause
