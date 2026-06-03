@echo off
setlocal
cd /d "%~dp0"

set "PY=runtime\python\python.exe"
if not exist "%PY%" set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Python runtime not found. Run install.bat first.
  pause
  exit /b 1
)
set PYTHONUNBUFFERED=1
"%PY%" -u nvfp4_tool\gui.py
pause
