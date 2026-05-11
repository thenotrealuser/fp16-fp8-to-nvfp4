@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv not found. Run install_venv.bat first.
  pause
  exit /b 1
)
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u nvfp4_tool\gui.py
pause
