@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv not found. Run install_venv.bat first, or create .venv manually.
  pause
  exit /b 1
)
set PY=%CD%\.venv\Scripts\python.exe
"%PY%" -m pip install --upgrade pip setuptools wheel
"%PY%" -m pip install --upgrade -r requirements_base.txt
"%PY%" -u nvfp4_tool\env_check.py
pause
