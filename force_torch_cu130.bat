@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv not found. Run install_venv.bat first.
  pause
  exit /b 1
)
set PY=%CD%\.venv\Scripts\python.exe
"%PY%" -m pip install --pre --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
"%PY%" -u nvfp4_tool\env_check.py
pause
