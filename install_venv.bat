@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Creating venv...
echo ============================================================
py -3.10 -m venv .venv || python -m venv .venv
if errorlevel 1 (
  echo Failed to create venv. Install Python 3.10/3.11 and retry.
  pause
  exit /b 1
)

set PY=%CD%\.venv\Scripts\python.exe

echo ============================================================
echo Upgrading pip...
echo ============================================================
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 pause

echo ============================================================
echo Installing PyTorch nightly CUDA 13/cu130...
echo This is big. If you already have a working cu130 ComfyUI Python,
echo you can cancel and use install_deps_only.bat instead.
echo ============================================================
"%PY%" -m pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
if errorlevel 1 (
  echo PyTorch install failed.
  pause
  exit /b 1
)

echo ============================================================
echo Installing converter dependencies...
echo ============================================================
"%PY%" -m pip install --upgrade -r requirements_base.txt
if errorlevel 1 (
  echo Dependency install failed.
  pause
  exit /b 1
)

echo ============================================================
echo Environment check...
echo ============================================================
"%PY%" -u nvfp4_tool\env_check.py
pause
