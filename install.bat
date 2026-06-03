@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "PY_VERSION=3.10.11"
set "RUNTIME_DIR=%CD%\runtime"
set "PY_DIR=%RUNTIME_DIR%\python"
set "PY_EXE=%PY_DIR%\python.exe"
set "VENV_DIR=%CD%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "BASE_PY="
set "PY_INSTALLER=%RUNTIME_DIR%\python-%PY_VERSION%-amd64.exe"
set "PY_INSTALL_LOG=%RUNTIME_DIR%\python-install.log"
set "PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.exe"

echo ============================================================
echo NVFP4 Converter installer
echo ============================================================
echo This installs a local Python under:
echo %PY_DIR%
echo.

if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"

if exist "%PY_EXE%" (
  "%PY_EXE%" -c "import tkinter" >nul 2>nul
  if errorlevel 1 (
    echo Existing Python runtime has no tkinter. Reinstalling full local Python...
    rmdir /s /q "%PY_DIR%"
  ) else (
    echo Local Python with tkinter already found.
  )
)

if not exist "%PY_EXE%" (
  echo Downloading Python %PY_VERSION% x64 installer...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER%'"
  if errorlevel 1 (
    echo Failed to download Python installer.
    pause
    exit /b 1
  )

  echo Installing local Python with pip and tkinter...
  if not exist "%PY_DIR%" mkdir "%PY_DIR%"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$argsList=@('/quiet','InstallAllUsers=0','TargetDir=%PY_DIR%','Include_pip=1','Include_tcltk=1','Include_test=0','Include_launcher=0','AssociateFiles=0','Shortcuts=0','PrependPath=0','/log','%PY_INSTALL_LOG%'); $p=Start-Process -FilePath '%PY_INSTALLER%' -ArgumentList $argsList -Wait -PassThru; exit $p.ExitCode"
  if errorlevel 1 (
    echo Failed to install local Python.
    echo Installer log: %PY_INSTALL_LOG%
    pause
    exit /b 1
  )
)

if not exist "%PY_EXE%" (
  echo Waiting for Python executable...
  for /l %%I in (1,1,30) do (
    if exist "%PY_EXE%" goto python_ready
    timeout /t 2 /nobreak >nul
  )
)

:python_ready
if not exist "%PY_EXE%" (
  echo Local Python was not created. The Python installer may have modified an existing Python 3.10 install.
  echo Falling back to a project .venv created from an available Python 3.10...
  if exist "%VENV_PY%" (
    set "PY_EXE=%VENV_PY%"
    goto runtime_ready
  )
  if exist "%LocalAppData%\Programs\Python\Python310\python.exe" set "BASE_PY=%LocalAppData%\Programs\Python\Python310\python.exe"
  if not defined BASE_PY (
    for /f "delims=" %%P in ('py -3.10 -c "import sys; print(sys.executable)" 2^>nul') do set "BASE_PY=%%P"
  )
  if not defined BASE_PY (
    for /f "delims=" %%P in ('python -c "import sys; print(sys.executable) if sys.version_info[:2] == (3, 10) else sys.exit(1)" 2^>nul') do set "BASE_PY=%%P"
  )
  if not defined BASE_PY (
    echo Could not find Python 3.10 after installer ran.
    echo Installer log: %PY_INSTALL_LOG%
    if exist "%PY_INSTALL_LOG%" type "%PY_INSTALL_LOG%"
    pause
    exit /b 1
  )
  echo Found base Python:
  echo(!BASE_PY!
  "!BASE_PY!" -c "import tkinter"
  if errorlevel 1 (
    echo The available Python 3.10 does not have tkinter.
    pause
    exit /b 1
  )
  echo Creating project venv...
  "!BASE_PY!" -m venv "!VENV_DIR!"
  if errorlevel 1 (
    echo Failed to create project .venv.
    pause
    exit /b 1
  )
  set "PY_EXE=!VENV_PY!"
)

:runtime_ready
"%PY_EXE%" -c "import tkinter"
if errorlevel 1 (
  echo tkinter is still unavailable. The GUI cannot run without Tcl/Tk.
  pause
  exit /b 1
)

echo ============================================================
echo Upgrading packaging tools...
echo ============================================================
"%PY_EXE%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo Failed to upgrade pip/setuptools/wheel.
  pause
  exit /b 1
)

echo ============================================================
echo Installing PyTorch nightly CUDA 13/cu130...
echo This download is large and can take a while.
echo ============================================================
"%PY_EXE%" -m pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
if errorlevel 1 (
  echo PyTorch install failed.
  pause
  exit /b 1
)

echo ============================================================
echo Installing converter dependencies...
echo ============================================================
"%PY_EXE%" -m pip install --upgrade -r requirements_base.txt
if errorlevel 1 (
  echo Dependency install failed.
  pause
  exit /b 1
)

echo ============================================================
echo Environment check...
echo ============================================================
"%PY_EXE%" -u nvfp4_tool\env_check.py
if errorlevel 1 (
  echo Environment check failed.
  pause
  exit /b 1
)

echo.
echo Install complete.
echo Run run_gui.bat to start the converter.
pause
